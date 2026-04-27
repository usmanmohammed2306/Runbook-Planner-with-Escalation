"""VALENCE agent for tau-bench.

Drop-in replacement for the baseline agents. The model never sees raw
mutation tool schemas as the primary interface; instead, every step it
receives a compact verified-action menu and returns a single JSON object
``{"action_id":"A1"}``. The kernel compiles that into a real benchmark
tool call (or a ``respond``), validates it, and dispatches it through
``env.step``. Translated calls are appended to the conversation in the
same shape as the baselines so trajectory consumers see real tool names.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client
from .kernel import AffordanceKernel, TINY_SYSTEM_PROMPT


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 800


def _is_context_overflow(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        ("context" in s and "length" in s)
        or "maximum context length" in s
        or "contextwindowexceeded" in s
        or exc.__class__.__name__ == "ContextWindowExceededError"
    )


def _float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_text(env_resp: Any) -> str:
    if env_resp is None:
        return ""
    obs = getattr(env_resp, "observation", env_resp)
    if isinstance(obs, (dict, list)):
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except Exception:
            return str(obs)
    return str(obs)


def _extract_initial_user_message(env_reset: Any) -> str:
    if env_reset is None:
        return ""
    for attr in ("observation", "content", "message", "user_message"):
        v = getattr(env_reset, attr, None)
        if v:
            return str(v)
    if isinstance(env_reset, str):
        return env_reset
    return str(env_reset)


def _obs_as_struct(text: str) -> Any:
    """Try to parse a tool observation as JSON; fall back to the raw string."""
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


class ValenceAgent(Agent):
    """LLM picks one verified action_id; the kernel compiles + validates it."""

    style_name = "valence"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "",
    ) -> None:
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.env_hint = env_hint  # accepted for runner parity; unused
        self.client = get_client()

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        parts: List[str] = [TINY_SYSTEM_PROMPT]
        if self.wiki:
            parts.append("\n--- Domain policy ---\n" + self.wiki)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)

        kernel = AffordanceKernel(respond_tool_name=RESPOND_TOOL_NAME)
        kernel.ingest_user_message(initial_user)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False
        step_error: str = ""

        for step in range(max_num_steps):
            remaining = max_num_steps - step
            affs = kernel.build_affordances(self.tools_info, remaining_steps=remaining)
            menu_text = kernel.render_menu(affs, k=8, remaining_steps=remaining)

            # Append the menu as a system-aimed user-visible block.
            messages.append({"role": "user",
                             "content": "[VALENCE menu]\n" + menu_text +
                                        "\nReturn only JSON: {\"action_id\":\"...\"}."})

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
            except Exception as exc:
                if _is_context_overflow(exc) and len(messages) > 4:
                    messages = [messages[0]] + messages[-6:]
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                        )
                    except Exception as exc2:
                        step_error = f"chat_completion_failed: {exc2}"
                        break
                else:
                    step_error = f"chat_completion_failed: {exc}"
                    break

            msg = resp.choices[0].message
            raw_text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": raw_text})

            action_id = kernel.parse_choice(raw_text)
            kernel.ingest_assistant_choice(action_id or "", raw=raw_text)
            compiled = kernel.compile_action(action_id)

            # Fail-closed fallback: if no valid action chosen, force a
            # final/respond with whatever short text the model produced.
            if compiled is None:
                content = raw_text[:RESPOND_MAX_CHARS] if raw_text else \
                    "Unable to determine a verified next step."
                try:
                    env_resp = env.step(Action(name=RESPOND_TOOL_NAME,
                                               kwargs={"content": content}))
                except Exception as env_exc:
                    if _is_context_overflow(env_exc):
                        step_error = f"env_respond_context_overflow: {env_exc}"
                        break
                    raise
                user_reply = _obs_text(env_resp)
                if user_reply:
                    messages.append({"role": "user", "content": user_reply})
                    kernel.ingest_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if done:
                    break
                continue

            # Validate (mutations gated; non-mutations pass through).
            vr = kernel.validate_mutation(compiled)
            if not vr.ok:
                # Attach the rejection as a tool-style observation so the
                # model can adapt next step.
                messages.append({
                    "role": "user",
                    "content": f"[VALENCE rejected action {compiled.action_id}: {vr.reason}]",
                })
                continue

            # Translate to a real benchmark Action and execute.
            translated_call_id = f"valence_{step}_{compiled.action_id}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": translated_call_id,
                    "type": "function",
                    "function": {
                        "name": compiled.tool_name,
                        "arguments": json.dumps(compiled.kwargs, default=str),
                    },
                }],
            })

            if compiled.kind == "final" or compiled.tool_name == RESPOND_TOOL_NAME:
                content = raw_text[:RESPOND_MAX_CHARS] if raw_text else \
                    "Final answer."
                try:
                    env_resp = env.step(Action(name=RESPOND_TOOL_NAME,
                                               kwargs={"content": content}))
                except Exception as env_exc:
                    if _is_context_overflow(env_exc):
                        step_error = f"env_respond_context_overflow: {env_exc}"
                        break
                    raise
                kernel.record_execution(compiled)
                user_reply = _obs_text(env_resp)
                messages.append({
                    "role": "tool",
                    "tool_call_id": translated_call_id,
                    "name": RESPOND_TOOL_NAME,
                    "content": user_reply,
                })
                if user_reply:
                    kernel.ingest_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if done:
                    break
                continue

            # Real tool dispatch.
            try:
                env_resp = env.step(Action(name=compiled.tool_name,
                                           kwargs=compiled.kwargs))
            except Exception as env_exc:
                if _is_context_overflow(env_exc):
                    step_error = f"env_step_context_overflow: {env_exc}"
                    done = True
                    break
                raise
            kernel.record_execution(compiled)
            tool_obs = _obs_text(env_resp)
            messages.append({
                "role": "tool",
                "tool_call_id": translated_call_id,
                "name": compiled.tool_name,
                "content": tool_obs,
            })
            kernel.ingest_tool_result(compiled.tool_name, compiled.kwargs,
                                       _obs_as_struct(tool_obs))
            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", self.style_name)
        info.setdefault("valence_stats", kernel.snapshot())
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
