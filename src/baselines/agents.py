"""Three baseline tau-bench agents sharing a single tool-calling loop.

Putting these in one module is intentional: the only difference between
``ToolCallingAgent``, ``ActAgent`` and ``ReActAgent`` is the system prompt.
Sharing the loop guarantees that any performance gap between them is due to
the prompting strategy rather than divergent control flow.

The loop is deliberately simple: one ``client.chat.completions.create`` call
per step, dispatch every emitted tool call to ``env.step``, and route any
natural-language assistant turn through ``env.step(name="respond", ...)`` —
the same convention tau-bench's reference agent uses. There is no planner,
no supervisor, no gate.

Robustness: context-window overflows from either the
chat completion or ``env.step`` are caught and either truncated-and-retried
or recorded as a step error so the loop exits cleanly.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client


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


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tcs
        ]
    return out


class _BaseInProcessAgent(Agent):
    """Shared loop for all three baselines.

    Subclasses only override ``_style_block`` to inject their prompting style.
    """

    style_name: str = "vanilla"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
    ) -> None:
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.client = get_client()

    # -- subclass hook --------------------------------------------------
    def _style_block(self) -> str:
        """Return the controller-specific instruction block."""
        return (
            "You are a helpful tool-using agent. Use the provided tools when needed. "
            "Make exactly one tool call at a time, wait for its result, and decide the "
            "next step. When the user's request is resolved, reply with a short final answer."
        )

    def _system_prompt(self) -> str:
        parts: List[str] = [self._style_block()]
        if self.wiki:
            parts.append("\n--- Domain policy ---\n" + self.wiki)
        return "\n".join(parts)

    # -- main loop ------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)

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
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_info or None,
                    tool_choice="auto" if self.tools_info else None,
                    temperature=self.temperature,
                )
            except Exception as exc:
                if _is_context_overflow(exc) and len(messages) > 3:
                    messages = [messages[0]] + messages[-6:]
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=self.tools_info or None,
                            tool_choice="auto" if self.tools_info else None,
                            temperature=self.temperature,
                        )
                    except Exception as exc2:
                        step_error = f"chat_completion_failed: {exc2}"
                        break
                else:
                    step_error = f"chat_completion_failed: {exc}"
                    break

            msg = resp.choices[0].message
            messages.append(_assistant_message_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        kwargs = {}
                    try:
                        env_resp = env.step(Action(name=name, kwargs=kwargs))
                    except Exception as env_exc:
                        if _is_context_overflow(env_exc):
                            step_error = f"env_step_context_overflow: {env_exc}"
                            done = True
                            break
                        raise
                    tool_obs = _obs_text(env_resp)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_obs,
                    })
                    reward = _float(getattr(env_resp, "reward", reward), reward)
                    info = getattr(env_resp, "info", info) or info
                    done = bool(getattr(env_resp, "done", False))
                    if done:
                        break
                if done:
                    break
                continue

            content = (msg.content or "").strip()
            if len(content) > RESPOND_MAX_CHARS:
                content = content[:RESPOND_MAX_CHARS] + " ..."
            try:
                env_resp = env.step(Action(name=RESPOND_TOOL_NAME, kwargs={"content": content}))
            except Exception as env_exc:
                if _is_context_overflow(env_exc):
                    step_error = f"env_respond_context_overflow: {env_exc}"
                    break
                raise
            user_reply = _obs_text(env_resp)
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", self.style_name)
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


class ToolCallingAgent(_BaseInProcessAgent):
    """Vanilla tool-calling: minimal prompt, no reasoning style enforced."""

    style_name = "tool-calling"

    def _style_block(self) -> str:
        return (
            "You are a helpful tool-using agent. Use the provided tools when "
            "needed. Make exactly one tool call at a time, wait for its result, "
            "and decide the next step. When the user's request is resolved, "
            "reply with a short final answer."
        )


class ActAgent(_BaseInProcessAgent):
    """Act-only: skip reasoning prose, emit tool calls directly.

    Mirrors the "Act" ablation from Yao et al. (ReAct, 2022). The agent is
    discouraged from emitting any natural-language commentary; the assistant
    message content stays empty whenever a tool call is appropriate.
    """

    style_name = "act"

    def _style_block(self) -> str:
        return (
            "You are a tool-using agent operating in ACT-ONLY mode.\n"
            "- Output ONLY tool calls; do NOT write reasoning, planning, or "
            "explanation in the assistant message.\n"
            "- Make exactly one tool call at a time and wait for its result.\n"
            "- Reply with natural-language ONLY when finalizing the user's request "
            "(short final answer)."
        )


class ReActAgent(_BaseInProcessAgent):
    """ReAct: one short Thought before each tool call.

    Mirrors Yao et al. (2022). The Thought lives in the assistant message
    content; the Action is the tool call. We cap reasoning to one short
    sentence so prompt growth stays bounded over a 30-step horizon.
    """

    style_name = "react"

    def _style_block(self) -> str:
        return (
            "You are a tool-using agent operating in ReAct mode (reasoning + acting).\n"
            "- Before EACH tool call, write exactly ONE short Thought sentence in your "
            "assistant message starting with 'Thought:' that explains in <=20 words why "
            "this tool is the right next step.\n"
            "- Then emit the tool call (one at a time). Wait for its observation before "
            "the next Thought.\n"
            "- Reply with a final answer (no Thought, no tool call) when the user's "
            "request is fully resolved."
        )
