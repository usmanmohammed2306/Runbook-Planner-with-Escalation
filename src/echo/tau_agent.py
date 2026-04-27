"""ECHO agent for tau-bench.

Drop-in replacement for ``ToolCallingAgent`` that wires the
``EchoCache`` annotator into the standard tool-calling loop. The control
flow is *byte-for-byte* identical to the baselines aside from one line:
after every ``env.step``, the tool observation is passed through
``cache.annotate(...)`` before being appended to the conversation. ECHO
adds no extra LLM calls, no branching, and never blocks a tool call.

Why mirror the baseline so closely:
    All four controllers share the same scaffolding (env reset,
    tool-call extraction, context-overflow recovery, respond branch).
    Keeping ECHO's loop a literal mirror of the baseline guarantees
    that any reward gap is caused by the cache annotations and not by
    incidental control-flow drift.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client
from .cache import EchoCache


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 800


# ---------------------------------------------------------------------------
# Helpers (kept local — exact same shape as baselines.agents to ensure the
# loop is byte-for-byte identical apart from the EchoCache hook).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_ECHO_SYSTEM_BLOCK = (
    "You are a helpful tool-using agent. Use the provided tools when needed. "
    "Make exactly one tool call at a time, wait for its result, and decide the "
    "next step. When the user's request is resolved, reply with a short final "
    "answer.\n\n"
    "Note on observation hints: tool results may begin with one or more "
    "bracketed advisories of the form `[echo:cache]`, `[echo:diverge]`, or "
    "`[echo:budget]`. These are deterministic system signals (not part of "
    "the tool's output):\n"
    "  * `[echo:cache]` means you already issued this exact call earlier — "
    "if the observation is unchanged, a different action is needed.\n"
    "  * `[echo:diverge]` means you have used the same tool three times in a "
    "row — consider a different tool or respond.\n"
    "  * `[echo:budget]` warns how many steps remain — close the task soon "
    "and produce a final answer.\n"
    "Use these hints to avoid loops and finish within the step budget."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class EchoAgent(Agent):
    """Episodic Cache + Horizon Orientation agent for tau-bench."""

    style_name = "echo"

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
        self.env_hint = env_hint  # accepted for parity with SageAgent; unused
        self.client = get_client()

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        parts: List[str] = [_ECHO_SYSTEM_BLOCK]
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

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        cache = EchoCache()
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
                    # ---- ECHO annotation (the only addition vs. baseline) ----
                    tool_obs = cache.annotate(
                        name=name,
                        args=kwargs,
                        observation=tool_obs,
                        step=step,
                        max_num_steps=max_num_steps,
                    )
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
        info.setdefault("echo_stats", cache.snapshot())
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
