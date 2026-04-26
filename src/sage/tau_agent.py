"""SAGE agent for tau-bench.

Drives the same tool-calling loop as the baselines but interposes the SAGE
gate between the model's emitted tool call and the dispatch to ``env.step``:

  decoded call ──► SAGE gate ──┬─► env.step  (allowed)
                               └─► structured-feedback tool message
                                   (blocked) → 1 free LLM retry

READ paths add zero overhead — the gate only blocks calls that fail one of
schema / provenance / idempotency. The gate is fully deterministic and
domain-agnostic; nothing in this file references retail- or airline-
specific tool names.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client
from .gate import GateResult, build_corpus, sage_gate


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


def _looks_like_error(s: str) -> bool:
    if not s:
        return False
    low = s.lower()
    return any(t in low for t in (
        "error", "not found", "invalid", "forbidden", "unauthorized",
        "cannot be", "could not", "does not exist",
    ))


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


def _normalize_args(args: Dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(args)


_SAGE_SYSTEM_BLOCK = (
    "You are a tool-using agent under SAGE — Schema-Anchored Grounded Execution. "
    "Before any tool call is dispatched, three deterministic checks run: "
    "(1) JSONSchema validation, (2) provenance — every identifier-shaped string "
    "argument (IDs, emails, codes) MUST literally appear in the user message, "
    "a prior tool result, the system prompt, or a schema enum, and "
    "(3) idempotency — no duplicate calls, no retry after repeated tool errors. "
    "If a call is blocked, you receive structured machine-readable feedback "
    "and ONE free retry. To avoid blocks: fetch IDs via READ tools before using "
    "them as arguments, never invent values, and check the schema for enum / "
    "type / required fields. Make exactly one tool call at a time."
)


class SageAgent(Agent):
    """Schema-Anchored Grounded Execution agent for tau-bench."""

    style_name = "sage"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "",
        max_retries_on_gate: int = 1,
    ) -> None:
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.env_hint = env_hint
        self.max_retries_on_gate = max(0, int(max_retries_on_gate))
        self.client = get_client()

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

        gate_stats: Dict[str, Any] = {
            "allowed": 0, "blocked": 0, "retries": 0,
            "checks_failed_total": {},
        }
        history: List[Tuple[str, str]] = []
        error_counts: Dict[str, int] = {}
        gate_retry_budget = self.max_retries_on_gate

        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False
        step_error: str = ""

        for step in range(max_num_steps):
            try:
                resp = self._chat(messages)
            except _ContextOverflowError as exc:
                messages = self._truncate_history(messages)
                try:
                    resp = self._chat(messages)
                except Exception as exc2:
                    step_error = f"chat_completion_failed: {exc2}"
                    break
            except Exception as exc:
                step_error = f"chat_completion_failed: {exc}"
                break

            msg = resp.choices[0].message
            messages.append(_assistant_message_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                # Pre-build corpus once per batch (cheap; reused across calls).
                corpus = build_corpus(messages, self.tools_info)
                advance_done = False
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        kwargs = {}
                    tool_spec = self._find_tool_spec(name)
                    result: GateResult = sage_gate(
                        messages=messages,
                        tool_specs=self.tools_info,
                        tool_spec=tool_spec,
                        tool_name=name,
                        args=kwargs,
                        history=history,
                        error_counts=error_counts,
                        corpus=corpus,
                    )

                    if not result.allow:
                        gate_stats["blocked"] += 1
                        for tag in result.checks_failed:
                            head = tag.split(":", 1)[0]
                            gate_stats["checks_failed_total"][head] = (
                                gate_stats["checks_failed_total"].get(head, 0) + 1
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": result.feedback,
                        })
                        if gate_retry_budget > 0:
                            gate_retry_budget -= 1
                            gate_stats["retries"] += 1
                            continue
                        # Budget exhausted: fall through and dispatch anyway
                        # so the loop never deadlocks.

                    # Dispatch.
                    gate_stats["allowed"] += 1
                    history.append((name, _normalize_args(kwargs)))
                    try:
                        env_resp = env.step(Action(name=name, kwargs=kwargs))
                    except Exception as env_exc:
                        if _is_context_overflow(env_exc):
                            step_error = f"env_step_context_overflow: {env_exc}"
                            advance_done = True
                            break
                        error_counts[name] = error_counts.get(name, 0) + 1
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps({"env_error": str(env_exc)}),
                        })
                        continue
                    obs = _obs_text(env_resp)
                    if _looks_like_error(obs):
                        error_counts[name] = error_counts.get(name, 0) + 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": obs,
                    })
                    reward = _float(getattr(env_resp, "reward", reward), reward)
                    info = getattr(env_resp, "info", info) or info
                    done = bool(getattr(env_resp, "done", False))
                    # Replenish retries after any successful dispatch.
                    gate_retry_budget = self.max_retries_on_gate
                    if done:
                        advance_done = True
                        break
                if advance_done:
                    break
                continue

            # No tool call: natural-language → respond.
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
        info.setdefault("sage_gate_stats", gate_stats)
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        parts: List[str] = [_SAGE_SYSTEM_BLOCK]
        if self.wiki:
            parts.append("\n--- Domain policy ---\n" + self.wiki[:3500])
        return "\n".join(parts)

    def _chat(self, messages: List[Dict[str, Any]]):
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_info or None,
                tool_choice="auto" if self.tools_info else None,
                temperature=self.temperature,
            )
        except Exception as exc:
            if _is_context_overflow(exc):
                raise _ContextOverflowError(str(exc)) from exc
            raise

    def _truncate_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 3:
            return messages
        return [messages[0]] + messages[-6:]

    def _find_tool_spec(self, name: str) -> Optional[Dict[str, Any]]:
        for ts in self.tools_info:
            fn = ts.get("function") if isinstance(ts, dict) else None
            if isinstance(fn, dict) and fn.get("name") == name:
                return ts
        return None


class _ContextOverflowError(RuntimeError):
    pass
