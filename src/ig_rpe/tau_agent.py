"""IG-RPE agent for tau-bench.

Drives the same tool-calling loop as the baseline but adds three things:

1. **Symbolic ledger** — records user identification, fetched orders, tool-call
   history, and user confirmations from observations.
2. **Deterministic gate** — before each WRITE tool call, validates invariants
   (user_verified, order_fetched, user_confirmed, not_duplicate, under_error_budget).
3. **Structured feedback loop** — on gate failure, the call is *not* dispatched
   to env.step; instead a machine-readable feedback message is appended and
   the LLM is given one free retry to correct itself.

READ calls bypass the gate entirely, so in the typical read-heavy prefix we
add zero overhead relative to vanilla tool calling.

Robustness: both ``client.chat.completions.create`` and ``env.step`` are
wrapped in try/except blocks that catch context-window overflows (the user
simulator can overflow on long conversations) and truncate history / respond
content to keep the loop alive.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client
from .gate import classify_tool, gate_check, summarize_tools_for_prompt
from .policy import extract_policy
from .state import Ledger


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 800  # cap payload we send to the user simulator


class IgRpeAgent(Agent):
    """Invariant-Gated Runbook Planner with Escalation."""

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "retail",
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
        self.policy = extract_policy(self.wiki, env_hint=self.env_hint)

    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)

        ledger = Ledger()
        gate_stats = {"allowed": 0, "blocked": 0, "retries": 0}

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(ledger, last_action="")},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})
            ledger.note_user_message(initial_user, turn=0)

        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False
        step_error: str = ""
        last_action_text: str = ""
        gate_retry_budget = self.max_retries_on_gate

        for step in range(max_num_steps):
            messages[0] = {
                "role": "system",
                "content": self._system_prompt(ledger, last_action=last_action_text),
            }

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
            assistant_msg = _assistant_message_dict(msg)
            messages.append(assistant_msg)
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                advance = False
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        kwargs = {}

                    allow, feedback, cls = gate_check(ledger, name, kwargs, self._find_tool_spec(name))
                    if not allow:
                        gate_stats["blocked"] += 1
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps({
                                "gate_blocked": True,
                                "feedback": feedback,
                                "classification": cls.reason,
                            }, ensure_ascii=False),
                        })
                        if gate_retry_budget > 0:
                            gate_retry_budget -= 1
                            gate_stats["retries"] += 1
                            # Don't dispatch; let the LLM try again next turn.
                            continue
                        # Budget exhausted: allow the call through but flag it.
                        allow = True

                    if allow:
                        gate_stats["allowed"] += 1
                        ledger.note_tool_call(name, kwargs, turn=step)
                        last_action_text = f"{name}({list(kwargs.keys())})"
                        try:
                            env_resp = env.step(Action(name=name, kwargs=kwargs))
                        except _ContextOverflowError as exc:
                            step_error = f"env_step_context_overflow: {exc}"
                            done = True
                            break
                        except Exception as exc:
                            tool_obs = json.dumps({"env_error": str(exc)})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": name,
                                "content": tool_obs,
                            })
                            ledger.note_tool_observation(name, kwargs, tool_obs, turn=step)
                            continue
                        tool_obs = _obs_text(env_resp)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": tool_obs,
                        })
                        ledger.note_tool_observation(name, kwargs, tool_obs, turn=step)
                        reward = _float(getattr(env_resp, "reward", reward), reward)
                        info = getattr(env_resp, "info", info) or info
                        done = bool(getattr(env_resp, "done", False))
                        # Replenish the gate-retry budget after a successful write.
                        if cls.is_write:
                            gate_retry_budget = self.max_retries_on_gate
                        if done:
                            break
                if done:
                    break
                continue

            # No tool call: natural-language message → route through env.respond.
            content = (msg.content or "").strip()
            if len(content) > RESPOND_MAX_CHARS:
                content = content[:RESPOND_MAX_CHARS] + " ..."
            try:
                env_resp = self._safe_env_step(env, Action(name=RESPOND_TOOL_NAME, kwargs={"content": content}))
            except _ContextOverflowError as exc:
                step_error = f"env_respond_context_overflow: {exc}"
                break
            user_reply = _obs_text(env_resp)
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
                ledger.note_user_message(user_reply, turn=step)
            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("igrpe_ledger_final", ledger.snapshot())
        info.setdefault("igrpe_gate_stats", gate_stats)
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    def _system_prompt(self, ledger: Ledger, last_action: str) -> str:
        parts = [
            "You are a tool-using agent operating under a domain policy. "
            "You MUST satisfy the invariant gate for every WRITE call.",
            "",
            "--- Contract (binding) ---",
            "1) Identify the user before any state-changing call "
            "(find_user_id_by_email or find_user_id_by_name_zip).",
            "2) Fetch an order via get_order_details before modifying/cancelling/returning it.",
            "3) Summarize the planned change and wait for an explicit 'yes' / 'confirm' from the user "
            "BEFORE calling any WRITE tool.",
            "4) Never call the same WRITE tool with the same arguments twice.",
            "5) If a tool errors, do not retry blindly — diagnose first.",
            "",
            "--- Policy bullets (relevant) ---",
        ]
        for bullet in self.policy.relevant(last_action or "", k=4):
            parts.append(f"- {bullet}")
        if self.tools_info:
            parts.append("")
            parts.append("--- Tool classification ---")
            parts.append(summarize_tools_for_prompt(self.tools_info))
        snap = ledger.snapshot()
        parts.append("")
        parts.append("--- Ledger (deterministic facts so far) ---")
        parts.append(
            f"user_verified={snap['user_verified']}; "
            f"orders_fetched={snap['orders_fetched'] or 'none'}; "
            f"tool_calls_made={len(snap['tool_calls'])}; "
            f"errors_by_tool={snap['errors_by_tool'] or 'none'}"
        )
        if self.wiki:
            parts.append("")
            parts.append("--- Domain policy (source) ---")
            parts.append(self.wiki[:3500])
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

    def _safe_env_step(self, env, action: Action):
        try:
            return env.step(action)
        except Exception as exc:
            if _is_context_overflow(exc):
                raise _ContextOverflowError(str(exc)) from exc
            raise

    def _truncate_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep system + last 6 messages when context overflows."""
        if len(messages) <= 3:
            return messages
        return [messages[0]] + messages[-6:]

    def _find_tool_spec(self, name: str) -> Optional[Dict[str, Any]]:
        for ts in self.tools_info:
            fn = ts.get("function") if isinstance(ts, dict) else None
            if isinstance(fn, dict) and fn.get("name") == name:
                return ts
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _ContextOverflowError(RuntimeError):
    pass


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
