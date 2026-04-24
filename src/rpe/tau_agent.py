"""RPE agent for tau-bench.

Implements a tau-bench-compatible :class:`Agent` subclass that drives the
same tool-calling loop as the built-in ``ToolCallingAgent`` but injects a
compact runbook into the system prompt on every turn and uses a supervisor
LLM call after each observation to decide whether to advance, keep, or
replan.

The goal here is NOT to re-invent the tool-calling loop. Baseline comparisons
remain vanilla tool-calling via tau-bench's own upstream agent; this file
only changes the *controller* around that loop.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# tau-bench imports — done lazily-friendly at module top; if they fail the
# runner will surface a clear error since RPE cannot work without them.
from tau_bench.agents.base import Agent
from tau_bench.types import Action, SolveResult  # type: ignore

from ..common.openai_client import get_client
from .planner import apply_decision, build_runbook, decide_next
from .runbook import Runbook


RESPOND_TOOL_NAME = "respond"


class RpeAgent(Agent):
    """Lightweight Runbook Planner with Escalation agent."""

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        max_escalations: int = 2,
        replan_every_steps: int = 1,
    ) -> None:
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.max_escalations = int(max_escalations)
        self.replan_every_steps = max(1, int(replan_every_steps))
        self.client = get_client()

    # ------------------------------------------------------------------
    # Public API expected by tau-bench
    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)
        task_text = _extract_task_text(env, env_reset)

        runbook = build_runbook(
            client=self.client,
            model=self.model,
            task_description=task_text,
            tool_specs=self.tools_info,
            policy_text=self.wiki,
            temperature=self.temperature,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(runbook)},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False

        step_error: str = ""
        for step in range(max_num_steps):
            # Refresh the system prompt so the live runbook is visible each turn.
            messages[0] = {"role": "system", "content": self._system_prompt(runbook)}

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_info or None,
                    tool_choice="auto" if self.tools_info else None,
                    temperature=self.temperature,
                )
            except Exception as exc:
                exc_str = str(exc)
                # Context window exceeded: drop the oldest non-system messages and retry once.
                if "context" in exc_str.lower() and "length" in exc_str.lower() and len(messages) > 3:
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
            assistant_msg = _assistant_message_dict(msg)
            messages.append(assistant_msg)

            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        kwargs = {}
                    action = Action(name=name, kwargs=kwargs)
                    env_resp = env.step(action)
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
                    if step % self.replan_every_steps == 0:
                        decision = decide_next(
                            self.client, self.model, runbook, tool_obs,
                            temperature=self.temperature,
                        )
                        apply_decision(runbook, decision, self.max_escalations)
                    if done:
                        break
                if done:
                    break
                continue

            # No tool call — treat as a user-facing message via env.respond.
            content = msg.content or ""
            action = Action(name=RESPOND_TOOL_NAME, kwargs={"content": content})
            env_resp = env.step(action)
            user_reply = _obs_text(env_resp)
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            decision = decide_next(
                self.client, self.model, runbook, user_reply or content,
                temperature=self.temperature,
            )
            apply_decision(runbook, decision, self.max_escalations)
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("rpe_runbook_final", runbook.to_dict())
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _system_prompt(self, runbook: Runbook) -> str:
        parts = [
            "You are a tool-using agent operating under a domain policy.",
            "Follow the runbook below. Prefer the primary action; use the fallback only when the primary fails or makes no progress. If both repeatedly fail, the supervisor will replan.",
        ]
        if self.wiki:
            parts.append("\nPolicy:\n" + self.wiki)
        parts.append("\n" + runbook.render_prompt_block())
        return "\n".join(parts)


# ------------------------------------------------------------------
# Local helpers (kept module-level for testability)
# ------------------------------------------------------------------
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


def _extract_task_text(env: Any, env_reset: Any) -> str:
    for attr in ("task", "instruction", "user_instruction"):
        v = getattr(env, attr, None)
        if v:
            inst = getattr(v, "instruction", None)
            return str(inst) if inst else str(v)
    return _extract_initial_user_message(env_reset)


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    """Convert an OpenAI SDK ChatCompletionMessage into a messages-array dict.

    We keep the pieces tau-bench typically serializes: role, content, tool_calls.
    """
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
