"""ECHO driver for the ACEBench Agent runner.

ACEBench's Agent split is scored offline against the saved trajectory:
the runner stubs every tool result with a placeholder JSON object and
what's measured is the *sequence of tool calls* the agent chooses to
issue. ECHO still applies — the same ``EchoCache`` that runs in
tau-bench prepends advisory hints to each (stubbed) observation, so
the model gets the same ``[echo:cache]`` / ``[echo:diverge]`` /
``[echo:budget]`` signals as in the live setting. This is exactly the
right shape for ACEBench: duplicate calls don't help the offline
score, and the cache hint pushes the agent to cover the full set of
*expected* tools instead of looping on one.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .cache import EchoCache


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
    "and produce a final answer."
)


def _system_prompt(task: Dict[str, Any]) -> str:
    parts: List[str] = [_ECHO_SYSTEM_BLOCK]
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        parts.append("\n--- Task system prompt ---\n" + sys_msg.strip())
    return "\n".join(parts)


def run_echo(
    *,
    client,
    model: str,
    task: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
    user_turn: str,
    system_prompt: str,
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    """Single-task ECHO loop with offline-stubbed tool observations.

    Returns the per-task record fields the ACEBench runner expects:
    ``status``, ``error``, ``messages``, ``tool_calls_made``, ``echo_stats``.
    """
    sys_blob = _system_prompt(task)
    if system_prompt and system_prompt.strip() and system_prompt.strip() not in sys_blob:
        sys_blob = sys_blob + "\n\n--- System prompt (extra) ---\n" + system_prompt.strip()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_blob}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    cache = EchoCache()
    tool_calls_made: List[str] = []
    status = "ok"
    error = ""

    try:
        for step in range(max_num_steps):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_specs or None,
                tool_choice="auto" if tool_specs else None,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            assistant: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            tcs = getattr(msg, "tool_calls", None) or []
            if tcs:
                assistant["tool_calls"] = [
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
            messages.append(assistant)

            if not tcs:
                break

            for tc in tcs:
                name = tc.function.name
                try:
                    kwargs = json.loads(tc.function.arguments or "{}")
                except Exception:
                    kwargs = {}
                tool_calls_made.append(name)
                stub = json.dumps({
                    "status": "simulated",
                    "note": "ACEBench tool results are evaluated offline.",
                })
                annotated = cache.annotate(
                    name=name,
                    args=kwargs,
                    observation=stub,
                    step=step,
                    max_num_steps=max_num_steps,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": annotated,
                })
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "echo_stats": cache.snapshot(),
    }
