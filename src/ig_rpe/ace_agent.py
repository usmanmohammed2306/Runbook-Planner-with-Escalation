"""IG-RPE driver for the ACEBench agent runner.

ACEBench does not expose a live environment — tools are evaluated offline
against the saved trajectory. We still benefit from the invariant gate
because it constrains when the agent chooses to *issue* a WRITE call, which
is one of the axes ACEBench scores.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .gate import gate_check, summarize_tools_for_prompt
from .policy import extract_policy
from .state import Ledger


def run_igrpe(
    client,
    model: str,
    task: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
    user_turn: str,
    system_prompt: str,
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    """Single-task IG-RPE loop with offline-stubbed tool observations.

    Returns a dict compatible with the ACEBench runner's record format:
    ``status``, ``error``, ``messages``, ``tool_calls_made``, ``igrpe_ledger_final``.
    """
    ledger = Ledger()
    gate_stats = {"allowed": 0, "blocked": 0, "retries": 0}
    policy = extract_policy(system_prompt or "", env_hint="retail")

    def sys_msg() -> Dict[str, Any]:
        parts = [
            "You are a tool-using agent under an invariant gate. "
            "WRITE calls require: user_verified, order_fetched (if referenced), "
            "user_confirmed, not_duplicate, under_error_budget.",
            "Make one tool call at a time.",
        ]
        for b in policy.relevant(user_turn, k=3):
            parts.append(f"- {b}")
        if tool_specs:
            parts.append(summarize_tools_for_prompt(tool_specs))
        if system_prompt:
            parts.append("--- Task system prompt ---")
            parts.append(system_prompt[:1500])
        return {"role": "system", "content": "\n".join(parts)}

    messages: List[Dict[str, Any]] = [sys_msg()]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})
        ledger.note_user_message(user_turn, turn=0)

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""
    gate_retry_budget = 1

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
            assistant = {"role": "assistant", "content": msg.content or ""}
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
                tool_spec = _find_spec(tool_specs, name)
                allow, feedback, cls = gate_check(ledger, name, kwargs, tool_spec)
                if not allow:
                    gate_stats["blocked"] += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps({"gate_blocked": True, "feedback": feedback}, ensure_ascii=False),
                    })
                    if gate_retry_budget > 0:
                        gate_retry_budget -= 1
                        gate_stats["retries"] += 1
                        continue
                    allow = True
                if allow:
                    gate_stats["allowed"] += 1
                    ledger.note_tool_call(name, kwargs, turn=step)
                    tool_calls_made.append(name)
                    stub = json.dumps({"status": "simulated", "note": "ACEBench offline stub."})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": stub,
                    })
                    ledger.note_tool_observation(name, kwargs, stub, turn=step)
                    if cls.is_write:
                        gate_retry_budget = 1
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "igrpe_ledger_final": ledger.snapshot(),
        "igrpe_gate_stats": gate_stats,
    }


def _find_spec(specs: List[Dict[str, Any]], name: str):
    for s in specs:
        fn = s.get("function") if isinstance(s, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return s
    return None
