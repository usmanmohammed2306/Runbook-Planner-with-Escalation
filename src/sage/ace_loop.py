"""SAGE driver for the ACEBench Agent runner.

ACEBench has no live environment — tool results are stubbed and scoring
runs offline against the saved trajectory. The SAGE gate still applies:
it constrains *which* calls the agent issues by enforcing schema and
provenance on every proposal. On ACEBench the corpus is just the system
prompt + the user turn + prior stubbed observations + schema enums —
which is exactly enough to catch made-up IDs and enum violations
(the dominant failure mode at this benchmark, particularly for 7B-class
models).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .gate import GateResult, build_corpus, sage_gate


_SAGE_SYSTEM_BLOCK = (
    "You are a tool-using agent under SAGE — Schema-Anchored Grounded Execution. "
    "Before any tool call is dispatched, three deterministic checks run: "
    "(1) JSONSchema validation, (2) provenance — every identifier-shaped string "
    "argument (IDs, emails, codes) MUST literally appear in the user message, "
    "a prior tool result, the system prompt, or a schema enum, and "
    "(3) idempotency — no duplicate calls, no retry after repeated tool errors. "
    "If a call is blocked, you receive structured machine-readable feedback "
    "and ONE free retry. To avoid blocks: never invent identifier values; "
    "use values present in the user message or schema enum, or fetch them "
    "via a READ tool first. Make exactly one tool call at a time."
)


def _system_prompt(task: Dict[str, Any]) -> str:
    parts: List[str] = [_SAGE_SYSTEM_BLOCK]
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        parts.append("\n--- Task system prompt ---\n" + sys_msg.strip())
    return "\n".join(parts)


def _normalize_args(args: Dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(args)


def _find_spec(specs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for s in specs:
        fn = s.get("function") if isinstance(s, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return s
    return None


def run_sage(
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
    """Single-task SAGE loop with offline-stubbed tool observations.

    Returns the per-task record fields the ACEBench runner expects:
    ``status``, ``error``, ``messages``, ``tool_calls_made``, ``sage_gate_stats``.
    """
    sys_blob = _system_prompt(task)
    if system_prompt and system_prompt.strip() and system_prompt.strip() not in sys_blob:
        sys_blob = sys_blob + "\n\n--- System prompt (extra) ---\n" + system_prompt.strip()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_blob}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    gate_stats: Dict[str, Any] = {
        "allowed": 0, "blocked": 0, "retries": 0,
        "checks_failed_total": {},
    }
    history: List[Tuple[str, str]] = []
    error_counts: Dict[str, int] = {}
    gate_retry_budget = 1

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

            corpus = build_corpus(messages, tool_specs)
            for tc in tcs:
                name = tc.function.name
                try:
                    kwargs = json.loads(tc.function.arguments or "{}")
                except Exception:
                    kwargs = {}
                tool_spec = _find_spec(tool_specs, name)
                result: GateResult = sage_gate(
                    messages=messages,
                    tool_specs=tool_specs,
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
                    # Budget exhausted: dispatch anyway.

                gate_stats["allowed"] += 1
                history.append((name, _normalize_args(kwargs)))
                tool_calls_made.append(name)
                stub = json.dumps({"status": "simulated", "note": "ACEBench offline stub."})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": stub,
                })
                gate_retry_budget = 1
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "sage_gate_stats": gate_stats,
    }
