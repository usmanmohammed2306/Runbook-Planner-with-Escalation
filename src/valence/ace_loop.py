"""VALENCE driver for the ACEBench Agent runner.

Same offline-stub shape as the baselines, but:
  * the LLM picks one ``action_id`` from a compact verified menu;
  * the kernel compiles + validates the choice;
  * the *translated* tool call (real tool name + handle-grounded args) is
    appended to ``messages`` and ``tool_calls_made`` so the upstream
    ACEBench scorer sees the same shape as the baseline runs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .kernel import AffordanceKernel, TINY_SYSTEM_PROMPT


_RESPOND = "respond"


def _system_prompt(task: Dict[str, Any]) -> str:
    parts: List[str] = [TINY_SYSTEM_PROMPT]
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        parts.append("\n--- Task system prompt ---\n" + sys_msg.strip())
    return "\n".join(parts)


def run_valence(
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
    """Single-task VALENCE loop with offline-stubbed tool observations."""
    sys_blob = _system_prompt(task)
    if system_prompt and system_prompt.strip() and system_prompt.strip() not in sys_blob:
        sys_blob = sys_blob + "\n\n--- System prompt (extra) ---\n" + system_prompt.strip()

    kernel = AffordanceKernel(respond_tool_name=_RESPOND)
    if user_turn:
        kernel.ingest_user_message(user_turn)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_blob}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""

    try:
        for step in range(max_num_steps):
            remaining = max_num_steps - step
            affs = kernel.build_affordances(tool_specs, remaining_steps=remaining)
            menu = kernel.render_menu(affs, k=8, remaining_steps=remaining)
            messages.append({
                "role": "user",
                "content": "[VALENCE menu]\n" + menu +
                           "\nReturn only JSON: {\"action_id\":\"...\"}.",
            })

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            raw_text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": raw_text})

            action_id = kernel.parse_choice(raw_text)
            kernel.ingest_assistant_choice(action_id or "", raw=raw_text)
            compiled = kernel.compile_action(action_id)
            if compiled is None:
                # No valid action — finalize.
                break

            vr = kernel.validate_mutation(compiled)
            if not vr.ok:
                messages.append({
                    "role": "user",
                    "content": f"[VALENCE rejected action {compiled.action_id}: {vr.reason}]",
                })
                continue

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

            if compiled.kind == "final" or compiled.tool_name == _RESPOND:
                kernel.record_execution(compiled)
                tool_calls_made.append(compiled.tool_name)
                stub = json.dumps({"status": "simulated", "note": "final."})
                messages.append({
                    "role": "tool",
                    "tool_call_id": translated_call_id,
                    "name": compiled.tool_name,
                    "content": stub,
                })
                break

            tool_calls_made.append(compiled.tool_name)
            stub_obs = {"status": "simulated",
                        "note": "ACEBench tool results are evaluated offline."}
            stub = json.dumps(stub_obs)
            messages.append({
                "role": "tool",
                "tool_call_id": translated_call_id,
                "name": compiled.tool_name,
                "content": stub,
            })
            kernel.record_execution(compiled)
            kernel.ingest_tool_result(compiled.tool_name, compiled.kwargs, stub_obs)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "valence_stats": kernel.snapshot(),
    }
