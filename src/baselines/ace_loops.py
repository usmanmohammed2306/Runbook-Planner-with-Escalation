"""ACEBench loops for vanilla / Act / ReAct.

ACEBench's Agent split is scored offline against the saved trajectory: the
runner stubs every tool result with a placeholder JSON object and what's
measured is the *sequence of tool calls* the agent chooses to issue. The
three baselines therefore differ only in their system prompts; the loop is
otherwise identical.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


_VANILLA = (
    "You are a helpful tool-using agent. Use the provided tools when needed. "
    "Make exactly one tool call at a time, wait for its result, and decide "
    "the next step. When the user's request is resolved, reply with a short "
    "final answer."
)

_ACT = (
    "You are a tool-using agent operating in ACT-ONLY mode.\n"
    "- Output ONLY tool calls; do NOT write reasoning or planning in the "
    "assistant message.\n"
    "- Make exactly one tool call at a time and wait for its result.\n"
    "- Reply in natural language ONLY when finalizing the user's request."
)

_REACT = (
    "You are a tool-using agent operating in ReAct mode (reasoning + acting).\n"
    "- Before EACH tool call, write exactly ONE short Thought sentence in your "
    "assistant message starting with 'Thought:' that explains in <=20 words "
    "why this tool is the right next step.\n"
    "- Then emit the tool call (one at a time). Wait for its observation before "
    "the next Thought.\n"
    "- Reply with a final answer (no Thought, no tool call) when the user's "
    "request is fully resolved."
)


_STYLE_BLOCKS: Dict[str, str] = {
    "baseline": _VANILLA,
    "act": _ACT,
    "react": _REACT,
}


def _system_prompt(style: str, task: Dict[str, Any]) -> str:
    block = _STYLE_BLOCKS.get(style, _VANILLA)
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        return block + "\n\n--- Task system prompt ---\n" + sys_msg.strip()
    return block


def run_baseline_style(
    *,
    style: str,
    client,
    model: str,
    task: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
    user_turn: str,
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    """Generic ACEBench loop for vanilla / Act / ReAct controllers.

    Returns the per-task record fields the ACEBench runner expects:
    ``status``, ``error``, ``messages``, ``tool_calls_made``.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": _system_prompt(style, task)}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""

    try:
        for _ in range(max_num_steps):
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

            stub = json.dumps({
                "status": "simulated",
                "note": "ACEBench tool results are evaluated offline.",
            })
            for tc in tcs:
                tool_calls_made.append(tc.function.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": stub,
                })
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
    }
