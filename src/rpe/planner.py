"""Planner calls for the Runbook Planner with Escalation (RPE).

Two lightweight LLM calls are exposed:
  * build_runbook()  — produce a compact runbook up-front from the task text.
  * decide_next()    — after each observation, advance / keep / replan.

Both are strictly JSON-only prompts. Parsing is tolerant; on any failure we
fall back to safe defaults so the outer execution loop keeps progressing.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .runbook import Runbook, default_runbook, parse_runbook_from_text, _extract_json_block


PLANNER_SYSTEM = (
    "You are a planning assistant for a tool-using agent.\n"
    "Produce a COMPACT runbook as JSON ONLY (no prose, no code fences).\n"
    "The runbook must have 2 to 5 milestones. Each milestone must contain:\n"
    "  id (short string), success_condition, required_info (array of strings),\n"
    "  primary_action, fallback_action, blocker_condition.\n"
    "\n"
    "Schema:\n"
    "{\n"
    '  "task_goal": "...",\n'
    '  "key_constraints": ["..."],\n'
    '  "milestones": [\n'
    '    {"id":"...","success_condition":"...","required_info":["..."],'
    '"primary_action":"...","fallback_action":"...","blocker_condition":"..."}\n'
    "  ]\n"
    "}\n"
    "Output ONLY valid JSON that matches this schema."
)


DECIDE_SYSTEM = (
    "You are the supervisor of a tool-using agent running a runbook.\n"
    "Given the current runbook and the latest observation/tool result,\n"
    "output a JSON decision ONLY (no prose, no code fences):\n"
    "{\n"
    '  "action": "advance" | "keep" | "replan",\n'
    '  "use_fallback": true | false,\n'
    '  "note": "...",\n'
    '  "runbook": { ...full runbook object... }   // REQUIRED iff action == "replan"\n'
    "}\n"
    "Rules:\n"
    "- advance:  the active milestone's success_condition is now satisfied.\n"
    "- keep:     milestone not yet met; continue. Set use_fallback=true if the primary\n"
    "            action just failed or made no progress and the fallback_action should be tried next.\n"
    "- replan:   two or more successive failures, repeated no-progress, or a blocker was hit —\n"
    "            produce a fresh, smaller runbook.\n"
    "Escalate to replan sparingly. Default to 'keep'."
)


def _summarize_tools(tool_specs: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for t in tool_specs or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict):
            n = fn.get("name")
            if n:
                names.append(str(n))
    return ", ".join(names) if names else "(none)"


def _clip(text: str, n: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= n else text[:n] + " ...[truncated]"


def build_runbook(
    client,
    model: str,
    task_description: str,
    tool_specs: List[Dict[str, Any]],
    policy_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 768,
) -> Runbook:
    user = (
        f"Task description:\n{_clip(task_description, 4000)}\n\n"
        f"Available tools: {_summarize_tools(tool_specs)}\n\n"
    )
    if policy_text:
        user += f"Domain policy (summary, may be clipped):\n{_clip(policy_text, 4000)}\n\n"
    user += "Produce the runbook JSON now."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return default_runbook(task_description)

    rb = parse_runbook_from_text(raw)
    if rb is None or not rb.milestones:
        return default_runbook(task_description)
    return rb


def decide_next(
    client,
    model: str,
    runbook: Runbook,
    observation: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    user = (
        "Current runbook:\n"
        f"{json.dumps(runbook.to_dict(), indent=2)}\n\n"
        "Latest observation:\n"
        f"{_clip(observation, 2000)}\n\n"
        "Return JSON decision now."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DECIDE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {"action": "keep", "use_fallback": False, "note": "decide-exception"}

    try:
        data = json.loads(_extract_json_block(raw))
    except Exception:
        return {"action": "keep", "use_fallback": False, "note": "decide-parse-fail"}

    if not isinstance(data, dict):
        return {"action": "keep", "use_fallback": False, "note": "decide-not-dict"}
    action = str(data.get("action", "keep")).lower()
    if action not in ("advance", "keep", "replan"):
        action = "keep"
    data["action"] = action
    data["use_fallback"] = bool(data.get("use_fallback", False))
    return data


def apply_decision(
    runbook: Runbook,
    decision: Dict[str, Any],
    max_escalations: int,
) -> Runbook:
    """Mutate ``runbook`` according to ``decision`` and return it.

    On ``replan`` we install the new milestone list but guard against runaway
    escalation. The fallback-streak counter powers the 'two successive failures
    => replan' heuristic described in the PDF.
    """
    action = decision.get("action", "keep")
    if action == "advance":
        runbook.advance()
        return runbook

    if action == "replan":
        if runbook.escalations >= max_escalations:
            # Respect the escalation cap; just keep making progress.
            runbook.fallback_streak = 0
            return runbook
        new = decision.get("runbook")
        if isinstance(new, dict):
            parsed = parse_runbook_from_text(json.dumps(new))
            if parsed and parsed.milestones:
                runbook.task_goal = parsed.task_goal or runbook.task_goal
                if parsed.key_constraints:
                    runbook.key_constraints = parsed.key_constraints
                runbook.milestones = parsed.milestones
                runbook.active_idx = 0
                runbook.escalations += 1
                runbook.fallback_streak = 0
                return runbook
        # Invalid replan payload: fall through to keep.
    # keep
    if decision.get("use_fallback"):
        runbook.fallback_streak += 1
    else:
        runbook.fallback_streak = 0
    return runbook
