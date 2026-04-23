"""Runbook data structures for the Runbook Planner with Escalation (RPE).

The runbook is intentionally compact: a task goal, a handful of constraints,
and a short ordered list of milestones. Each milestone names a success
condition, the information it needs, a primary action, a fallback action, and
a blocker condition. The structure mirrors the design in the project PDF.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Milestone:
    id: str
    success_condition: str = ""
    required_info: List[str] = field(default_factory=list)
    primary_action: str = ""
    fallback_action: str = ""
    blocker_condition: str = ""


@dataclass
class Runbook:
    task_goal: str = ""
    key_constraints: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    active_idx: int = 0
    escalations: int = 0
    fallback_streak: int = 0

    def active(self) -> Optional[Milestone]:
        if 0 <= self.active_idx < len(self.milestones):
            return self.milestones[self.active_idx]
        return None

    def advance(self) -> None:
        self.active_idx += 1
        self.fallback_streak = 0

    def is_done(self) -> bool:
        return self.active_idx >= len(self.milestones)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_goal": self.task_goal,
            "key_constraints": list(self.key_constraints),
            "milestones": [asdict(m) for m in self.milestones],
            "active_idx": self.active_idx,
            "escalations": self.escalations,
            "fallback_streak": self.fallback_streak,
        }

    def render_prompt_block(self) -> str:
        am = self.active()
        lines: List[str] = ["## Current Runbook", f"Goal: {self.task_goal or '(unset)'}"]
        if self.key_constraints:
            lines.append("Constraints:")
            for c in self.key_constraints:
                lines.append(f"- {c}")
        if am is None:
            lines.append("Active milestone: (all milestones completed — finalize or respond).")
            return "\n".join(lines)
        lines.append(
            f"Active milestone ({self.active_idx + 1}/{len(self.milestones)}): {am.id}"
        )
        lines.append(f"  success_condition: {am.success_condition}")
        if am.required_info:
            lines.append(f"  required_info: {', '.join(am.required_info)}")
        lines.append(f"  primary_action: {am.primary_action}")
        if am.fallback_action:
            lines.append(f"  fallback_action: {am.fallback_action}")
        if am.blocker_condition:
            lines.append(f"  blocker_condition: {am.blocker_condition}")
        return "\n".join(lines)


def _extract_json_block(text: str) -> str:
    """Tolerant extractor — grabs the first {...} block."""
    if not text:
        return "{}"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


def parse_runbook_from_text(text: str) -> Optional[Runbook]:
    try:
        data = json.loads(_extract_json_block(text))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    ms_raw = data.get("milestones") or []
    milestones: List[Milestone] = []
    for m in ms_raw:
        if not isinstance(m, dict):
            continue
        milestones.append(
            Milestone(
                id=str(m.get("id") or f"m{len(milestones) + 1}"),
                success_condition=str(m.get("success_condition") or ""),
                required_info=[str(x) for x in (m.get("required_info") or [])],
                primary_action=str(m.get("primary_action") or ""),
                fallback_action=str(m.get("fallback_action") or ""),
                blocker_condition=str(m.get("blocker_condition") or ""),
            )
        )
    return Runbook(
        task_goal=str(data.get("task_goal") or "").strip(),
        key_constraints=[str(x) for x in (data.get("key_constraints") or [])],
        milestones=milestones,
    )


def default_runbook(task_description: str) -> Runbook:
    """Fallback runbook used when the planner LLM fails to return valid JSON."""
    return Runbook(
        task_goal=(task_description or "").strip()[:256] or "fulfill the user's request",
        key_constraints=[
            "follow the domain policy",
            "do not fabricate information",
            "stop when the user's request has been resolved",
        ],
        milestones=[
            Milestone(
                id="understand_request",
                success_condition="The user's concrete need is identified.",
                required_info=["user request"],
                primary_action="Ask a short clarifying question if any key detail is missing.",
                fallback_action="Proceed with the most likely interpretation and state it.",
                blocker_condition="User intent is ambiguous and cannot be clarified.",
            ),
            Milestone(
                id="gather_and_act",
                success_condition="Required information is gathered and the appropriate tool is invoked.",
                required_info=["tool arguments"],
                primary_action="Call the most relevant tool with well-formed arguments.",
                fallback_action="Try an alternative tool or adjust arguments.",
                blocker_condition="No available tool matches the user's need.",
            ),
            Milestone(
                id="confirm_and_close",
                success_condition="Result is communicated and the task is closed.",
                required_info=["final answer or tool output"],
                primary_action="Summarize outcome for the user and end the turn.",
                fallback_action="Acknowledge the blocker and offer a next step.",
                blocker_condition="Result cannot be obtained.",
            ),
        ],
    )
