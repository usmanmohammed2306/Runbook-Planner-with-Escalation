"""Aggregate per-run metrics.json files into outputs/summary/{summary.json,summary.md}.

Supports a 3-way comparison across three conditions on the same fixed base model:

  1. baseline — vanilla tool-calling (tau-bench's reference path)
  2. rpe      — lightweight Runbook Planner with Escalation
  3. igrpe    — Invariant-Gated RPE (new): deterministic gate + symbolic ledger

Each section renders rows for all available conditions; missing runs are
reported as ``status=missing`` without breaking the table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.io_utils import read_json, write_json


# (section_label, {condition: subdir})
SECTIONS: List[Tuple[str, Dict[str, str]]] = [
    ("tau-bench retail", {
        "baseline": "tau_retail_baseline",
        "rpe": "tau_retail_rpe",
        "igrpe": "tau_retail_igrpe",
    }),
    ("tau-bench airline", {
        "baseline": "tau_airline_baseline",
        "rpe": "tau_airline_rpe",
        "igrpe": "tau_airline_igrpe",
    }),
    ("ACEBench Agent", {
        "baseline": "acebench_agent_baseline",
        "rpe": "acebench_agent_rpe",
        "igrpe": "acebench_agent_igrpe",
    }),
]

CONDITIONS: List[str] = ["baseline", "rpe", "igrpe"]
CONDITION_LABELS: Dict[str, str] = {
    "baseline": "Baseline",
    "rpe": "RPE",
    "igrpe": "IG-RPE",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("build_summary")
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--active-model", default="")
    p.add_argument("--served-name", default="")
    return p.parse_args()


def _load(outputs_dir: Path, subdir: str) -> Dict[str, Any]:
    path = outputs_dir / subdir / "metrics.json"
    data = read_json(path, default=None)
    if data is None:
        return {
            "status": "missing",
            "note": f"No metrics.json found at {path}",
            "metrics": {},
        }
    return data


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "n/a"


def _num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _tau_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    def cell(cond: str, key: str, fmt) -> str:
        metrics = (by_cond.get(cond, {}) or {}).get("metrics", {}) or {}
        return fmt(metrics.get(key))
    header = f"| {label} | Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(CONDITIONS))) + "|"
    rows: List[str] = []
    for key, human, fmt in [
        ("success_rate", "success rate", _pct),
        ("avg_reward", "avg reward", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
        ("error_tasks", "error tasks", lambda x: str(x) if x is not None else "n/a"),
        ("avg_trajectory_messages", "avg traj msgs", _num),
    ]:
        row = f"| {label} | {human} | " + " | ".join(cell(c, key, fmt) for c in CONDITIONS) + " |"
        rows.append(row)
    status_row = f"| {label} | status | " + " | ".join(
        str((by_cond.get(c) or {}).get("status", "n/a")) for c in CONDITIONS
    ) + " |"
    rows.append(status_row)
    return [header, sep] + rows


def _ace_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    def cell(cond: str, key: str, fmt) -> str:
        metrics = (by_cond.get(cond, {}) or {}).get("metrics", {}) or {}
        return fmt(metrics.get(key))
    header = f"| {label} | Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(CONDITIONS))) + "|"
    rows: List[str] = []
    for key, human, fmt in [
        ("completion_rate", "completion rate", _pct),
        ("tool_name_coverage", "tool-name coverage", _pct),
        ("avg_tool_calls", "avg tool calls", _num),
        ("avg_steps", "avg steps", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
    ]:
        row = f"| {label} | {human} | " + " | ".join(cell(c, key, fmt) for c in CONDITIONS) + " |"
        rows.append(row)
    status_row = f"| {label} | status | " + " | ".join(
        str((by_cond.get(c) or {}).get("status", "n/a")) for c in CONDITIONS
    ) + " |"
    rows.append(status_row)
    return [header, sep] + rows


def build(outputs_dir: Path, active_model: str, served_name: str) -> Dict[str, Any]:
    if not active_model:
        am_file = outputs_dir / "active_model.txt"
        if am_file.exists():
            try:
                active_model = am_file.read_text(encoding="utf-8").strip()
            except Exception:
                active_model = ""
    summary: Dict[str, Any] = {
        "active_model": active_model,
        "served_name": served_name,
        "sections": [],
    }
    for label, subdirs in SECTIONS:
        by_cond = {cond: _load(outputs_dir, subdirs[cond]) for cond in CONDITIONS if cond in subdirs}
        summary["sections"].append({"label": label, "by_condition": by_cond})
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Baseline vs RPE vs IG-RPE — Comparison Summary")
    lines.append("")
    lines.append(f"- Active model: `{summary.get('active_model') or '(unknown)'}`")
    lines.append(f"- Served name:  `{summary.get('served_name') or '(unknown)'}`")
    lines.append("")
    lines.append(
        "Same fixed base model across all three conditions. Baseline = vanilla "
        "tool-calling. RPE = runbook planner with an LLM supervisor. IG-RPE = "
        "invariant-gated tool-calling with a symbolic ledger (this project's "
        "contribution)."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for section in summary["sections"]:
        label = section["label"]
        by_cond = section["by_condition"]
        lines.append(f"### {label}")
        if label.startswith("tau"):
            lines.extend(_tau_rows(label, by_cond))
        else:
            lines.extend(_ace_rows(label, by_cond))
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for section in summary["sections"]:
        for cond in CONDITIONS:
            data = (section.get("by_condition") or {}).get(cond) or {}
            note = data.get("note")
            if note:
                lines.append(f"- **{section['label']} / {CONDITION_LABELS[cond]}**: {note}")
    lines.append("")
    lines.append("## Method notes")
    lines.append("- Baseline uses tau-bench's upstream `tool-calling` agent; metrics parsed from its per-trial JSON.")
    lines.append("- RPE adds a compact runbook + per-turn supervisor; two extra LLM calls per step.")
    lines.append("- IG-RPE adds zero LLM calls on READ paths and at most one retry on a blocked WRITE; the gate is deterministic.")
    lines.append("- ACEBench metrics are internal diagnostic signals (completion rate, tool-name coverage). For the official score, re-run upstream `score_agent.py` against the saved trajectories.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir)
    summary = build(outputs_dir, args.active_model, args.served_name)

    out_dir = outputs_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
