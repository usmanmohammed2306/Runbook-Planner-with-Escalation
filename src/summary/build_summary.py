"""Aggregate per-run metrics.json files into outputs/summary/{summary.json,summary.md}."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.io_utils import read_json, write_json


SECTIONS = [
    ("tau-bench retail",  "tau_retail_baseline",       "tau_retail_rpe"),
    ("tau-bench airline", "tau_airline_baseline",      "tau_airline_rpe"),
    ("ACEBench Agent",    "acebench_agent_baseline",   "acebench_agent_rpe"),
]


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


def _fmt_tau_row(label: str, baseline: Dict[str, Any], rpe: Dict[str, Any]) -> List[str]:
    bm = baseline.get("metrics", {}) or {}
    rm = rpe.get("metrics", {}) or {}
    return [
        f"| {label} | success rate | {_pct(bm.get('success_rate'))} | {_pct(rm.get('success_rate'))} |",
        f"| {label} | avg reward | {_num(bm.get('avg_reward'))} | {_num(rm.get('avg_reward'))} |",
        f"| {label} | tasks | {bm.get('num_tasks', 'n/a')} | {rm.get('num_tasks', 'n/a')} |",
        f"| {label} | status | {baseline.get('status', 'n/a')} | {rpe.get('status', 'n/a')} |",
    ]


def _fmt_ace_row(label: str, baseline: Dict[str, Any], rpe: Dict[str, Any]) -> List[str]:
    bm = baseline.get("metrics", {}) or {}
    rm = rpe.get("metrics", {}) or {}
    return [
        f"| {label} | completion rate | {_pct(bm.get('completion_rate'))} | {_pct(rm.get('completion_rate'))} |",
        f"| {label} | tool-name coverage | {_pct(bm.get('tool_name_coverage'))} | {_pct(rm.get('tool_name_coverage'))} |",
        f"| {label} | avg tool calls | {_num(bm.get('avg_tool_calls'))} | {_num(rm.get('avg_tool_calls'))} |",
        f"| {label} | avg steps | {_num(bm.get('avg_steps'))} | {_num(rm.get('avg_steps'))} |",
        f"| {label} | tasks | {bm.get('num_tasks', 'n/a')} | {rm.get('num_tasks', 'n/a')} |",
        f"| {label} | status | {baseline.get('status', 'n/a')} | {rpe.get('status', 'n/a')} |",
    ]


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
    for label, base_dir, rpe_dir in SECTIONS:
        b = _load(outputs_dir, base_dir)
        r = _load(outputs_dir, rpe_dir)
        summary["sections"].append({
            "label": label,
            "baseline": b,
            "rpe": r,
        })
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RPE vs Baseline — Comparison Summary")
    lines.append("")
    lines.append(f"- Active model: `{summary.get('active_model') or '(unknown)'}`")
    lines.append(f"- Served name:  `{summary.get('served_name') or '(unknown)'}`")
    lines.append("")
    lines.append("The same fixed base model is used for both baseline (vanilla tool-calling) "
                 "and RPE (lightweight Runbook Planner with Escalation).")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Section | Metric | Baseline | RPE |")
    lines.append("|---|---|---|---|")
    for section in summary["sections"]:
        label = section["label"]
        baseline = section["baseline"]
        rpe = section["rpe"]
        if label.startswith("tau"):
            lines.extend(_fmt_tau_row(label, baseline, rpe))
        else:
            lines.extend(_fmt_ace_row(label, baseline, rpe))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for section in summary["sections"]:
        for key in ("baseline", "rpe"):
            note = section[key].get("note")
            if note:
                lines.append(f"- **{section['label']} / {key}**: {note}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- tau-bench metrics come from the upstream per-trial JSON files parsed by our runner.")
    lines.append("- ACEBench metrics in this summary are *internal diagnostic signals* (completion rate, tool-name coverage).")
    lines.append("  For the official ACEBench score, re-run upstream `score_agent.py` against the saved trajectories.")
    lines.append("- RPE adds two extra LLM calls per turn (planner up-front + decide after each observation);")
    lines.append("  wall-clock time and token usage will be higher than baseline even at matched task counts.")
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
