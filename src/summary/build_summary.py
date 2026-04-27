"""Aggregate per-run metrics.json files into outputs/summary/{summary.json,summary.md}.

Renders a four-way comparison across the same fixed base model:

  1. baseline — vanilla tool-calling (minimal system prompt)
  2. act      — Act (Yao et al. 2022): action-only, no reasoning prose
  3. react    — ReAct (Yao et al. 2022): one-line Thought before each Action
  4. echo     — Episodic Cache + Horizon Orientation (this project's
                contribution): deterministic, non-blocking
                ``[echo:cache]`` / ``[echo:diverge]`` / ``[echo:budget]``
                advisory annotations on tool observations

Each section renders a row per metric for all four conditions. Missing runs
are reported as ``status=missing`` so the table never collapses on partial
data.

The summary also computes a ``deltas`` block per benchmark — ECHO minus the
strongest baseline on the headline metric (success rate for tau-bench,
completion rate for ACEBench) — to make the wins immediately visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.io_utils import read_json, write_json


SECTIONS: List[Tuple[str, Dict[str, str]]] = [
    ("tau-bench retail", {
        "baseline": "tau_retail_baseline",
        "act": "tau_retail_act",
        "react": "tau_retail_react",
        "echo": "tau_retail_echo",
    }),
    ("tau-bench airline", {
        "baseline": "tau_airline_baseline",
        "act": "tau_airline_act",
        "react": "tau_airline_react",
        "echo": "tau_airline_echo",
    }),
    ("ACEBench Agent", {
        "baseline": "acebench_agent_baseline",
        "act": "acebench_agent_act",
        "react": "acebench_agent_react",
        "echo": "acebench_agent_echo",
    }),
]

CONDITIONS: List[str] = ["baseline", "act", "react", "echo"]
CONDITION_LABELS: Dict[str, str] = {
    "baseline": "Vanilla TC",
    "act": "Act",
    "react": "ReAct",
    "echo": "ECHO (ours)",
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


def _rows_with_keys(label: str, by_cond: Dict[str, Dict[str, Any]],
                    rows_spec: List[Tuple[str, str, Any]]) -> List[str]:
    def cell(cond: str, key: str, fmt) -> str:
        metrics = (by_cond.get(cond, {}) or {}).get("metrics", {}) or {}
        return fmt(metrics.get(key))
    header = f"| {label} | Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(CONDITIONS))) + "|"
    rows: List[str] = []
    for key, human, fmt in rows_spec:
        row = f"| {label} | {human} | " + " | ".join(cell(c, key, fmt) for c in CONDITIONS) + " |"
        rows.append(row)
    status_row = f"| {label} | status | " + " | ".join(
        str((by_cond.get(c) or {}).get("status", "n/a")) for c in CONDITIONS
    ) + " |"
    rows.append(status_row)
    return [header, sep] + rows


def _tau_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    return _rows_with_keys(label, by_cond, [
        ("success_rate", "success rate", _pct),
        ("avg_reward", "avg reward", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
        ("error_tasks", "error tasks", lambda x: str(x) if x is not None else "n/a"),
        ("avg_trajectory_messages", "avg traj msgs", _num),
    ])


def _ace_rows(label: str, by_cond: Dict[str, Dict[str, Any]]) -> List[str]:
    return _rows_with_keys(label, by_cond, [
        ("completion_rate", "completion rate", _pct),
        ("tool_name_coverage", "tool-name coverage", _pct),
        ("avg_tool_calls", "avg tool calls", _num),
        ("avg_steps", "avg steps", _num),
        ("num_tasks", "tasks", lambda x: str(x) if x is not None else "n/a"),
    ])


def _headline_metric_key(label: str) -> str:
    return "success_rate" if label.startswith("tau") else "completion_rate"


def _strongest_baseline(by_cond: Dict[str, Dict[str, Any]], key: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (condition_name, value) for the best non-ECHO controller on ``key``."""
    best_cond: Optional[str] = None
    best_val: Optional[float] = None
    for c in ("baseline", "act", "react"):
        m = (by_cond.get(c, {}) or {}).get("metrics", {}) or {}
        v = m.get(key)
        if v is None:
            continue
        try:
            v_f = float(v)
        except Exception:
            continue
        if best_val is None or v_f > best_val:
            best_val = v_f
            best_cond = c
    return best_cond, best_val


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
        "deltas": [],
    }
    for label, subdirs in SECTIONS:
        by_cond = {cond: _load(outputs_dir, subdirs[cond]) for cond in CONDITIONS if cond in subdirs}
        summary["sections"].append({"label": label, "by_condition": by_cond})

        key = _headline_metric_key(label)
        best_cond, best_val = _strongest_baseline(by_cond, key)
        echo_metrics = (by_cond.get("echo", {}) or {}).get("metrics", {}) or {}
        echo_val = echo_metrics.get(key)
        try:
            echo_val_f = float(echo_val) if echo_val is not None else None
        except Exception:
            echo_val_f = None
        delta: Dict[str, Any] = {
            "section": label,
            "metric": key,
            "best_baseline": best_cond,
            "best_baseline_value": best_val,
            "echo_value": echo_val_f,
            "delta_vs_best_baseline": (
                echo_val_f - best_val
                if echo_val_f is not None and best_val is not None
                else None
            ),
        }
        summary["deltas"].append(delta)
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Vanilla / Act / ReAct / ECHO — Comparison Summary")
    lines.append("")
    lines.append(f"- Active model: `{summary.get('active_model') or '(unknown)'}`")
    lines.append(f"- Served name:  `{summary.get('served_name') or '(unknown)'}`")
    lines.append("")
    lines.append(
        "Same fixed base model and same in-process loop across all four conditions. "
        "Vanilla TC = minimal system prompt with native function-calling. "
        "Act = action-only, no reasoning prose (Yao et al. 2022). "
        "ReAct = one-line Thought before each Action (Yao et al. 2022). "
        "ECHO = Episodic Cache + Horizon Orientation: deterministic, non-blocking "
        "advisory annotations (`[echo:cache]`, `[echo:diverge]`, `[echo:budget]`) "
        "on every tool observation (this project's contribution)."
    )
    lines.append("")
    lines.append("## Headline deltas (ECHO vs. best of vanilla / Act / ReAct)")
    lines.append("")
    lines.append("| Benchmark | Metric | Best baseline | Best baseline value | ECHO | Δ |")
    lines.append("|---|---|---|---|---|---|")
    for d in summary.get("deltas", []):
        bb = d.get("best_baseline") or "n/a"
        bb_label = CONDITION_LABELS.get(bb, bb)
        bbv = d.get("best_baseline_value")
        egv = d.get("echo_value")
        dv = d.get("delta_vs_best_baseline")
        is_pct_metric = d.get("metric") in ("success_rate", "completion_rate", "tool_name_coverage")
        bbv_s = _pct(bbv) if is_pct_metric else _num(bbv)
        egv_s = _pct(egv) if is_pct_metric else _num(egv)
        if dv is None:
            dv_s = "n/a"
        elif is_pct_metric:
            dv_s = f"{100.0 * dv:+.1f} pp"
        else:
            dv_s = f"{dv:+.2f}"
        lines.append(f"| {d.get('section')} | {d.get('metric')} | {bb_label} | {bbv_s} | {egv_s} | {dv_s} |")
    lines.append("")
    lines.append("## Per-benchmark detail")
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
    lines.append("- All four controllers share the SAME in-process tool-calling loop "
                 "(same model, same tools, same temperature, same max-steps, same "
                 "truncation budget). The only thing that varies is the system "
                 "prompt — and, for ECHO only, the deterministic EchoCache that "
                 "annotates tool observations with advisory hints before they reach "
                 "the model.")
    lines.append("- Vanilla TC: minimal role + policy in the system prompt; native "
                 "function-calling does the rest.")
    lines.append("- Act: prompt instructs the model to emit tool calls only, no "
                 "reasoning prose.")
    lines.append("- ReAct: prompt requires one short `Thought:` line before each tool "
                 "call (capped to ~20 words to keep prompt growth bounded).")
    lines.append("- ECHO: zero extra LLM calls; non-blocking by construction. After "
                 "every dispatched tool call, the EchoCache prepends up to three "
                 "bracketed hints to the observation: `[echo:cache]` when the same "
                 "`(tool_name, canonical_args)` pair was already dispatched in this "
                 "episode, `[echo:diverge]` when the same tool name was used three "
                 "times in a row, and `[echo:budget]` at exactly 7 and 3 steps "
                 "remaining. The model still receives the full original observation; "
                 "the hints are advisory, deterministic, and domain-agnostic.")
    lines.append("- ACEBench metrics here are diagnostic. For the official score, "
                 "re-run upstream `score_agent.py` against the saved trajectories.")
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
