"""ACEBench Agent runner — baseline and RPE.

ACEBench is cloned at ``external/ACEBench`` by setup_env.sh. The Agent split
evaluates multi-turn tool use in realistic dialogue. Upstream ACEBench has
its own inference + scoring pipeline; rather than fork it, this runner:

  1. Discovers Agent-category task files inside the cloned ACEBench repo using
     a small list of known relative paths. If none are found, the run is
     recorded as ``status=skipped`` with a diagnostic ``note`` so the summary
     builder still produces valid output.
  2. For each task it drives a fresh conversation against our local vLLM
     endpoint — either with a vanilla tool-calling loop (baseline) or through
     the RPE controller. Trajectories are written as JSONL for offline
     inspection.
  3. Computes lightweight internal metrics that are honest about what they
     measure: completion rate (loop exited without error), average tool-call
     count, and — where a ``ground_truth`` trace is provided in the task —
     a name-level tool-call coverage score. These are intended as diagnostic
     signals, not as a replacement for ACEBench's official scorer.

If you need the official ACEBench score, re-run upstream
``score_agent.py`` against the saved trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json
from ..common.openai_client import get_client
from ..rpe.planner import apply_decision, build_runbook, decide_next
from ..rpe.runbook import Runbook


REPO_ROOT = Path(__file__).resolve().parents[2]
ACE_REPO = REPO_ROOT / "external" / "ACEBench"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ace_runner")
    p.add_argument("--agent", required=True, choices=["baseline", "rpe"])
    p.add_argument("--model", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-num-steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def _candidate_task_paths(language: str) -> List[Path]:
    lang = language.lower()
    names = [
        f"data_all/data_{lang}/data_agent_{lang}.json",
        f"data_all/data_agent_{lang}.json",
        f"data/data_{lang}/data_agent_{lang}.json",
        f"data/data_agent_{lang}.json",
        f"data_agent_{lang}.json",
    ]
    candidates = [ACE_REPO / n for n in names]
    # Also scan shallowly in case the layout differs slightly.
    if ACE_REPO.exists():
        for p in ACE_REPO.rglob(f"data_agent_{lang}.json"):
            candidates.append(p)
    seen: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.append(p)
    return seen


def _load_tasks(language: str, limit: int) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """Return ``(tasks, source_path)``. Empty ``tasks`` means load failed."""
    for path in _candidate_task_paths(language):
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        tasks: List[Dict[str, Any]] = []
        # Accept either a JSON array or JSONL.
        if raw[0] == "[":
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    tasks = [t for t in data if isinstance(t, dict)]
            except Exception:
                tasks = []
        else:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        tasks.append(obj)
                except Exception:
                    continue
        if tasks:
            return tasks[:limit], path
    return [], None


# ---------------------------------------------------------------------------
# Task field normalization
# ---------------------------------------------------------------------------
def _extract_user_turn(task: Dict[str, Any]) -> str:
    for key in ("question", "query", "instruction", "user", "prompt"):
        v = task.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = task.get("messages") or task.get("conversation")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return str(m["content"])
    return ""


def _extract_tool_specs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize task['tools'] / task['functions'] into OpenAI tool specs."""
    raw = task.get("tools") or task.get("functions") or task.get("available_tools") or []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            out.append(item)
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not name:
            continue
        spec = {
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(fn.get("description", "")),
                "parameters": fn.get("parameters") or fn.get("params") or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        out.append(spec)
    return out


def _extract_ground_truth_tools(task: Dict[str, Any]) -> List[str]:
    """Best-effort: pull the expected sequence of tool names from the task."""
    for key in ("ground_truth", "gold", "expected", "answer", "target"):
        gt = task.get(key)
        if gt is None:
            continue
        names = _walk_for_tool_names(gt)
        if names:
            return names
    return []


def _walk_for_tool_names(obj: Any) -> List[str]:
    out: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            name = x.get("name") or x.get("tool_name") or x.get("function")
            if isinstance(name, str) and name and "args" not in name and len(name) < 80:
                parent_looks_toollike = any(k in x for k in ("arguments", "args", "parameters"))
                if parent_looks_toollike:
                    out.append(name)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


# ---------------------------------------------------------------------------
# Conversation loops
# ---------------------------------------------------------------------------
def _system_prompt_baseline(task: Dict[str, Any]) -> str:
    sys_msg = task.get("system") or task.get("system_prompt")
    if isinstance(sys_msg, str) and sys_msg.strip():
        return sys_msg.strip()
    return (
        "You are a helpful tool-using agent. Use the provided tools when needed. "
        "Make exactly one tool call at a time, wait for its result, and then decide the next step. "
        "When the user's request is resolved, reply with a short final answer."
    )


def _run_baseline(
    client,
    model: str,
    task: Dict[str, Any],
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    tools = _extract_tool_specs(task)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": _system_prompt_baseline(task)}]
    user = _extract_user_turn(task)
    if user:
        messages.append({"role": "user", "content": user})

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""

    try:
        for _ in range(max_num_steps):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
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
                tool_calls_made.append(tc.function.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps({"status": "simulated", "note": "ACEBench is evaluated offline; tool result is stubbed for driver execution."}),
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


def _run_rpe(
    client,
    model: str,
    task: Dict[str, Any],
    max_num_steps: int,
    temperature: float,
    max_escalations: int = 2,
) -> Dict[str, Any]:
    tools = _extract_tool_specs(task)
    user = _extract_user_turn(task)
    runbook = build_runbook(
        client=client,
        model=model,
        task_description=user or str(task)[:1000],
        tool_specs=tools,
        policy_text="",
        temperature=temperature,
    )

    def sys_msg() -> Dict[str, Any]:
        base = (
            "You are a tool-using agent following a runbook. Prefer the primary action for the "
            "active milestone; use the fallback only when the primary just failed or made no "
            "progress. Make one tool call at a time.\n\n"
        )
        return {"role": "system", "content": base + runbook.render_prompt_block()}

    messages: List[Dict[str, Any]] = [sys_msg()]
    if user:
        messages.append({"role": "user", "content": user})

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""

    try:
        for step in range(max_num_steps):
            messages[0] = sys_msg()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
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
                # Natural language — treat as the model indicating closure.
                decision = decide_next(client, model, runbook, assistant["content"], temperature=temperature)
                apply_decision(runbook, decision, max_escalations)
                break

            for tc in tcs:
                tool_calls_made.append(tc.function.name)
                stub = json.dumps({"status": "simulated", "note": "ACEBench tool results are evaluated offline."})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": stub,
                })
                decision = decide_next(client, model, runbook, stub, temperature=temperature)
                apply_decision(runbook, decision, max_escalations)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "rpe_runbook_final": runbook.to_dict(),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _coverage(expected: List[str], actual: List[str]) -> float:
    if not expected:
        return 0.0
    expected_set = set(expected)
    actual_set = set(actual)
    hit = sum(1 for n in expected_set if n in actual_set)
    return hit / len(expected_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ns = _parse_args()
    out_dir = ensure_dir(ns.output_dir)
    traj_path = Path(out_dir) / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    tasks, source = _load_tasks(ns.language, ns.limit)
    summary: Dict[str, Any] = {
        "benchmark": "ACEBench",
        "category": "agent",
        "agent": ns.agent,
        "model": ns.model,
        "language": ns.language,
        "source_file": str(source) if source else None,
        "config": {
            "limit": ns.limit,
            "max_num_steps": ns.max_num_steps,
            "temperature": ns.temperature,
        },
    }

    if not tasks:
        checked = [str(p) for p in _candidate_task_paths(ns.language)]
        summary["status"] = "skipped"
        summary["note"] = (
            "No ACEBench Agent task file could be located. "
            "Expected one of the paths below to exist under the cloned ACEBench repo. "
            "Re-run setup_env.sh (to clone ACEBench) or set the correct path manually."
        )
        summary["searched_paths"] = checked
        summary["metrics"] = {"num_tasks": 0}
        write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
        print(json.dumps(summary, indent=2))
        return 0

    client = get_client()
    run_fn = _run_baseline if ns.agent == "baseline" else _run_rpe

    records: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        try:
            res = run_fn(
                client=client,
                model=ns.model,
                task=task,
                max_num_steps=ns.max_num_steps,
                temperature=ns.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res = {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "messages": [],
                "tool_calls_made": [],
            }
        expected = _extract_ground_truth_tools(task)
        actual = res.get("tool_calls_made", [])
        coverage = _coverage(expected, actual)
        record = {
            "index": i,
            "task_id": task.get("id") or task.get("task_id") or i,
            "status": res.get("status"),
            "error": res.get("error"),
            "expected_tools": expected,
            "actual_tools": actual,
            "tool_coverage": coverage,
            "num_steps": sum(1 for m in res.get("messages", []) if m.get("role") == "assistant"),
            "messages": res.get("messages", []),
            "rpe_runbook_final": res.get("rpe_runbook_final"),
        }
        append_jsonl(traj_path, record)
        records.append(record)

    completed = [r for r in records if r.get("status") == "ok"]
    coverages = [r["tool_coverage"] for r in records if r.get("expected_tools")]
    metrics = {
        "num_tasks": len(records),
        "completion_rate": safe_mean([1.0 if r.get("status") == "ok" else 0.0 for r in records]),
        "avg_tool_calls": safe_mean([float(len(r.get("actual_tools", []))) for r in records]),
        "avg_steps": safe_mean([float(r.get("num_steps", 0)) for r in records]),
        "tool_name_coverage": safe_mean(coverages) if coverages else None,
        "tasks_with_ground_truth": len(coverages),
        "error_tasks": len(records) - len(completed),
    }
    summary["status"] = "ok"
    summary["metrics"] = metrics
    write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
