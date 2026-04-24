"""Unified tau-bench runner for both the vanilla baseline and RPE.

Baseline runs are delegated UNCHANGED to tau-bench's own ``run.py`` via
subprocess, so the vanilla tool-calling baseline is exactly tau-bench's
reference path (this satisfies the project constraint that the baseline stays
vanilla tool-calling).

RPE runs drive tau-bench's env loader (``tau_bench.envs.get_env``) directly
and call :class:`RpeAgent.solve` per task. We avoid tau-bench's CLI here
because its ``--agent-strategy`` choice list doesn't include ``rpe`` and its
internal ``tau_bench.run.run()`` has varied across versions.

Both paths emit JSON compatible with the summary builder: tau-bench's own
``results-*.json`` artifacts for baseline, and a matching ``results-rpe.json``
for RPE. ``metrics.json`` is written alongside in both cases.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("tau_runner")
    p.add_argument("--env", required=True, choices=["retail", "airline"])
    p.add_argument("--agent", required=True, choices=["baseline", "rpe"])
    p.add_argument("--model", required=True)
    p.add_argument("--user-model", required=True)
    p.add_argument("--model-provider", default="openai")
    p.add_argument("--user-model-provider", default="openai")
    p.add_argument("--user-strategy", default="llm")
    p.add_argument("--task-split", default="test")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=20)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-num-steps", type=int, default=30)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _collect_tau_results(output_dir: str) -> List[Dict[str, Any]]:
    """Parse tau-bench's per-trial JSON files under output_dir into a flat list."""
    results: List[Dict[str, Any]] = []
    out = Path(output_dir)
    # tau-bench baseline creates files like tool-calling-qwen-agent-*.json
    # RPE creates results-rpe.json. Accept both patterns.
    for path in sorted(out.rglob("results-*.json")) + sorted(out.rglob("results.json")) + sorted(out.rglob("*qwen-agent*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            results.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict) and "results" in data:
            items = data.get("results")
            if isinstance(items, list):
                results.extend(x for x in items if isinstance(x, dict))
    # De-dup on (trial, task_id) when possible.
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("trial"), r.get("task_id"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(r.get("reward", 0.0) or 0.0) for r in records]
    successes = [1.0 if (r.get("reward") is not None and float(r.get("reward") or 0.0) >= 1.0) else 0.0 for r in records]
    step_counts: List[float] = []
    for r in records:
        msgs = r.get("messages") or r.get("traj") or []
        if isinstance(msgs, list):
            step_counts.append(float(len(msgs)))
    info_errors = sum(1 for r in records if isinstance(r.get("info"), dict) and r["info"].get("error"))
    return {
        "num_tasks": len(records),
        "success_rate": safe_mean(successes),
        "avg_reward": safe_mean(rewards),
        "avg_trajectory_messages": safe_mean(step_counts),
        "error_tasks": info_errors,
    }


def _run_baseline_subprocess(ns: argparse.Namespace) -> int:
    """Drive the vanilla tool-calling baseline through upstream run.py.

    Using the upstream CLI keeps the baseline IDENTICAL to tau-bench's own
    reference execution path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tau_dir = repo_root / "external" / "tau-bench"
    tau_run_py = tau_dir / "run.py"
    if not tau_run_py.exists():
        raise FileNotFoundError(f"Expected tau-bench run.py at {tau_run_py}; run setup_env.sh first.")
    cmd = [
        sys.executable, "-u", str(tau_run_py),
        "--env", ns.env,
        "--agent-strategy", "tool-calling",
        "--model-provider", ns.model_provider,
        "--model", ns.model,
        "--user-model-provider", ns.user_model_provider,
        "--user-model", ns.user_model,
        "--user-strategy", ns.user_strategy,
        "--temperature", str(ns.temperature),
        "--num-trials", str(ns.num_trials),
        "--task-split", ns.task_split,
        "--start-index", str(ns.start_index),
        "--end-index", str(ns.end_index),
        "--max-concurrency", str(ns.max_concurrency),
        "--log-dir", os.path.abspath(ns.output_dir),
    ]
    print(f"[tau_runner] subprocess: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(tau_dir))


def _run_rpe_inprocess(ns: argparse.Namespace) -> None:
    """Drive RPE directly against tau-bench's env loader.

    This intentionally avoids tau-bench's CLI (whose ``--agent-strategy`` flag
    has a fixed choice list that doesn't include ``rpe``) and its internal
    ``tau_bench.run.run()`` function (whose signature varies across versions).
    Instead we iterate the task range, construct a fresh env per task via
    ``tau_bench.envs.get_env``, solve with :class:`RpeAgent`, and persist
    results in a tau-bench-compatible ``results.json`` shape so the summary
    builder works uniformly.
    """
    from tau_bench.envs import get_env  # type: ignore

    from ..rpe.tau_agent import RpeAgent

    out_dir = Path(ns.output_dir)
    ensure_dir(out_dir)
    traj_path = out_dir / "results-rpe.json"
    jsonl_path = out_dir / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()
    if jsonl_path.exists():
        jsonl_path.unlink()

    def _make_env(task_index: int):
        # tau-bench's `get_env` parameter name for the user-side provider has
        # used both ``user_provider`` and ``user_model_provider`` historically.
        kwargs_common = dict(
            env_name=ns.env,
            user_strategy=ns.user_strategy,
            user_model=ns.user_model,
            task_split=ns.task_split,
            task_index=task_index,
        )
        try:
            return get_env(user_provider=ns.user_model_provider, **kwargs_common)
        except TypeError:
            return get_env(user_model_provider=ns.user_model_provider, **kwargs_common)

    records: List[Dict[str, Any]] = []
    for task_index in range(ns.start_index, ns.end_index):
        for trial in range(ns.num_trials):
            env = _make_env(task_index)
            agent = RpeAgent(
                tools_info=getattr(env, "tools_info", []) or [],
                wiki=getattr(env, "wiki", "") or "",
                model=ns.model,
                provider=ns.model_provider,
                temperature=float(ns.temperature),
            )
            try:
                result = agent.solve(env, task_index=task_index, max_num_steps=ns.max_num_steps)
                reward = float(getattr(result, "reward", 0.0) or 0.0)
                info = getattr(result, "info", {}) or {}
                messages = getattr(result, "messages", []) or []
                total_cost = float(getattr(result, "total_cost", 0.0) or 0.0)
                status = "ok"
                err = ""
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                reward, info, messages, total_cost = 0.0, {"error": str(exc)}, [], 0.0
                status = "error"
                err = f"{exc.__class__.__name__}: {exc}"
            record = {
                "task_id": task_index,
                "trial": trial,
                "reward": reward,
                "info": info,
                "messages": messages,
                "total_cost": total_cost,
                "status": status,
                "error": err,
            }
            append_jsonl(jsonl_path, record)
            records.append(record)

    write_json(traj_path, records)


def main() -> int:
    ns = _parse_args()
    ensure_dir(ns.output_dir)

    agent_strategy = "tool-calling" if ns.agent == "baseline" else "rpe"
    status = "ok"
    error: str = ""
    try:
        if ns.agent == "baseline":
            rc = _run_baseline_subprocess(ns)
            if rc != 0:
                status = "error"
                error = f"tau-bench baseline subprocess exited with code {rc}"
        else:
            _run_rpe_inprocess(ns)
    except Exception as exc:  # noqa: BLE001 — we want any failure recorded
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"
        traceback.print_exc()

    records = _collect_tau_results(ns.output_dir)
    metrics = _compute_metrics(records)
    summary = {
        "benchmark": "tau-bench",
        "env": ns.env,
        "agent": ns.agent,
        "model": ns.model,
        "status": status,
        "error": error,
        "config": {
            "task_split": ns.task_split,
            "start_index": ns.start_index,
            "end_index": ns.end_index,
            "num_trials": ns.num_trials,
            "max_num_steps": ns.max_num_steps,
            "temperature": ns.temperature,
            "agent_strategy": agent_strategy,
        },
        "metrics": metrics,
    }
    write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
    print(json.dumps(summary["metrics"], indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
