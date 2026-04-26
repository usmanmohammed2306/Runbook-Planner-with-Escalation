"""Unified tau-bench runner for the four-way comparison.

All four controllers share a single in-process loop. The only thing that
varies between them is the agent class instantiated per task:

  * ``baseline`` — :class:`baselines.ToolCallingAgent` (vanilla tool-calling
    with a minimal system prompt)
  * ``act``      — :class:`baselines.ActAgent` (no reasoning prose, action-only)
  * ``react``    — :class:`baselines.ReActAgent` (one-line Thought before each
    Action)
  * ``sage``     — :class:`sage.tau_agent.SageAgent` (Schema-Anchored Grounded
    Execution: deterministic provenance + schema gate; this project's
    contribution)

Sharing the loop guarantees the gap between conditions is due to the
controller and not divergent control flow / tool-result formatting.

The litellm-truncation sitecustomize patch is loaded in-process for every
condition because tau-bench's user simulator calls ``litellm.completion``
under the hood and can otherwise blow the model's context window on long
multi-turn dialogues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..common.io_utils import append_jsonl, ensure_dir, safe_mean, write_json


AGENT_CHOICES = ["baseline", "act", "react", "sage"]


def _try_install_litellm_patch() -> None:
    """Best-effort: load the litellm-truncation sitecustomize hook in-process."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        patch_dir = repo_root / "src" / "_taubench_patches"
        if str(patch_dir) not in sys.path:
            sys.path.insert(0, str(patch_dir))
        import sitecustomize  # noqa: F401  (executes _install on import)
    except Exception as exc:  # noqa: BLE001
        print(f"[tau_runner] litellm patch not loaded: {exc}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("tau_runner")
    p.add_argument("--env", required=True, choices=["retail", "airline"])
    p.add_argument("--agent", required=True, choices=AGENT_CHOICES)
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


def _resolve_agent_cls(kind: str):
    if kind == "baseline":
        from ..baselines.agents import ToolCallingAgent
        return ToolCallingAgent
    if kind == "act":
        from ..baselines.agents import ActAgent
        return ActAgent
    if kind == "react":
        from ..baselines.agents import ReActAgent
        return ReActAgent
    if kind == "sage":
        from ..sage.tau_agent import SageAgent
        return SageAgent
    raise ValueError(f"Unknown agent kind: {kind}")


def _collect_records(output_dir: Path) -> List[Dict[str, Any]]:
    """Load every per-trial record under ``output_dir`` into a flat list."""
    records: List[Dict[str, Any]] = []
    for path in sorted(output_dir.rglob("results-*.json")) + sorted(output_dir.rglob("results.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            records.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict) and "results" in data:
            items = data.get("results")
            if isinstance(items, list):
                records.extend(x for x in items if isinstance(x, dict))
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in records:
        key = (r.get("trial"), r.get("task_id"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(r.get("reward", 0.0) or 0.0) for r in records]
    successes = [
        1.0 if (r.get("reward") is not None and float(r.get("reward") or 0.0) >= 1.0) else 0.0
        for r in records
    ]
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


def _make_env(ns: argparse.Namespace, task_index: int):
    """Construct a fresh tau-bench env for ``task_index``.

    tau-bench's ``get_env`` parameter name for the user-side provider has
    used both ``user_provider`` and ``user_model_provider`` historically;
    try both.
    """
    from tau_bench.envs import get_env  # type: ignore

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


def _solve_one(
    ns: argparse.Namespace,
    AgentCls,
    task_index: int,
    trial: int,
) -> Dict[str, Any]:
    """Run a single (task_index, trial) and return the record dict.

    Each call constructs its own env + agent (no shared mutable state) so
    this is safe to call from a thread pool. The vLLM server batches
    concurrent requests internally, which is what gives us the speed-up
    without touching the model code.
    """
    env = _make_env(ns, task_index)
    agent_kwargs: Dict[str, Any] = dict(
        tools_info=getattr(env, "tools_info", []) or [],
        wiki=getattr(env, "wiki", "") or "",
        model=ns.model,
        provider=ns.model_provider,
        temperature=float(ns.temperature),
    )
    if ns.agent == "sage":
        agent_kwargs["env_hint"] = ns.env
    agent = AgentCls(**agent_kwargs)
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
    return {
        "task_id": task_index,
        "trial": trial,
        "reward": reward,
        "info": info,
        "messages": messages,
        "total_cost": total_cost,
        "status": status,
        "error": err,
    }


def _run_inprocess(ns: argparse.Namespace) -> None:
    """Drive ``ns.agent`` through the in-process tau-bench loop.

    Tasks run concurrently when ``--max-concurrency > 1``: each thread runs
    its own (env, agent) pair end-to-end and dispatches LLM calls to the
    shared vLLM server, which batches them internally. The Python work in
    each thread is dominated by network I/O, so the GIL is released during
    the `chat.completions.create` call — concurrency scales with vLLM's
    batch size, not with thread count alone.
    """
    _try_install_litellm_patch()
    AgentCls = _resolve_agent_cls(ns.agent)

    out_dir = Path(ns.output_dir)
    ensure_dir(out_dir)
    traj_path = out_dir / f"results-{ns.agent}.json"
    jsonl_path = out_dir / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()
    if jsonl_path.exists():
        jsonl_path.unlink()

    work: List[Tuple[int, int]] = [
        (ti, tr)
        for ti in range(ns.start_index, ns.end_index)
        for tr in range(ns.num_trials)
    ]

    write_lock = threading.Lock()
    records: List[Dict[str, Any]] = []
    max_workers = max(1, int(ns.max_concurrency))

    if max_workers == 1:
        for (ti, tr) in work:
            rec = _solve_one(ns, AgentCls, ti, tr)
            append_jsonl(jsonl_path, rec)
            records.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_solve_one, ns, AgentCls, ti, tr): (ti, tr) for (ti, tr) in work}
            for fut in as_completed(futures):
                rec = fut.result()
                with write_lock:
                    append_jsonl(jsonl_path, rec)
                    records.append(rec)

    records.sort(key=lambda r: (r.get("task_id", 0), r.get("trial", 0)))
    write_json(traj_path, records)


def main() -> int:
    ns = _parse_args()
    ensure_dir(ns.output_dir)

    status = "ok"
    error: str = ""
    try:
        _run_inprocess(ns)
    except Exception as exc:  # noqa: BLE001 — record any failure
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"
        traceback.print_exc()

    records = _collect_records(Path(ns.output_dir))
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
        },
        "metrics": metrics,
    }
    write_json(os.path.join(ns.output_dir, "metrics.json"), summary)
    print(json.dumps(summary["metrics"], indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
