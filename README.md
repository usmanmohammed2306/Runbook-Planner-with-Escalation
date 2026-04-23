# Runbook Planner with Escalation (RPE)

Lightweight prototype that compares two agent controllers **on the same fixed
base model**:

1. **Baseline** — vanilla tool-calling via tau-bench's built-in `tool-calling`
   strategy.
2. **RPE** — a compact Runbook Planner with Escalation implemented in
   `src/rpe/`. The planner produces a small runbook (goal, constraints, 2–5
   milestones with primary / fallback / blocker), and a supervisor LLM call
   after each observation decides **advance / keep / replan**.

Benchmarks: **τ-bench retail**, **τ-bench airline**, **ACEBench Agent**.

## Fixed base model (priority order)

| Priority | Hugging Face ID | Notes |
|---|---|---|
| 1 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | Primary. Matches the project PDF. |
| 2 | `Qwen/Qwen3-4B-Instruct-2507` | Non-FP8 fallback if FP8 is not compatible on this GPU/vLLM. |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | Last-resort fallback. |

`run_project.sh` tries them in order and runs **the first one that serves
successfully** for every subsequent baseline and RPE run — guaranteeing the
same model on both sides.

## Two shell scripts (and only two)

- `setup_env.sh` — creates `.venv`, installs requirements, clones
  `tau-bench` and `ACEBench` into `external/`, runs version checks.
- `run_project.sh` — launches vLLM with the model fallback chain, runs the
  six evaluations, shuts vLLM down, and writes the summary.

## Quickstart

```bash
bash setup_env.sh
bash run_project.sh
```

Outputs:

```
outputs/
  active_model.txt
  vllm.log
  tau_retail_baseline/        # tau-bench native logs + metrics.json
  tau_retail_rpe/
  tau_airline_baseline/
  tau_airline_rpe/
  acebench_agent_baseline/    # trajectories.jsonl + metrics.json
  acebench_agent_rpe/
  summary/
    summary.json
    summary.md
```

## Configuration

All tunables are env-var overridable (defaults in `configs/project.yaml` and
`run_project.sh`):

- `TAU_START_INDEX` / `TAU_END_INDEX` — tau-bench task slice (default 0–20)
- `TAU_NUM_TRIALS` — repeats per task
- `ACE_LIMIT` — ACEBench Agent task cap (default 30)
- `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `DTYPE`, `PORT` — vLLM knobs

## What RPE is and isn't

RPE is **lightweight on purpose**: the runbook is a ~5-field JSON blob
injected into the system prompt every turn, plus one extra LLM call after
each observation to decide advance / keep / replan. There are no search
trees, no external memory, and no second model. RPE uses the *same* base
model as the baseline — the only thing changed between the two systems is
the controller.
