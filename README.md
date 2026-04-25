# Runbook Planner with Escalation (RPE) + IG-RPE

Lightweight prototype that compares **three** agent controllers **on the same fixed
base model**:

1. **Baseline** — vanilla tool-calling via tau-bench's built-in `tool-calling`
   strategy.
2. **RPE** — compact Runbook Planner with Escalation (`src/rpe/`). A small
   runbook (goal, constraints, 2–5 milestones with primary / fallback / blocker)
   is injected into the system prompt, plus a supervisor LLM call after each
   observation decides **advance / keep / replan**.
3. **IG-RPE** (`src/ig_rpe/`) — the contribution of this project. A
   neuro-symbolic controller: tools are classified READ vs WRITE, a symbolic
   **ledger** accumulates facts from observations, and a deterministic **gate**
   enforces invariants before every WRITE call. Adds **zero** extra LLM calls
   on the READ path and at most one retry on a blocked WRITE.

Benchmarks: **τ-bench retail**, **τ-bench airline**, **ACEBench Agent**.

## Why IG-RPE

Early pilot runs of RPE on τ-bench retail with Qwen scored **below** the
vanilla baseline (8.3 % vs 20 %). Inspection of the failure trajectories showed
three recurring patterns on a 7B-class model:

- issuing a WRITE call (cancel / modify / exchange / refund) **before verifying
  the user** or **before fetching the relevant order**;
- retrying a failed WRITE with **the exact same arguments**;
- declaring task completion **without an explicit user confirmation**.

Adding more prose instructions (RPE's runbook) did not reliably fix these —
it just crowded the prompt. IG-RPE treats the problem instead as a
**neuro-symbolic contract**: the LLM proposes a WRITE, a deterministic gate
evaluates invariants over the ledger, and structured feedback is returned on
failure. Reads flow through unchanged.

## Invariants enforced on every WRITE

| Invariant | Meaning |
|---|---|
| `user_verified` | A `find_user_id_*` / `get_user_details` lookup has succeeded for the current user. |
| `order_fetched_if_referenced` | If `order_id` appears in args, `get_order_details` must have been called for it. |
| `user_confirmed` | An affirmative user turn (yes / confirm / go ahead) within the last two user messages. |
| `not_duplicate` | The same `(tool, args)` pair has not already been issued this session. |
| `under_error_budget` | The tool has not already errored twice. |

On gate failure, a compact machine-readable message is appended as the tool
result; the LLM gets **one** free retry to gather more evidence or propose a
different action. Budget replenishes after any successful WRITE.

## Fixed base model (priority order)

| Priority | Hugging Face ID | Notes |
|---|---|---|
| 1 | `Qwen/Qwen2.5-7B-Instruct` | Primary. 32 K context, stable tool-calling. |
| 2 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 fallback. |
| 3 | `Qwen/Qwen3-4B-Instruct-2507` | Non-FP8 last-resort fallback. |

`run_project.sh` tries them in order and runs **the first one that serves
successfully** for every subsequent baseline / RPE / IG-RPE run — guaranteeing
the same model across all three conditions.

## Two shell scripts (and only two)

- `setup_env.sh` — creates `.venv`, installs requirements, clones
  `tau-bench` and `ACEBench` into `external/`, runs version checks.
- `run_project.sh` — launches vLLM with the model fallback chain, runs the
  **nine** evaluations (3 conditions × 3 benchmarks), shuts vLLM down, and
  writes the summary.

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
  tau_retail_baseline/          # tau-bench native logs + metrics.json
  tau_retail_rpe/
  tau_retail_igrpe/             # NEW
  tau_airline_baseline/
  tau_airline_rpe/
  tau_airline_igrpe/            # NEW
  acebench_agent_baseline/
  acebench_agent_rpe/
  acebench_agent_igrpe/         # NEW
  summary/
    summary.json                # 3-way comparison
    summary.md                  # rendered table
```

Per-trajectory IG-RPE records include `info.igrpe_ledger_final` and
`info.igrpe_gate_stats` for debugging (allowed / blocked / retries).

## Configuration

Env-var overridable tunables (defaults in `run_project.sh`):

- `TAU_START_INDEX` / `TAU_END_INDEX` — tau-bench task slice (default 0–20)
- `TAU_NUM_TRIALS` — repeats per task
- `ACE_LIMIT` — ACEBench Agent task cap (default 30)
- `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `DTYPE`, `PORT` — vLLM knobs

## What IG-RPE is and isn't

IG-RPE is **lightweight on purpose**: no search tree, no external memory, no
second model, no fine-tuning. The ledger is a plain Python dataclass, the
gate is ~150 lines of deterministic code, and invariants are one-line checker
functions. Same base model as the baseline and RPE — the only thing changed
between conditions is the controller.
