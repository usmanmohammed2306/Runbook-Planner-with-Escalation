# IG-RPE — Invariant-Gated Runbook Planner with Escalation

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`) | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`) | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`) | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **IG-RPE** (`igrpe`, *ours*) | Deterministic invariant gate over a symbolic ledger. |

All four conditions go through the **same in-process loop, model,
temperature, tool schemas, max-steps and truncation budget**. The only
varying axis is the controller (and, for IG-RPE only, the deterministic
gate that filters proposed WRITE calls). This makes any performance gap
attributable to the controller — not divergent control flow or
tool-result formatting.

## Why IG-RPE — and why it is the proposed system

Naïve "more prompting" doesn't fix the failure modes a 7B-class model
exhibits on multi-turn tool-use benchmarks. Inspection of failure
trajectories under vanilla / Act / ReAct on τ-retail shows three recurring
patterns:

- issuing a WRITE call (cancel / modify / exchange / refund) **before
  verifying the user** or **before fetching the relevant order**;
- **retrying a failed WRITE with the exact same arguments**;
- declaring task completion **without an explicit user confirmation**.

Adding more prose instructions (longer prompts, ReAct thoughts) does not
reliably fix these — it just crowds the prompt. **IG-RPE treats the
problem as a neuro-symbolic contract**: the LLM proposes a WRITE, a
deterministic gate evaluates invariants over a symbolic ledger of
observable facts, and structured machine-readable feedback is returned
on failure. Reads flow through unchanged, so on the read-heavy prefix of
every trajectory IG-RPE adds **zero** extra LLM calls.

We picked IG-RPE over RPE (an earlier in-tree variant that adds two
LLM calls per step for runbook-update + supervisor) because IG-RPE:

1. Has **strictly lower** wall-clock cost (no extra LLM calls on READ;
   at most one structured-feedback retry on a blocked WRITE).
2. Is **deterministic**: the gate is ~150 lines of regular Python; its
   behavior is reproducible and inspectable.
3. Targets the *actual* failure modes observed in trajectories rather
   than relying on the LLM to plan and replan via natural language.

## Invariants enforced on every WRITE

| Invariant | Meaning |
|---|---|
| `user_verified` | A `find_user_id_*` / `get_user_details` lookup has succeeded for the current user. |
| `order_fetched_if_referenced` | If `order_id` appears in the args, `get_order_details` must have been called for it earlier. |
| `user_confirmed` | An affirmative user turn (yes / confirm / go ahead) within the last two user messages. |
| `not_duplicate` | The same `(tool, args)` pair has not already been issued this session. |
| `under_error_budget` | The tool has not already errored twice. |

On gate failure, a compact machine-readable message is appended as the
tool result; the LLM gets **one** free retry to gather more evidence or
propose a different action. The retry budget replenishes after any
successful WRITE.

## Fixed base model (priority order)

| Priority | Hugging Face ID | Notes |
|---|---|---|
| 1 | `Qwen/Qwen2.5-7B-Instruct` | Primary. 32 K context, stable tool-calling. |
| 2 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 fallback. |
| 3 | `Qwen/Qwen3-4B-Instruct-2507` | Non-FP8 last-resort fallback. |

`run_project.sh` tries the candidates in order and runs **the first one
that serves successfully** for *all four* controllers — guaranteeing the
same model across every condition.

## Two shell scripts (and only two)

- `setup_env.sh` — creates `.venv`, installs requirements, clones
  `tau-bench` and `ACEBench` into `external/`, runs version checks.
- `run_project.sh` — launches vLLM with the model fallback chain, runs
  the **twelve** evaluations (4 controllers × 3 benchmarks), shuts vLLM
  down, and writes the summary.

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
  tau_retail_baseline/   tau_retail_act/   tau_retail_react/   tau_retail_igrpe/
  tau_airline_baseline/  tau_airline_act/  tau_airline_react/  tau_airline_igrpe/
  acebench_agent_baseline/ acebench_agent_act/ acebench_agent_react/ acebench_agent_igrpe/
  summary/
    summary.json   # 4-way comparison + IG-RPE-vs-best-baseline deltas
    summary.md     # rendered table
```

Per-trajectory IG-RPE records include `info.igrpe_ledger_final` and
`info.igrpe_gate_stats` for debugging (allowed / blocked / retries).

## Time budget

Sized to fit **under ~5 hours** on a single A100 (`--enforce-eager`):

- τ-bench: 10 tasks × 2 envs × 4 controllers = 80 trajectories
- ACEBench: 15 tasks × 4 controllers = 60 trajectories

Override via `TAU_END_INDEX`, `ACE_LIMIT`, `TAU_NUM_TRIALS` etc. before
invoking `run_project.sh`.

## What IG-RPE is and isn't

IG-RPE is **lightweight on purpose**: no search tree, no external
memory, no second model, no fine-tuning. The ledger is a plain Python
dataclass; the gate is ~150 lines of deterministic code; invariants are
one-line checker functions. The same base model is used as the three
baselines — the only thing changed between conditions is the controller
itself.
