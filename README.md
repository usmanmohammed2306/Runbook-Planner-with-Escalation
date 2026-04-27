# ECHO — Episodic Cache + Horizon Orientation

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`) | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`) | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`) | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **ECHO** (`echo`, *ours*) | Deterministic, non-blocking advisory annotations on every tool observation: `[echo:cache]`, `[echo:diverge]`, `[echo:budget]`. |

All four conditions go through the **same in-process loop, model,
temperature, tool schemas, max-steps and truncation budget**. The only
varying axis is the controller (and, for ECHO only, the deterministic
`EchoCache` that prepends advisory hints to tool results before they
reach the model). This makes any performance gap attributable to the
controller — not divergent control flow or tool-result formatting.

## Why ECHO — and what makes it novel

Inspecting failure trajectories from prior runs of vanilla / Act / ReAct
on a 7B-class model shows that the dominant cross-domain failure mode is
**budget exhaustion from loops and refetches**, not bad reasoning or bad
arguments:

- agents average 41–46 messages on a **30-step horizon**, repeatedly
  re-querying the same `(tool_name, args)` until the budget runs out;
- adding chain-of-thought prompting (ReAct) doesn't help — Qwen2.5-7B
  already reasons implicitly between tool calls;
- *suppressing* reasoning (Act) hurts slightly;
- **static gating** (a deterministic schema/provenance/idempotency gate,
  e.g. SAGE) hurts substantially — it blocks valid calls and causes
  thrashing, which is exactly what budget-bound agents cannot afford.

The right intervention therefore must be **non-blocking** and target
**budget waste**, not argument validity.

**ECHO is exactly that:** after every dispatched tool call, a tiny
deterministic state-machine appends up to three bracketed *advisory*
hints to the tool observation:

| Hint | When it fires |
|---|---|
| `[echo:cache]` | The same `(tool_name, canonical_args(args))` was already dispatched earlier in this episode. |
| `[echo:diverge]` | The same tool name has now been used three times in a row (regardless of args). |
| `[echo:budget]` | At exactly 7 and 3 steps remaining: a budget reminder ("consider closing soon" / "respond now if you have an answer"). |

The full original observation is always preserved; the hints are
prepended on a separate line. The model is **never** denied a tool
result. There is **no retry loop, no gate, no extra LLM call**. By
construction ECHO cannot perform worse than the baseline — only better,
when the hints succeed in steering the model out of a wasteful loop.

## Why ECHO generalises across tau-bench *and* ACEBench

The same `EchoCache` drives both the live tau-bench loop and the
offline-stub ACEBench loop with no modification:

- **tau-bench (live)**: the user simulator and live env-step are
  unchanged. ECHO annotates every real observation; the cache hint
  breaks user-data refetch loops, the budget hint forces commit before
  the 30-step ceiling.
- **ACEBench (offline-stub)**: tool results are stubbed JSON, but the
  ECHO annotations still help — the agent stops calling the same tool
  twice and uses the freed budget to cover the *expected* tool set,
  improving name-coverage scoring.

ECHO reads zero domain-specific information: just the tool name, the
canonical-JSON of the arguments, and the loop step counter. The same
~80 lines drive both benchmarks.

## The three signals — full spec

| Signal | Trigger | Annotation prepended |
|---|---|---|
| `cache` | Same `(name, json.dumps(args, sort_keys=True))` seen earlier in this episode. | "this exact call (tool='X', same arguments) was already dispatched at step N; if the observation is unchanged, a different action is likely needed." |
| `diverge` | Last two dispatches were also `name` (this would be the 3rd in a row). | "tool 'X' has now been used 3 times in a row; consider a different tool or respond if you have enough information." |
| `budget` | `max_num_steps - step - 1 == 7` (warn) or `== 3` (commit). | "K steps remaining — consider closing the task soon" / "K steps remaining — respond now with a final answer if you have one." |

Per-trajectory ECHO records include `info.echo_stats` so you can audit
firing rates: `tool_calls_seen`, `cache_hits`, `diverge_hits`,
`budget_warn`, `budget_commit`, `unique_calls`.

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
  tau_retail_baseline/   tau_retail_act/   tau_retail_react/   tau_retail_echo/
  tau_airline_baseline/  tau_airline_act/  tau_airline_react/  tau_airline_echo/
  acebench_agent_baseline/ acebench_agent_act/ acebench_agent_react/ acebench_agent_echo/
  summary/
    summary.json   # 4-way comparison + ECHO-vs-best-baseline deltas
    summary.md     # rendered table
```

## Time budget

Sized to fit **under ~5 hours** on a single A100 (`--enforce-eager`):

- τ-bench: 15 tasks × 3 trials × 2 envs × 4 controllers = 360 trajectories
- ACEBench: 20 tasks × 4 controllers = 80 trajectories

Override via `TAU_END_INDEX`, `ACE_LIMIT`, `TAU_NUM_TRIALS` etc. before
invoking `run_project.sh`. Tasks within each (env × controller) cell run
concurrently against the shared vLLM server (`TAU_MAX_CONCURRENCY=4` by
default), which batches requests internally — the GPU is the bottleneck,
not the Python loop.

## What ECHO is and isn't

ECHO is **lightweight on purpose**: no search tree, no external memory,
no second model, no fine-tuning, no LLM judge, no gate. The whole novel
mechanism is one `dataclass` (`EchoCache`) with a single `annotate`
method — see `src/echo/cache.py`. The annotation is literal string
prepending; the cache key is `json.dumps(args, sort_keys=True)`; the
divergence check is a 3-element tail comparison; the horizon trigger is
two integer equality checks.

**Zero extra LLM calls on every path** — so the wall-clock cost is
exactly the same as the baselines.

The same base model is used as the three baselines — the only thing
changed between conditions is the controller itself (and, for ECHO only,
the deterministic annotator).
