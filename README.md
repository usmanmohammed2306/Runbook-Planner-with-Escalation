# SAGE — Schema-Anchored Grounded Execution

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`) | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`) | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`) | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **SAGE** (`sage`, *ours*) | Deterministic gate: schema + provenance + idempotency. |

All four conditions go through the **same in-process loop, model,
temperature, tool schemas, max-steps and truncation budget**. The only
varying axis is the controller (and, for SAGE only, the deterministic
gate that filters proposed calls). This makes any performance gap
attributable to the controller — not divergent control flow or
tool-result formatting.

## Why SAGE — and what makes it novel

Naïve "more prompting" doesn't fix the failure modes a 7B-class model
exhibits on multi-turn tool-use benchmarks. Inspecting failure
trajectories under vanilla / Act / ReAct on **both τ-bench and ACEBench**
shows that the dominant cross-domain failure is **argument hallucination**:

- the model invents an `order_id` (`order-W12345`) it never fetched;
- it passes an `email` that wasn't in the user message;
- it picks an `enum` value not present in the tool's schema;
- it omits a required parameter or supplies the wrong type.

These failures are **invariant across domains** — they happen on
tau-retail (where IDs come from the user simulator), on tau-airline
(reservation codes), and on ACEBench's varied generic tools (which only
provide a JSONSchema, no live environment).

**SAGE is a domain-agnostic neuro-symbolic contract:** the LLM proposes a
tool call; a deterministic ~250-line gate validates *three* conditions
purely from the JSONSchema and the literal conversation transcript.
Failures are returned as compact machine-readable feedback; the model gets
**one** free retry to re-ground before the call is forced through to keep
the loop alive.

The novelty is the specific recipe: **provenance grounding** (every
identifier-shaped string argument must appear in the corpus) **+ JSONSchema
validation + idempotency**, all deterministic, all in a single gate, applied
identically to a multi-turn benchmark *and* an offline benchmark. Related
work in 2025–26 (AgentProp-Bench runtime interceptors, Cleanlab trust
scoring, three-layer guardrails) either uses LLM judges, fine-tuning, or
schema-only checks — none combine literal-context provenance with schema
in a single deterministic gate that ports unchanged from tau-bench to
ACEBench.

## Why SAGE generalizes where IG-RPE didn't

An earlier in-tree variant (IG-RPE) hardcoded retail/airline-specific
invariants (`user_verified`, `order_fetched`, `user_confirmed`). It helped
on tau-bench but was neutral on ACEBench, where the tool vocabulary is
varied and there is no user simulator. **SAGE removes every domain term**:
it only reads from (a) the tool's JSONSchema and (b) the raw conversation
transcript. The same module drives both the tau-bench `Agent` subclass and
the ACEBench offline-stub loop without modification.

## The three checks enforced on every proposed call

| Check | What it enforces |
|---|---|
| `schema` | Required parameters present; types match (`string`, `integer`, `number`, `boolean`, `array`, `object`); enum values are within the allowed set. |
| `provenance` | Every identifier-shaped string argument (length 3–80, no whitespace, contains a digit / `@` / `-` / `_` / `/`, or is an ALL-CAPS short code) literally appears in the conversation corpus: user messages, prior tool observations, system prompt, or any of the tool schemas' enum values. Free-form prose arguments (e.g., `content: "yes please"`) are NOT checked. |
| `idempotency` | The same `(tool, normalized_args)` is not issued twice; a tool that has errored ≥ 2 times is not retried with similar args. |

On block, a JSON object is returned as the tool result:

```json
{
  "sage_blocked": true,
  "tool": "cancel_pending_order",
  "checks_failed": ["ungrounded_arg:order_id:order-W99999"],
  "guidance": "- ungrounded_arg: ... fetch the value from a prior tool first, or ask the user.",
  "next_step": "Re-examine the conversation, fetch any missing IDs via a READ tool, ..."
}
```

The LLM receives this on its next turn and either re-grounds or asks the
user for the missing value. The retry budget replenishes after any
successful dispatch.

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
  tau_retail_baseline/   tau_retail_act/   tau_retail_react/   tau_retail_sage/
  tau_airline_baseline/  tau_airline_act/  tau_airline_react/  tau_airline_sage/
  acebench_agent_baseline/ acebench_agent_act/ acebench_agent_react/ acebench_agent_sage/
  summary/
    summary.json   # 4-way comparison + SAGE-vs-best-baseline deltas
    summary.md     # rendered table
```

Per-trajectory SAGE records include `info.sage_gate_stats` for debugging
(allowed / blocked / retries, plus a histogram of which check tripped:
`schema`, `provenance`, `idempotency`).

## Time budget

Sized to fit **under ~5 hours** on a single A100 (`--enforce-eager`):

- τ-bench: 15 tasks × 3 trials × 2 envs × 4 controllers = 360 trajectories
- ACEBench: 20 tasks × 4 controllers = 80 trajectories

Override via `TAU_END_INDEX`, `ACE_LIMIT`, `TAU_NUM_TRIALS` etc. before
invoking `run_project.sh`. Tasks within each (env × controller) cell run
concurrently against the shared vLLM server (`TAU_MAX_CONCURRENCY=4` by
default), which batches requests internally — the GPU is the bottleneck,
not the Python loop.

## What SAGE is and isn't

SAGE is **lightweight on purpose**: no search tree, no external memory, no
second model, no fine-tuning, no LLM judge. The gate is ~250 lines of
deterministic Python. The provenance check is literal substring matching
into a lower-cased corpus. The schema check reads the tool's JSONSchema
directly. The idempotency check is a hash-set lookup. Zero extra LLM calls
on the happy path; ≤ 1 retry-call on a blocked tool — so the wall-clock
cost is essentially the same as the baselines.

The same base model is used as the three baselines — the only thing changed
between conditions is the controller itself (and, for SAGE only, the
deterministic gate).
