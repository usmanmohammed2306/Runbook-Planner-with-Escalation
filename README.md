# VALENCE — Verified Affordance Lattice for Efficient Non-hallucinating Control

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`)  | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`)              | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`)          | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **VALENCE** (`valence`, *ours*) | The LLM never generates raw mutation tool calls; instead it picks one verified `action_id` from a compact, deterministically-built affordance menu, and the kernel compiles that into a real benchmark tool call. |

All four conditions share the **same in-process loop, model,
temperature, tool schemas, max-steps, and truncation budget**. The only
varying axis is the controller — for VALENCE, the deterministic
`AffordanceKernel` that compiles a model-chosen `action_id` into the
real tool call.

## The invariant

> **No state-changing API call may receive an argument unless that value
> was minted by the VALENCE kernel from one of: prior tool evidence,
> exact user-text span, schema enum, deterministic resolver output,
> verified arithmetic/selection over grounded values, or
> sandbox/transaction-validated transformation.**

If provenance is missing, mutations **fail closed** — they don't appear
as executable in the menu, they don't compile, and (belt-and-braces) the
transaction validator rejects them at the boundary even if they slip
through.

## How VALENCE works (per step)

1. **Ingest**: the kernel records the user message and any tool
   observations as immutable `Event`s.
2. **Mint handles**: typed values (`order_id`, `user_id`,
   `payment_method_id`, `money`, `datetime`, …) are extracted from
   tool-result JSON (preferred) and from exact user-text spans (regex).
   Each handle carries a pointer back to the source event.
3. **Build affordances**: for each available tool the kernel tries to
   bind every required parameter to a handle. Mutations are *executable*
   only when every required risky arg is bound; otherwise they appear
   non-executable and the menu surfaces useful read/search/ask/final
   alternatives.
4. **Render menu**: top-k=8 deterministic ranking, e.g.

   ```
   Available verified actions:
   A1 mutation: cancel_order(order_id=O1234) — cancel an existing order
   A2 read: get_order_details(order_id=O1234) — fetch order details
   A3 read: get_user_details(user_id=alex_smith_42) — fetch user
   A4 search: search_items(query=black jacket) — search catalog
   A5 final: produce a short final answer to the user.
   Budget: 12 steps left.
   ```
5. **LLM call** with the *tiny* system prompt:

   > You must choose exactly one verified action_id from the menu.
   > Do not invent tool arguments. If no executable mutation is
   > available, choose read/search/ask/final.
   > Return only JSON: `{"action_id":"A1"}`.

6. **Compile + validate**: `kernel.compile_action("A1")` → a
   `CompiledAction(tool_name=..., kwargs=..., argument_refs=...)`. If
   the action_id was not in the rendered menu (hallucination), or the
   chosen affordance was non-executable, compile returns `None`. For
   mutations, `validate_mutation` re-checks that every kwarg has a
   provenance ref and that this exact `(tool, kwargs)` signature has
   not been executed before in this episode.
7. **Translate + execute**: the validated call goes through `env.step`
   exactly like the baselines, so the upstream scorer (tau-bench reward,
   ACEBench `score_agent.py`) sees the *real* tool name and arguments.

## Diagnostics (per controller, per cell)

`info.valence_stats` records:

- `grounded_mutation_rate`        — fraction of mutation attempts that validated
- `rejected_ungrounded_mutations` — mutations blocked for missing provenance
- `duplicate_mutation_rejections` — same `(tool, kwargs)` already executed
- `valence_compile_failures`      — hallucinated / non-executable choices
- `action_menu_size_mean`         — average top-k size shown to the model
- `mutation_attempts`, `total_actions`

## Fixed base model (priority order)

| Priority | Hugging Face ID                  | Notes |
|---|---|---|
| 1 | `Qwen/Qwen2.5-7B-Instruct`        | Primary. 32 K context, stable tool-calling. |
| 2 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 fallback. |
| 3 | `Qwen/Qwen3-4B-Instruct-2507`     | Non-FP8 last-resort fallback. |

`run_project.sh` tries the candidates in order and runs **the first one
that serves successfully** for *all four* controllers — guaranteeing the
same model across every condition.

## Two shell scripts (and only two)

- `setup_env.sh`   — creates `.venv`, installs requirements, clones
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
  tau_retail_baseline/   tau_retail_act/   tau_retail_react/   tau_retail_valence/
  tau_airline_baseline/  tau_airline_act/  tau_airline_react/  tau_airline_valence/
  acebench_agent_baseline/ acebench_agent_act/ acebench_agent_react/ acebench_agent_valence/
  summary/
    summary.json   # 4-way comparison + VALENCE-vs-best-baseline + VALENCE-vs-baseline deltas
    summary.md     # rendered table
```

## Tests

The invariant is covered by offline unit tests (no live model needed):

```bash
python3 -m unittest tests.test_valence -v
```

The suite checks: hallucinated `action_id` cannot compile; grounded
mutations compile and validate; resolver output carries provenance;
ambiguous resolvers fail closed; selectors only choose among
tool-returned candidates; duplicate mutations are rejected; the rendered
menu is bounded; choose-action JSON translates to a real tool call;
baselines still import; VALENCE controllers initialize.

## What VALENCE is — and isn't

VALENCE is **lightweight on purpose**: no extra LLM call per step, no
search tree, no policy retriever, no LLM judge, no fine-tuning. The
novel mechanism is one `AffordanceKernel` (≈ 200 LoC) plus typed
handles, deterministic resolvers, and a transaction validator.

The same kernel drives both the live tau-bench loop (real `env.step`)
and the offline-stub ACEBench loop (translated calls preserved for
upstream scoring). Wall-clock cost is essentially identical to the
baselines.
