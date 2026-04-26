"""SAGE — Schema-Anchored Grounded Execution.

A lightweight, domain-agnostic deterministic gate that runs between the
LLM's tool-call decode and the dispatch to ``env.step`` (tau-bench) or
the offline stub (ACEBench). It enforces three checks on every proposed
call:

  1. **Schema validation** — required args present, types match, enum
     values in the allowed set (uses each tool's JSONSchema directly).
  2. **Provenance grounding** — every identifier-shaped string argument
     (IDs, emails, codes) must literally appear in the conversation
     corpus (user messages + prior tool observations + system prompt +
     enum values). Prevents the model from inventing IDs.
  3. **Idempotency** — the same ``(tool, normalized_args)`` is not
     issued twice; a tool that has errored twice is not retried with
     similar arguments.

On block: a compact machine-readable feedback message is appended as the
tool result and the LLM is given **one** free retry to gather more
evidence or re-ground its arguments. The retry budget replenishes after
any successful write.

This module is intentionally domain-agnostic: it reads only from the
tool's JSONSchema and from raw conversation text. Unlike IG-RPE, no
retail/airline vocabulary is hardcoded — so it works equally well on
ACEBench's varied tool tasks and on tau-bench's multi-turn dialogues.
"""
from __future__ import annotations

from .gate import GateResult, sage_gate, build_corpus, ConversationCorpus

__all__ = ["GateResult", "sage_gate", "build_corpus", "ConversationCorpus"]
