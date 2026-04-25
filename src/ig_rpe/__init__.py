"""Invariant-Gated RPE (IG-RPE).

A neuro-symbolic tool-calling controller. Tool specs are partitioned into
READ (safe, idempotent lookups) and WRITE (state-changing) sets. Before any
WRITE call is executed, the agent must emit structured JSON invariants which
are validated by a deterministic gate against a ledger built from prior
turns. On gate failure the agent receives structured feedback and retries
with more evidence or adjusted plan. READ calls bypass the gate entirely —
so in the common case we add zero overhead relative to vanilla tool calling.
"""
