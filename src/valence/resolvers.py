"""Deterministic resolvers for VALENCE.

Resolvers may transform values, but every output value must inherit
provenance from one or more handles. There is no LLM resolver. Whenever
the input is ambiguous, the resolver fails closed by returning ``None``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .handles import CONF_RESOLVED, Handle


@dataclass(frozen=True)
class ResolvedToken:
    """Output of a deterministic resolver, carrying provenance."""

    type: str
    value: Any
    resolver: str
    source_handle_ids: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)
    confidence: str = CONF_RESOLVED


# ---------------------------------------------------------------------------
# Date resolver
# ---------------------------------------------------------------------------
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def resolve_date(handle: Handle) -> Optional[ResolvedToken]:
    """Pass-through resolver for already-ISO dates. Fails closed otherwise."""
    if handle is None or handle.type != "datetime":
        return None
    val = str(handle.value).strip()
    if not _ISO_DATE.match(val):
        return None
    return ResolvedToken(
        type="datetime",
        value=val,
        resolver="resolve_date.iso_passthrough",
        source_handle_ids=[handle.handle_id],
        provenance={"input": val},
    )


# ---------------------------------------------------------------------------
# Money / arithmetic resolver
# ---------------------------------------------------------------------------
def resolve_money(op: str, *, base: Optional[Handle] = None,
                  other: Optional[Handle] = None,
                  literal: Optional[float] = None) -> Optional[ResolvedToken]:
    """Compute money from handle-backed values only.

    Supported ``op``:
      * ``"exact"``    — value is literal *only if* ``base`` is None and a
                          handle backs ``literal`` (we still require a handle
                          for provenance, so ``base`` is mandatory in practice)
      * ``"full"``     — base.value
      * ``"half"``     — base.value / 2
      * ``"diff"``     — base.value - other.value (both must be handles)
    """
    op = str(op or "").lower()
    if op == "full":
        if base is None or base.type != "money":
            return None
        return ResolvedToken(
            type="money",
            value=float(base.value),
            resolver="resolve_money.full",
            source_handle_ids=[base.handle_id],
        )
    if op == "half":
        if base is None or base.type != "money":
            return None
        return ResolvedToken(
            type="money",
            value=round(float(base.value) / 2.0, 2),
            resolver="resolve_money.half",
            source_handle_ids=[base.handle_id],
        )
    if op == "diff":
        if base is None or other is None or base.type != "money" or other.type != "money":
            return None
        return ResolvedToken(
            type="money",
            value=round(float(base.value) - float(other.value), 2),
            resolver="resolve_money.diff",
            source_handle_ids=[base.handle_id, other.handle_id],
        )
    if op == "exact":
        if base is None or base.type != "money":
            return None
        return ResolvedToken(
            type="money",
            value=float(base.value),
            resolver="resolve_money.exact",
            source_handle_ids=[base.handle_id],
        )
    return None


# ---------------------------------------------------------------------------
# Selector resolver — chooses *only* among tool-returned candidates
# ---------------------------------------------------------------------------
def resolve_selector(mode: str, candidates: List[Handle],
                     *, query: Optional[str] = None) -> Optional[ResolvedToken]:
    """Pick one handle from a list of candidates produced by tool evidence.

    ``candidates`` MUST be handles previously minted from tool observations.
    Modes:
      * ``"first"``           — first candidate
      * ``"cheapest"``        — money-typed: minimum value
      * ``"most_expensive"``  — money-typed: maximum value
      * ``"exact_match"``     — string-equality with ``query``; ambiguous
                                  multi-match fails closed.
    """
    mode = str(mode or "").lower()
    if not candidates:
        return None
    if mode == "first":
        c = candidates[0]
        return ResolvedToken(
            type=c.type, value=c.value,
            resolver="resolve_selector.first",
            source_handle_ids=[c.handle_id],
        )
    if mode in ("cheapest", "most_expensive"):
        money = [c for c in candidates if c.type == "money" and isinstance(c.value, (int, float))]
        if not money:
            return None
        chosen = min(money, key=lambda h: float(h.value)) if mode == "cheapest" \
            else max(money, key=lambda h: float(h.value))
        return ResolvedToken(
            type="money", value=float(chosen.value),
            resolver=f"resolve_selector.{mode}",
            source_handle_ids=[chosen.handle_id],
        )
    if mode == "exact_match":
        if query is None:
            return None
        q = str(query)
        matches = [c for c in candidates if str(c.value) == q]
        if len(matches) != 1:
            return None  # ambiguous or missing → fail closed
        c = matches[0]
        return ResolvedToken(
            type=c.type, value=c.value,
            resolver="resolve_selector.exact_match",
            source_handle_ids=[c.handle_id],
            provenance={"query": q},
        )
    return None


# ---------------------------------------------------------------------------
# Enum resolver — schema enums only
# ---------------------------------------------------------------------------
def resolve_enum(value: str, allowed: Iterable[str],
                 source_event_id: str) -> Optional[ResolvedToken]:
    """Return value iff it is an exact match against the schema enum list."""
    if value is None:
        return None
    allowed_list = list(allowed or [])
    if not allowed_list:
        return None
    if value not in allowed_list:
        return None
    return ResolvedToken(
        type="enum",
        value=value,
        resolver="resolve_enum.exact",
        source_handle_ids=[],
        provenance={"source_event_id": source_event_id, "allowed": allowed_list},
    )
