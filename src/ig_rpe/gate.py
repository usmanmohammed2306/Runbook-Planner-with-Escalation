"""Deterministic gate for IG-RPE.

Classifies tools as READ or WRITE based on the tool name / description and
runs the invariant suite on WRITE calls. The classifier is intentionally
conservative: unknown tools default to READ so IG-RPE never *blocks* safe
lookups, only destructive operations that match known write patterns.

The classifier is configurable per-environment (tau-retail, tau-airline,
ACEBench). A shared heuristic fallback covers any tool we haven't enumerated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

from .invariants import evaluate, render_feedback
from .state import Ledger


# Tool names observed across tau-bench retail + airline that mutate state.
# If a tool's name contains any of these substrings (case-insensitive) AND
# it is NOT in the ``READ_OVERRIDES`` set, it is classified as WRITE.
WRITE_KEYWORDS: Tuple[str, ...] = (
    "modify", "cancel", "exchange", "return_", "return_delivered",
    "book_", "update_", "transfer_to_human", "change_", "add_baggage",
    "send_certificate", "issue_refund", "process_refund", "place_order",
    "delete", "create_", "remove_", "set_",
)

# Things that look write-ish by name but are actually safe lookups.
READ_OVERRIDES: Tuple[str, ...] = (
    "calculate", "list_all_", "get_", "search", "lookup", "find_", "read_",
    "show_", "describe_", "user_details",
)


@dataclass
class Classification:
    is_write: bool
    reason: str


def classify_tool(tool_name: str, tool_spec: Dict[str, Any] | None = None) -> Classification:
    """Return whether ``tool_name`` is a WRITE (state-changing) operation."""
    if not tool_name:
        return Classification(False, "empty-name")
    name = tool_name.lower()

    # Strong reads first so overrides take precedence.
    for kw in READ_OVERRIDES:
        if name.startswith(kw) or name == kw.strip("_"):
            return Classification(False, f"read-override:{kw}")

    for kw in WRITE_KEYWORDS:
        if kw in name:
            return Classification(True, f"write-kw:{kw}")

    # Fall through: inspect description if provided.
    if tool_spec:
        desc = ""
        fn = tool_spec.get("function") if isinstance(tool_spec, dict) else None
        if isinstance(fn, dict):
            desc = str(fn.get("description") or "")
        low = desc.lower()
        if any(x in low for x in ("update", "modify", "cancel", "exchange", "refund", "place an order", "delete")):
            return Classification(True, "write-desc")

    return Classification(False, "default-read")


def gate_check(
    ledger: Ledger,
    tool_name: str,
    args: Dict[str, Any],
    tool_spec: Dict[str, Any] | None = None,
) -> Tuple[bool, str, Classification]:
    """Run the gate. Returns (allow, feedback_if_blocked, classification).

    READ calls are always allowed. WRITE calls run the invariant suite.
    """
    cls = classify_tool(tool_name, tool_spec)
    if not cls.is_write:
        return True, "", cls
    ok, results = evaluate(ledger, tool_name, args)
    if ok:
        return True, "", cls
    return False, render_feedback(results), cls


def summarize_tools_for_prompt(tool_specs: List[Dict[str, Any]], max_items: int = 40) -> str:
    """Render a compact READ/WRITE classification block for the system prompt."""
    if not tool_specs:
        return ""
    reads: List[str] = []
    writes: List[str] = []
    for ts in tool_specs:
        fn = ts.get("function") if isinstance(ts, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if not name:
            continue
        cls = classify_tool(name, ts)
        (writes if cls.is_write else reads).append(name)
    lines: List[str] = []
    if reads:
        lines.append("READ (safe, free to call): " + ", ".join(sorted(reads)[:max_items]))
    if writes:
        lines.append("WRITE (gated; invariants enforced): " + ", ".join(sorted(writes)[:max_items]))
    return "\n".join(lines)
