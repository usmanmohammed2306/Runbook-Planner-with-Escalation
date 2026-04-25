"""Invariant definitions for IG-RPE.

An invariant is a named, machine-checkable precondition that must hold before
a WRITE tool is executed. The agent emits each invariant as JSON alongside
the proposed WRITE call; a deterministic gate evaluates each invariant
against the ledger.

Invariants are intentionally small and composable. Adding a new one only
requires: (1) giving it a name, (2) writing a one-line checker that consumes
the ledger + the proposed (tool_name, args), (3) listing it in
``REQUIRED_INVARIANTS_FOR_WRITE``.

All invariants are *advisory over* the baseline — they never block a READ
call and never block a WRITE call the agent has good evidence for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .state import Ledger


@dataclass
class InvariantResult:
    name: str
    ok: bool
    detail: str = ""


Invariant = Callable[[Ledger, str, Dict[str, Any]], InvariantResult]


# --- individual invariants ------------------------------------------------

def inv_user_verified(ledger: Ledger, tool: str, args: Dict[str, Any]) -> InvariantResult:
    ok = ledger.user_verified()
    return InvariantResult(
        name="user_verified",
        ok=ok,
        detail="" if ok else (
            "No user has been identified yet. Call find_user_id_by_email or "
            "find_user_id_by_name_zip first, or ask the user for their email / name+zip."
        ),
    )


def inv_order_fetched_if_referenced(ledger: Ledger, tool: str, args: Dict[str, Any]) -> InvariantResult:
    """Any tool that references an ``order_id`` must have fetched it first."""
    oid = None
    for key in ("order_id", "orderId", "order"):
        if key in args and args[key]:
            oid = str(args[key])
            break
    if not oid:
        return InvariantResult("order_fetched_if_referenced", True)
    ok = ledger.order_fetched(oid)
    return InvariantResult(
        name="order_fetched_if_referenced",
        ok=ok,
        detail="" if ok else (
            f"order_id '{oid}' was referenced but has never been fetched via "
            f"get_order_details. Fetch it first to confirm the state before writing."
        ),
    )


def inv_user_confirmed_recently(ledger: Ledger, tool: str, args: Dict[str, Any]) -> InvariantResult:
    """For destructive writes, require an affirmative user confirmation within 2 user turns."""
    ok = ledger.recent_confirmation(window=2)
    return InvariantResult(
        name="user_confirmed",
        ok=ok,
        detail="" if ok else (
            "You have not received an explicit confirmation ('yes', 'go ahead', etc.) "
            "from the user for this exact action in the last two user turns. "
            "Summarize the planned change and ask the user to confirm before writing."
        ),
    )


def inv_not_duplicate(ledger: Ledger, tool: str, args: Dict[str, Any]) -> InvariantResult:
    ok = not ledger.is_duplicate_call(tool, args)
    return InvariantResult(
        name="not_duplicate",
        ok=ok,
        detail="" if ok else (
            f"You have already called {tool} with these exact arguments earlier in this "
            f"conversation. If that call failed, diagnose the error first instead of retrying blindly."
        ),
    )


def inv_under_error_budget(ledger: Ledger, tool: str, args: Dict[str, Any], budget: int = 2) -> InvariantResult:
    seen = ledger.errors_by_tool.get(tool, 0)
    ok = seen < budget
    return InvariantResult(
        name="under_error_budget",
        ok=ok,
        detail="" if ok else (
            f"The tool {tool} has already failed {seen} times. Stop retrying it; try a "
            f"different path or hand off to transfer_to_human_agents only if the policy allows."
        ),
    )


# --- registry -------------------------------------------------------------

REQUIRED_INVARIANTS_FOR_WRITE: List[Invariant] = [
    inv_user_verified,
    inv_order_fetched_if_referenced,
    inv_user_confirmed_recently,
    inv_not_duplicate,
    inv_under_error_budget,
]


def evaluate(ledger: Ledger, tool: str, args: Dict[str, Any]) -> Tuple[bool, List[InvariantResult]]:
    """Run every WRITE invariant against the ledger. Returns (passed_all, results)."""
    results: List[InvariantResult] = []
    ok_all = True
    for inv in REQUIRED_INVARIANTS_FOR_WRITE:
        res = inv(ledger, tool, args)
        results.append(res)
        if not res.ok:
            ok_all = False
    return ok_all, results


def render_feedback(results: List[InvariantResult]) -> str:
    """Render a compact, actionable message the LLM can consume to retry."""
    failed = [r for r in results if not r.ok]
    if not failed:
        return "All invariants satisfied."
    lines = ["[GATE] Your WRITE call was blocked. The following invariants failed:"]
    for r in failed:
        lines.append(f"- {r.name}: {r.detail}")
    lines.append(
        "Do NOT retry the same WRITE call. First satisfy each failed invariant "
        "(call a READ tool, ask the user to confirm, or choose a different action), "
        "then propose the write again."
    )
    return "\n".join(lines)
