"""Affordance lattice construction for VALENCE.

Given the event log, the minted handles and the available tool schemas,
``build_affordances`` enumerates a small, deterministic set of actions
the model is allowed to choose from. Each action carries a provenance
record describing which handle(s) supplied each argument; mutations
appear as executable only when every required risky argument is
handle-bound.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .handles import Handle


# Tools whose names match these prefixes are treated as state-changing
# mutations and gated behind handle-only binding.
_MUTATION_PREFIXES = (
    "update_", "modify_", "change_", "cancel_", "create_", "delete_",
    "book_", "place_", "submit_", "return_", "exchange_", "pay_",
    "transfer_", "send_", "remove_", "add_", "set_", "edit_",
)

# Read-only / informational tool prefixes. These do not require typed-handle
# binding for free-form arguments (e.g. search queries from user text).
_READ_PREFIXES = (
    "get_", "list_", "find_", "search_", "lookup_", "calculate_",
    "calc_", "view_", "show_", "retrieve_", "fetch_", "describe_",
)

# Tools that are *not* mutations even though their name might look risky.
_NEUTRAL_TOOLS = {"respond", "transfer_to_human_agents", "think", "finish"}


def classify_tool(name: str) -> str:
    """Return one of {'mutation', 'read', 'neutral'}."""
    n = (name or "").lower()
    if n in _NEUTRAL_TOOLS:
        return "neutral"
    if any(n.startswith(p) for p in _MUTATION_PREFIXES):
        return "mutation"
    if any(n.startswith(p) for p in _READ_PREFIXES):
        return "read"
    # Unknown — treat as mutation to fail-closed.
    return "mutation"


# ---------------------------------------------------------------------------
# Affordance dataclass
# ---------------------------------------------------------------------------
@dataclass
class Affordance:
    action_id: str
    kind: str                                  # read | search | ask | mutation | final
    display_text: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    argument_refs: Dict[str, str] = field(default_factory=dict)  # arg -> handle_id / "user_text"
    missing_requirements: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    is_executable: bool = True


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def _required_params(schema: Dict[str, Any]) -> List[str]:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
    if isinstance(params, dict):
        req = params.get("required") or []
        if isinstance(req, list):
            return [str(p) for p in req]
    return []


def _param_props(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
    if isinstance(params, dict):
        props = params.get("properties") or {}
        if isinstance(props, dict):
            return props
    return {}


def _tool_name(schema: Dict[str, Any]) -> str:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    return str(fn.get("name", "")) if isinstance(fn, dict) else ""


def _short_desc(schema: Dict[str, Any], max_chars: int = 80) -> str:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    d = str(fn.get("description", "")).strip().splitlines()[0] if isinstance(fn, dict) else ""
    return d[:max_chars]


# ---------------------------------------------------------------------------
# Argument binding from handles
# ---------------------------------------------------------------------------
# Heuristic: parameter name → preferred handle types.
_PARAM_TYPE_HINT = {
    "user_id": ("user_id",),
    "customer_id": ("customer_id", "user_id"),
    "order_id": ("order_id",),
    "reservation_id": ("reservation_id",),
    "booking_id": ("booking_id",),
    "item_id": ("item_id",),
    "product_id": ("product_id", "item_id"),
    "flight_id": ("flight_id",),
    "payment_method_id": ("payment_method_id",),
    "payment_id": ("payment_method_id",),
    "status": ("status",),
    "amount": ("money",),
    "price": ("money",),
    "total": ("money",),
    "date": ("datetime",),
    "departure_date": ("datetime",),
    "return_date": ("datetime",),
}


def _bind_param(param: str, prop: Dict[str, Any], handles: List[Handle],
                user_text_handles: List[Handle]) -> Optional[Tuple[Any, str, Handle]]:
    """Try to bind a parameter name to a (value, ref, handle) triple.

    Returns ``None`` if no exact handle match is available.
    """
    pname = param.lower()
    types = _PARAM_TYPE_HINT.get(pname, ())

    # Schema enum match: if the property declares an enum, do not bind from
    # a handle — the kernel will pass the schema enum through resolve_enum
    # at compile-time, only after the model has chosen the action_id.
    enum = (prop or {}).get("enum") if isinstance(prop, dict) else None

    # 1. Typed handle by hint.
    for typ in types:
        for h in reversed(handles):  # prefer most recent
            if h.type == typ:
                return (h.value, h.handle_id, h)

    # 2. Direct value match: a handle whose source path or content equals the
    # param name.
    for h in reversed(handles):
        sp = (h.source_path or "").lower()
        if sp.endswith("." + pname) or sp == pname:
            return (h.value, h.handle_id, h)

    # 3. For free-form string params (no enum), allow user-text handles.
    if not enum:
        for h in reversed(user_text_handles):
            if h.type in ("string", "user_id", "order_id"):
                return (h.value, h.handle_id, h)

    return None


# ---------------------------------------------------------------------------
# Build affordances
# ---------------------------------------------------------------------------
def build_affordances(*, tool_schemas: List[Dict[str, Any]],
                      handles: List[Handle],
                      user_text_handles: List[Handle],
                      executed_signatures: Optional[List[str]] = None,
                      remaining_steps: int = 30,
                      respond_tool_name: str = "respond") -> List[Affordance]:
    """Enumerate the affordance lattice for the current step.

    The order is deterministic: by tool index in ``tool_schemas`` first,
    then by binding completeness. The kernel later renders the top-k.
    """
    executed = set(executed_signatures or [])
    out: List[Affordance] = []
    aid = 0

    def next_id() -> str:
        nonlocal aid
        aid += 1
        return f"A{aid}"

    for schema in tool_schemas:
        name = _tool_name(schema)
        if not name:
            continue
        kind_class = classify_tool(name)
        required = _required_params(schema)
        props = _param_props(schema)

        bound: Dict[str, Any] = {}
        refs: Dict[str, str] = {}
        missing: List[str] = []

        for p in required:
            prop = props.get(p, {})
            res = _bind_param(p, prop, handles, user_text_handles)
            if res is None:
                # For read/search, allow unbound non-typed params (the
                # model is asked to select the action; the kernel will
                # leave the param empty if no handle exists). We still
                # record it as "missing" so the menu shows the gap, but
                # mark non-mutations executable when at least one bound
                # arg exists or the schema allows empty params.
                missing.append(p)
                continue
            value, ref, _h = res
            bound[p] = value
            refs[p] = ref

        # Sub-kind decided by tool prefix and binding state.
        if kind_class == "mutation":
            executable = (len(missing) == 0)
            kind = "mutation"
        elif name == respond_tool_name or name in _NEUTRAL_TOOLS:
            kind = "final" if name == respond_tool_name else "ask"
            executable = True
        else:
            # read / search
            kind = "search" if name.startswith(("search_", "find_", "lookup_")) else "read"
            # Read/search are executable when *all* required typed params bind
            # OR when the missing params are free-form strings we leave empty.
            # Simpler rule: executable iff every required param is bound.
            executable = (len(missing) == 0)

        sig = _sig(name, bound)
        if executable and kind == "mutation" and sig in executed:
            # Duplicate mutation suppression at the menu level.
            continue

        display = _format_display(kind, name, bound, missing, _short_desc(schema))
        out.append(Affordance(
            action_id=next_id(),
            kind=kind,
            display_text=display,
            tool_name=name,
            arguments=bound,
            argument_refs=refs,
            missing_requirements=missing,
            provenance={"binding": dict(refs), "remaining_steps": remaining_steps},
            is_executable=executable,
        ))

    # Always provide a "final" respond affordance.
    out.append(Affordance(
        action_id=next_id(),
        kind="final",
        display_text="final: produce a short final answer to the user.",
        tool_name=respond_tool_name,
        arguments={},
        argument_refs={},
        missing_requirements=[],
        provenance={"reason": "always_available_final"},
        is_executable=True,
    ))
    return out


def _sig(name: str, kwargs: Dict[str, Any]) -> str:
    parts = [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
    return name + "(" + ",".join(parts) + ")"


def _format_display(kind: str, name: str, bound: Dict[str, Any],
                    missing: List[str], desc: str) -> str:
    if not bound and not missing:
        argsig = ""
    elif missing and not bound:
        argsig = f"(missing: {', '.join(missing)})"
    elif missing:
        bound_str = ", ".join(f"{k}={bound[k]}" for k in bound)
        argsig = f"({bound_str}; missing: {', '.join(missing)})"
    else:
        argsig = "(" + ", ".join(f"{k}={bound[k]}" for k in bound) + ")"
    head = {"read": "read", "search": "search", "ask": "ask",
            "mutation": "mutation", "final": "final"}.get(kind, kind)
    base = f"{head}: {name}{argsig}"
    if desc:
        base = f"{base} — {desc}"
    return base


def rank_and_truncate(affordances: List[Affordance], k: int = 8,
                      remaining_steps: int = 30) -> List[Affordance]:
    """Rank cheaply and deterministically; keep the top-k."""

    def score(a: Affordance) -> Tuple[int, int, int]:
        # Lower tuple sorts first (highest priority).
        if a.kind == "mutation" and a.is_executable:
            primary = 0
        elif a.kind in ("read", "search") and a.is_executable:
            primary = 1
        elif a.kind == "ask":
            primary = 2
        elif a.kind == "final":
            # Final is high priority when budget is short.
            primary = 1 if remaining_steps <= 3 else 4
        else:
            primary = 3 if a.is_executable else 5
        # Secondary: more bound args first.
        secondary = -len(a.arguments)
        # Tertiary: stable tie-break on action_id numeric suffix.
        try:
            tertiary = int(a.action_id.lstrip("A"))
        except Exception:
            tertiary = 0
        return (primary, secondary, tertiary)

    ranked = sorted(affordances, key=score)
    final_idx = next((i for i, a in enumerate(ranked) if a.kind == "final"), None)
    top = ranked[: max(1, k)]
    if final_idx is not None and not any(a.kind == "final" for a in top):
        # Always make sure the menu carries a "final" option.
        top = top[:-1] + [ranked[final_idx]]
    return top


def render_menu_text(affordances: List[Affordance], remaining_steps: int) -> str:
    lines = ["Available verified actions:"]
    for a in affordances:
        marker = "" if a.is_executable else " [not executable: missing grounded args]"
        lines.append(f"{a.action_id} {a.display_text}{marker}")
    lines.append(f"Budget: {remaining_steps} steps left.")
    return "\n".join(lines)


# Public for tests
__all__ = [
    "Affordance",
    "classify_tool",
    "build_affordances",
    "rank_and_truncate",
    "render_menu_text",
]
