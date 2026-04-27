"""Typed handles for VALENCE — the only legal source of mutation arguments.

A handle wraps an exact value with provenance pointing back into the
EventLog. Mutation tool calls may only receive a handle's value (or the
value of a deterministic resolver applied to handles); raw model strings
never reach the env.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Confidence levels.
CONF_EXACT = "exact"          # came verbatim from a tool result or user span
CONF_RESOLVED = "resolved"    # produced by a deterministic resolver

# Standard handle types.
TYPES = (
    "user_id", "customer_id", "order_id", "reservation_id", "booking_id",
    "item_id", "product_id", "flight_id", "payment_method_id",
    "status", "money", "datetime", "enum",
    "string",  # generic exact text span (search query etc.)
)


@dataclass(frozen=True)
class Handle:
    handle_id: str
    type: str
    value: Any
    source_event_id: str
    source_path: str            # JSON path or "user_text[a:b]"
    confidence: str = CONF_EXACT
    resolver: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regex-based user-text minting
# ---------------------------------------------------------------------------
# Conservative patterns. False positives are tolerable for search/read; only
# mutation kwargs require typed-handle binding, and the schema-driven minting
# from tool observations is the primary source for those.
_RE_ORDER = re.compile(r"\b(?:order[_ -]?#?\s*)?(#?[Oo]\d{4,})\b")
_RE_RESERVATION = re.compile(r"\b(?:reservation[_ -]?#?\s*)?(#?[Rr]\d{4,})\b")
_RE_BOOKING = re.compile(r"\b(?:booking[_ -]?#?\s*)?(#?[Bb]\d{4,})\b")
_RE_USER = re.compile(r"\b([a-z]+_[a-z]+_\d{1,6})\b")  # tau-bench user_id pattern
_RE_EMAIL = re.compile(r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b")
_RE_MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
_RE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _strip(token: str) -> str:
    return token.lstrip("#").strip()


def mint_handles_from_user_text(text: str, event_id: str,
                                next_id: "_IdMinter") -> List[Handle]:
    out: List[Handle] = []
    if not text:
        return out
    s = str(text)

    def add(typ: str, val: str, m: "re.Match[str]") -> None:
        out.append(Handle(
            handle_id=next_id.next(typ),
            type=typ,
            value=_strip(val),
            source_event_id=event_id,
            source_path=f"user_text[{m.start()}:{m.end()}]",
            confidence=CONF_EXACT,
            provenance={"source": "user_text", "span": [m.start(), m.end()]},
        ))

    for m in _RE_ORDER.finditer(s):
        add("order_id", m.group(1), m)
    for m in _RE_RESERVATION.finditer(s):
        add("reservation_id", m.group(1), m)
    for m in _RE_BOOKING.finditer(s):
        add("booking_id", m.group(1), m)
    for m in _RE_USER.finditer(s):
        add("user_id", m.group(1), m)
    for m in _RE_EMAIL.finditer(s):
        add("string", m.group(1), m)  # email as exact string handle
    for m in _RE_MONEY.finditer(s):
        out.append(Handle(
            handle_id=next_id.next("money"),
            type="money",
            value=float(m.group(1)),
            source_event_id=event_id,
            source_path=f"user_text[{m.start()}:{m.end()}]",
            confidence=CONF_EXACT,
            provenance={"source": "user_text", "literal": m.group(0)},
        ))
    for m in _RE_DATE.finditer(s):
        out.append(Handle(
            handle_id=next_id.next("datetime"),
            type="datetime",
            value=m.group(1),
            source_event_id=event_id,
            source_path=f"user_text[{m.start()}:{m.end()}]",
            confidence=CONF_EXACT,
            provenance={"source": "user_text"},
        ))
    return out


# ---------------------------------------------------------------------------
# Tool-observation minting — walk JSON for typed _id keys, prices, dates.
# ---------------------------------------------------------------------------
_KEY_TO_TYPE: Dict[str, str] = {
    "user_id": "user_id",
    "customer_id": "customer_id",
    "order_id": "order_id",
    "reservation_id": "reservation_id",
    "booking_id": "booking_id",
    "item_id": "item_id",
    "product_id": "product_id",
    "flight_id": "flight_id",
    "payment_method_id": "payment_method_id",
    "payment_id": "payment_method_id",
    "status": "status",
    "price": "money",
    "amount": "money",
    "total": "money",
    "date": "datetime",
    "datetime": "datetime",
    "departure_date": "datetime",
    "return_date": "datetime",
}


def mint_handles_from_observation(observation: Any, event_id: str,
                                   next_id: "_IdMinter") -> List[Handle]:
    out: List[Handle] = []
    seen: set = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                kpath = f"{path}.{k}" if path else k
                typ = _KEY_TO_TYPE.get(k.lower())
                if typ and isinstance(v, (str, int, float)) and v not in (None, ""):
                    key = (typ, str(v))
                    if key not in seen:
                        seen.add(key)
                        out.append(Handle(
                            handle_id=next_id.next(typ),
                            type=typ,
                            value=v if typ in ("money",) and isinstance(v, (int, float)) else str(v),
                            source_event_id=event_id,
                            source_path=kpath,
                            confidence=CONF_EXACT,
                            provenance={"source": "tool_observation", "key": k},
                        ))
                walk(v, kpath)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(observation, "")
    return out


class _IdMinter:
    """Generates short, deterministic, type-prefixed handle IDs."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    def next(self, typ: str) -> str:
        self._counts[typ] = self._counts.get(typ, 0) + 1
        return f"H_{typ}_{self._counts[typ]}"


def index_by_type(handles: Iterable[Handle]) -> Dict[str, List[Handle]]:
    out: Dict[str, List[Handle]] = {}
    for h in handles:
        out.setdefault(h.type, []).append(h)
    return out


def find_handle_for_value(handles: Iterable[Handle],
                          value: Any) -> Optional[Handle]:
    """Return the first handle whose value matches `value` exactly."""
    sval = str(value)
    for h in handles:
        if str(h.value) == sval:
            return h
    return None
