"""VALENCE — Verified Affordance Lattice for Efficient Non-hallucinating Control.

Public surface:

* :class:`AffordanceKernel` — single enforcement point.
* :class:`EventLog`         — immutable evidence log.
* :class:`Handle`           — typed, provenance-bearing value.
* :class:`Affordance`       — one menu entry.
* :class:`CompiledAction`   — validated, executable form.

Mutation arguments must originate from a Handle or a deterministic
ResolvedToken; ungrounded mutations are rejected before they reach env.
"""
from __future__ import annotations

from .event_log import Event, EventLog
from .handles import Handle
from .lattice import Affordance
from .kernel import AffordanceKernel, TINY_SYSTEM_PROMPT
from .resolvers import ResolvedToken, resolve_date, resolve_enum, resolve_money, resolve_selector
from .transaction import CompiledAction, TransactionValidator, ValidationResult

__all__ = [
    "Affordance",
    "AffordanceKernel",
    "CompiledAction",
    "Event",
    "EventLog",
    "Handle",
    "ResolvedToken",
    "TINY_SYSTEM_PROMPT",
    "TransactionValidator",
    "ValidationResult",
    "resolve_date",
    "resolve_enum",
    "resolve_money",
    "resolve_selector",
]
