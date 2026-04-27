"""Immutable evidence event log for VALENCE.

Every piece of grounding information the kernel can use must come from
exactly one source: an event in this log.  Handles, resolved tokens and
compiled mutation arguments all carry an ``source_event_id`` pointing
back into this log so that the provenance chain is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Event kinds — strings, deliberately not an Enum, to keep JSON-friendly.
USER_MESSAGE = "user_message"
ASSISTANT_CHOICE = "assistant_choice"
TRANSLATED_TOOL_CALL = "translated_tool_call"
TOOL_OBSERVATION = "tool_observation"
FINAL_ANSWER = "final_answer"


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    source: str            # "user" | "assistant" | "kernel" | "env"
    raw: Any               # original payload (string, dict, list)
    order: int             # monotonically increasing index
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Append-only immutable event log.

    The log is the single source of truth for VALENCE: every Handle and
    every ResolvedToken carries an event_id pointing to a record here.
    Events are never removed or rewritten.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def _next_id(self, kind: str) -> str:
        return f"E{len(self._events):04d}_{kind}"

    def append(self, kind: str, source: str, raw: Any,
               metadata: Optional[Dict[str, Any]] = None) -> Event:
        ev = Event(
            event_id=self._next_id(kind),
            kind=kind,
            source=source,
            raw=raw,
            order=len(self._events),
            metadata=dict(metadata or {}),
        )
        self._events.append(ev)
        return ev

    # Convenience appenders ---------------------------------------------------
    def add_user_message(self, text: str) -> Event:
        return self.append(USER_MESSAGE, "user", str(text or ""))

    def add_assistant_choice(self, action_id: str, raw_response: str = "") -> Event:
        return self.append(
            ASSISTANT_CHOICE, "assistant",
            {"action_id": action_id, "raw": raw_response},
        )

    def add_translated_tool_call(self, action_id: str, tool_name: str,
                                  kwargs: Dict[str, Any]) -> Event:
        return self.append(
            TRANSLATED_TOOL_CALL, "kernel",
            {"action_id": action_id, "tool_name": tool_name, "kwargs": dict(kwargs)},
        )

    def add_tool_observation(self, tool_name: str, kwargs: Dict[str, Any],
                             observation: Any) -> Event:
        return self.append(
            TOOL_OBSERVATION, "env",
            {"tool_name": tool_name, "kwargs": dict(kwargs), "observation": observation},
        )

    def add_final_answer(self, text: str) -> Event:
        return self.append(FINAL_ANSWER, "assistant", str(text or ""))

    # Lookup ------------------------------------------------------------------
    def get(self, event_id: str) -> Optional[Event]:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None
