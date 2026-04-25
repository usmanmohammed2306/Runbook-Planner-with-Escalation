"""Symbolic conversation ledger for IG-RPE.

The ledger is a small, deterministic structure maintained across turns. It
records the facts the gate needs to evaluate invariants: which user has been
identified, which orders / resources have been fetched, what confirmations
the user has given, and which tool calls have already been made. The ledger
intentionally stores only *observable* facts — never LLM speculation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# Tokens that reliably indicate an affirmative user confirmation in tau-bench
# user-simulator responses. Kept short and conservative; falling back to
# negative heuristics elsewhere.
AFFIRMATIVE_TOKENS = (
    "yes", "yeah", "yep", "confirm", "confirmed", "go ahead", "please do",
    "sure", "that's right", "that is right", "correct", "do it", "proceed",
    "approve", "approved", "sounds good", "ok with that", "okay, go",
)

NEGATIVE_TOKENS = (
    "no,", "no ", "don't", "do not", "wait", "stop", "cancel that",
    "not yet", "hold on", "nope",
)

# Regex patterns for extracting common IDs out of observation text.
_ORDER_ID_RE = re.compile(r"#?\b([A-Z]{1,3}-\d{4,}|\bW\d{6,}|\bO\d{6,}|\border[_\s-]?\d{4,})\b", re.IGNORECASE)
_USER_ID_RE = re.compile(r"\b(user_id|userId)\D{0,4}([a-zA-Z0-9_\-]+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


@dataclass
class Ledger:
    """Append-only symbolic state accumulated from observations + actions."""

    user_ids_identified: Set[str] = field(default_factory=set)
    emails_seen: Set[str] = field(default_factory=set)
    orders_fetched: Set[str] = field(default_factory=set)
    orders_details_cache: Dict[str, str] = field(default_factory=dict)
    # List of (tool_name, args_hash, turn_index).
    tool_call_history: List[Tuple[str, str, int]] = field(default_factory=list)
    # List of (role, text, turn_index).
    user_messages: List[Tuple[str, int]] = field(default_factory=list)
    last_write_proposal: Optional[Dict[str, Any]] = None
    errors_by_tool: Dict[str, int] = field(default_factory=dict)

    # -- ingestion ------------------------------------------------------
    def note_user_message(self, text: str, turn: int) -> None:
        if not text:
            return
        self.user_messages.append((text, turn))
        self._extract_ids_from_text(text)

    def note_tool_call(self, name: str, args: Dict[str, Any], turn: int) -> None:
        h = _args_hash(args)
        self.tool_call_history.append((name, h, turn))

    def note_tool_observation(self, name: str, args: Dict[str, Any], obs_text: str, turn: int) -> None:
        """Ingest observation text and update symbolic facts."""
        # Errors — track so we can bail gracefully on repeated failure.
        if _looks_like_error(obs_text):
            self.errors_by_tool[name] = self.errors_by_tool.get(name, 0) + 1
            return

        # User identification: any tool that returned a user_id or the exact
        # find_user_* tools.
        lname = name.lower()
        if lname.startswith("find_user") or lname == "get_user_details" or "user" in lname:
            # Prefer the argument value if the tool returned "ok" / an id.
            for key in ("user_id", "userId"):
                v = args.get(key)
                if isinstance(v, str) and v.strip():
                    self.user_ids_identified.add(v.strip())
            # Also try to lift ids straight out of the observation.
            for m in _USER_ID_RE.finditer(obs_text or ""):
                self.user_ids_identified.add(m.group(2))
            if "@" in obs_text:
                for m in _EMAIL_RE.finditer(obs_text):
                    self.emails_seen.add(m.group(0))

        # Order-detail fetches: remember which orders we have grounded data for.
        if lname == "get_order_details" or "order_detail" in lname:
            oid = _stringify(args.get("order_id"))
            if oid:
                self.orders_fetched.add(oid)
                self.orders_details_cache[oid] = (obs_text or "")[:4000]
            # Some tools take an argument under a different key.
            for k in ("order", "orderId"):
                v = _stringify(args.get(k))
                if v:
                    self.orders_fetched.add(v)
                    self.orders_details_cache[v] = (obs_text or "")[:4000]

    def _extract_ids_from_text(self, text: str) -> None:
        for m in _ORDER_ID_RE.finditer(text or ""):
            pass  # Presence in *user text* isn't a fetch — don't mark fetched.
        for m in _EMAIL_RE.finditer(text or ""):
            self.emails_seen.add(m.group(0))

    # -- queries --------------------------------------------------------
    def user_verified(self) -> bool:
        return bool(self.user_ids_identified) or bool(self.emails_seen)

    def order_fetched(self, order_id: str) -> bool:
        return _stringify(order_id) in self.orders_fetched

    def recent_confirmation(self, window: int = 2) -> bool:
        """Has the user issued an affirmative confirmation in the last ``window`` user turns?"""
        recent = self.user_messages[-window:]
        for txt, _turn in recent:
            low = (txt or "").lower()
            if any(tok in low for tok in NEGATIVE_TOKENS):
                return False
            if any(tok in low for tok in AFFIRMATIVE_TOKENS):
                return True
        return False

    def is_duplicate_call(self, name: str, args: Dict[str, Any]) -> bool:
        """Have we already issued the exact same (name, args) payload this session?"""
        h = _args_hash(args)
        return any(n == name and a == h for (n, a, _t) in self.tool_call_history)

    def last_call_signature(self) -> Optional[Tuple[str, str]]:
        if not self.tool_call_history:
            return None
        n, h, _ = self.tool_call_history[-1]
        return (n, h)

    # -- serialization --------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "user_ids_identified": sorted(self.user_ids_identified),
            "emails_seen": sorted(self.emails_seen),
            "orders_fetched": sorted(self.orders_fetched),
            "tool_calls": [{"name": n, "args_hash": h, "turn": t} for (n, h, t) in self.tool_call_history],
            "errors_by_tool": dict(self.errors_by_tool),
            "user_verified": self.user_verified(),
        }


def _args_hash(args: Dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(args)


def _stringify(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _looks_like_error(obs: str) -> bool:
    if not obs:
        return False
    low = obs.lower()
    signals = (
        "error", "not found", "invalid", "forbidden", "unauthorized",
        "cannot be", "could not", "does not exist",
    )
    return any(s in low for s in signals)
