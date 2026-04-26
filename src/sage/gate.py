"""Deterministic gate for SAGE.

Three independent checkers compose into a single ``sage_gate(...)`` call:

* :func:`check_schema`     — JSONSchema validation (required, types, enums).
* :func:`check_provenance` — every identifier-shaped string argument must
  appear in the conversation corpus.
* :func:`check_idempotency` — duplicate calls and post-error retries are
  rejected.

Each checker returns a list of failure tags so the structured feedback can
tell the LLM *which* invariant tripped and *why*. The gate is intentionally
conservative: free-form natural-language arguments (``content``, ``message``,
long prose) are NEVER provenance-checked, only identifier-shaped strings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Conversation corpus — accumulated text the agent has seen so far.
# ---------------------------------------------------------------------------
@dataclass
class ConversationCorpus:
    """Lower-cased concatenation of every observable text block.

    Built fresh from ``messages`` on every gate call. Used to test whether
    a candidate argument value has provenance in context. Storage is one
    big string + a token set so substring lookup is O(len(value))."""

    text: str = ""
    enum_values: Set[str] = field(default_factory=set)

    def contains(self, value: str) -> bool:
        if not value:
            return False
        v = value.strip().lower()
        if not v:
            return False
        # Direct substring hit in any observable block.
        if v in self.text:
            return True
        # Or a known enum value (declared in the tool schema itself).
        return v in self.enum_values


def _msg_text(msg: Dict[str, Any]) -> str:
    """Extract every text-bearing field from a chat message."""
    parts: List[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
    # Tool-call argument JSON is also fair game (a prior call can ground a
    # later one — e.g., the user said "cancel order #W123" and the model
    # passed it as an argument).
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        a = fn.get("arguments")
        if isinstance(a, str):
            parts.append(a)
    return " \n ".join(p for p in parts if p)


def build_corpus(
    messages: Iterable[Dict[str, Any]],
    tool_specs: Optional[List[Dict[str, Any]]] = None,
) -> ConversationCorpus:
    """Assemble the lookup corpus from history + every enum in every tool."""
    blocks: List[str] = []
    for m in messages or []:
        t = _msg_text(m)
        if t:
            blocks.append(t)
    text = (" \n ".join(blocks)).lower()

    enums: Set[str] = set()
    for ts in tool_specs or []:
        fn = ts.get("function") if isinstance(ts, dict) else None
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params, dict) else {}
        if not isinstance(props, dict):
            continue
        for prop in props.values():
            if not isinstance(prop, dict):
                continue
            ev = prop.get("enum")
            if isinstance(ev, list):
                for v in ev:
                    if v is None:
                        continue
                    enums.add(str(v).strip().lower())
    return ConversationCorpus(text=text, enum_values=enums)


# ---------------------------------------------------------------------------
# Identifier heuristic — which string args deserve provenance enforcement?
# ---------------------------------------------------------------------------
_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._\-/@+:]*$"
)
_HAS_DIGIT_OR_SPECIAL = re.compile(r"[\d@/_\-#:]")


def looks_like_identifier(s: str) -> bool:
    """Return True if ``s`` should be provenance-checked.

    Heuristic: short, no-whitespace strings that look like IDs / emails /
    order numbers / codes. Free-form prose ("Yes please cancel my order")
    fails this and is NOT checked, so the gate stays out of the way for
    legitimate natural-language arguments.
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) < 3 or len(s) > 80:
        return False
    if " " in s:
        return False
    # Must have at least one digit OR a structural special char to qualify
    # as identifier-shaped. Pure alpha words ("retail", "true") are skipped.
    if not _HAS_DIGIT_OR_SPECIAL.search(s):
        # ALL-CAPS short codes ("USD", "EU") still qualify as codes.
        if not (s.isupper() and len(s) <= 5):
            return False
    return bool(_IDENTIFIER_RE.match(s))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def _matches_type(value: Any, ptype: str) -> bool:
    if ptype == "string":
        return isinstance(value, str)
    if ptype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ptype == "boolean":
        return isinstance(value, bool)
    if ptype == "array":
        return isinstance(value, list)
    if ptype == "object":
        return isinstance(value, dict)
    if ptype == "null":
        return value is None
    return True


def check_schema(tool_spec: Optional[Dict[str, Any]], args: Dict[str, Any]) -> List[str]:
    """Return list of failure tags. Empty list means schema OK."""
    if not isinstance(tool_spec, dict):
        return []
    fn = tool_spec.get("function")
    if not isinstance(fn, dict):
        return []
    params = fn.get("parameters")
    if not isinstance(params, dict):
        return []
    properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = params.get("required") if isinstance(params.get("required"), list) else []

    failures: List[str] = []
    for r in required:
        if r not in args:
            failures.append(f"missing_required:{r}")

    for k, v in args.items():
        prop = properties.get(k) if isinstance(properties, dict) else None
        if not isinstance(prop, dict):
            continue
        ptype = prop.get("type")
        if isinstance(ptype, str) and not _matches_type(v, ptype):
            failures.append(f"type_mismatch:{k}:expected_{ptype}")
        elif isinstance(ptype, list):
            if not any(_matches_type(v, t) for t in ptype if isinstance(t, str)):
                failures.append(f"type_mismatch:{k}:expected_{'|'.join(ptype)}")
        enum_vals = prop.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            try:
                if v not in enum_vals:
                    sample = enum_vals[:5]
                    failures.append(f"enum_violation:{k}:not_in_{sample}")
            except Exception:
                pass
    return failures


# ---------------------------------------------------------------------------
# Provenance grounding
# ---------------------------------------------------------------------------
def check_provenance(
    corpus: ConversationCorpus,
    args: Dict[str, Any],
) -> List[str]:
    """Every identifier-shaped string arg must appear in the corpus."""
    failures: List[str] = []
    for k, v in args.items():
        if isinstance(v, list):
            for item in v:
                if looks_like_identifier(item) and not corpus.contains(item):
                    failures.append(f"ungrounded_arg:{k}:{_short(item)}")
        elif looks_like_identifier(v) and not corpus.contains(v):
            failures.append(f"ungrounded_arg:{k}:{_short(v)}")
    return failures


def _short(s: Any, n: int = 32) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


# ---------------------------------------------------------------------------
# Idempotency + error budget
# ---------------------------------------------------------------------------
def _normalize_args(args: Dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(args)


def check_idempotency(
    history: List[Tuple[str, str]],
    error_counts: Dict[str, int],
    tool_name: str,
    args: Dict[str, Any],
    error_budget: int = 2,
) -> List[str]:
    failures: List[str] = []
    norm = _normalize_args(args)
    if any(n == tool_name and a == norm for (n, a) in history):
        failures.append(f"duplicate_call:{tool_name}")
    if error_counts.get(tool_name, 0) >= error_budget:
        failures.append(f"error_budget_exceeded:{tool_name}")
    return failures


# ---------------------------------------------------------------------------
# Combined gate
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    allow: bool
    feedback: str
    checks_failed: List[str]


_FEEDBACK_HINTS: Dict[str, str] = {
    "missing_required": "Required argument missing. Re-issue with the missing field populated.",
    "type_mismatch": "Argument has the wrong type. Check the tool schema.",
    "enum_violation": "Argument is not one of the allowed enum values. Pick a valid option from the schema.",
    "ungrounded_arg": (
        "Argument value has no provenance — it does not appear in the user message, prior tool results, "
        "system prompt, or schema enum. Either fetch the value from a prior tool first, or ask the user."
    ),
    "duplicate_call": "This (tool, args) pair was already issued. Don't repeat — interpret the prior result.",
    "error_budget_exceeded": "This tool has errored repeatedly with similar args. Diagnose first; do not retry blindly.",
}


def _explain(tags: List[str]) -> str:
    if not tags:
        return ""
    seen: Set[str] = set()
    lines: List[str] = []
    for t in tags:
        head = t.split(":", 1)[0]
        if head in seen:
            continue
        seen.add(head)
        hint = _FEEDBACK_HINTS.get(head, "")
        if hint:
            lines.append(f"- {head}: {hint}")
    return "\n".join(lines)


def sage_gate(
    *,
    messages: List[Dict[str, Any]],
    tool_specs: List[Dict[str, Any]],
    tool_spec: Optional[Dict[str, Any]],
    tool_name: str,
    args: Dict[str, Any],
    history: List[Tuple[str, str]],
    error_counts: Dict[str, int],
    corpus: Optional[ConversationCorpus] = None,
) -> GateResult:
    """Run schema + provenance + idempotency on a proposed call."""
    if corpus is None:
        corpus = build_corpus(messages, tool_specs)

    schema_fail = check_schema(tool_spec, args)
    prov_fail = check_provenance(corpus, args)
    idem_fail = check_idempotency(history, error_counts, tool_name, args)

    failures = schema_fail + prov_fail + idem_fail
    if not failures:
        return GateResult(allow=True, feedback="", checks_failed=[])

    feedback_obj = {
        "sage_blocked": True,
        "tool": tool_name,
        "checks_failed": failures,
        "guidance": _explain(failures),
        "next_step": (
            "Re-examine the conversation, fetch any missing IDs via a READ tool, "
            "or ask the user. Then issue a corrected call. You have ONE retry."
        ),
    }
    try:
        feedback = json.dumps(feedback_obj, ensure_ascii=False)
    except Exception:
        feedback = str(feedback_obj)
    return GateResult(allow=False, feedback=feedback, checks_failed=failures)
