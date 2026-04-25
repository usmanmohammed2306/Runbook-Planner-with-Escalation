"""Policy-bullet extraction for IG-RPE.

The domain wiki in tau-bench is a dense, multi-paragraph document. A 7B model
struggles to ground every call against the full text each turn. We convert
the wiki into a short list of imperative bullets *once* per task and inject
only the bullets whose keywords overlap the currently-proposed action.

The extraction is a best-effort regex pass with an optional LLM refinement
fallback. If anything fails we fall back to a small set of tau-bench-specific
rules that every evaluator has seen in the wiki.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


IMPERATIVE_PAT = re.compile(
    r"(?mi)^\s*(?:-|\*|\d+\.)\s*((?:you\s+(?:must|can|should|may)|never|always|do not|don't)[^.\n]{5,200}\.)"
)

TAU_RETAIL_DEFAULTS: List[str] = [
    "Always identify the user first via find_user_id_by_email or find_user_id_by_name_zip.",
    "Never modify or cancel an order without explicit user confirmation in the same session.",
    "Before modifying an order, call get_order_details to verify the current state.",
    "For exchanges, both the new and old items must share the same product_id.",
    "Gift-card refunds go back to the gift card; credit-card refunds to the original card.",
    "Do not retry a failed write with the same arguments. Diagnose first.",
]

TAU_AIRLINE_DEFAULTS: List[str] = [
    "Always verify the user (book reference or email) before changing a booking.",
    "Before cancelling or modifying a reservation, fetch its details.",
    "Obtain explicit user confirmation before making any non-refundable change.",
    "Do not retry a failed booking change with identical arguments.",
]


@dataclass
class Policy:
    bullets: List[str]

    def relevant(self, action_text: str, k: int = 4) -> List[str]:
        if not self.bullets:
            return []
        tokens = set(_tokens(action_text.lower()))
        scored = []
        for b in self.bullets:
            score = len(tokens & set(_tokens(b.lower())))
            scored.append((score, b))
        scored.sort(key=lambda x: -x[0])
        out = [b for (s, b) in scored if s > 0][:k]
        if not out:
            # Fall back to the first-k bullets which tend to be the most universal.
            out = self.bullets[:k]
        return out


def extract_policy(wiki_text: str, env_hint: str = "retail") -> Policy:
    """Extract imperative bullets from ``wiki_text``. Falls back to defaults."""
    bullets: List[str] = []
    if wiki_text:
        for m in IMPERATIVE_PAT.finditer(wiki_text):
            sent = m.group(1).strip()
            if sent and sent not in bullets:
                bullets.append(sent)
    if len(bullets) < 4:
        defaults = TAU_AIRLINE_DEFAULTS if env_hint.lower() == "airline" else TAU_RETAIL_DEFAULTS
        for b in defaults:
            if b not in bullets:
                bullets.append(b)
    # Cap aggressively — we only need the top-k per turn.
    return Policy(bullets=bullets[:20])


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z_]+", s)
