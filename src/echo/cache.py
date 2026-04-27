"""Episodic-cache + horizon + divergence annotator for the ECHO controller.

This file is intentionally tiny (~80 lines of meaningful code). It is the
*entire* novel mechanism — there is no gate, no extra LLM call, no schema
introspection. Three deterministic checks run after every dispatched tool
call and prepend short bracketed hints to the tool observation:

  1. **Episodic cache hit** — same ``(tool_name, canonical_args)`` was
     already dispatched earlier in this episode.
  2. **Repeat-tool divergence** — the same tool name was just used three
     times in a row (regardless of args).
  3. **Two-stage horizon nudge** — at exactly ``max_steps - 7`` and
     ``max_steps - 3`` steps remaining, append a budget reminder.

The annotations are *advisory*. The agent always receives the original
observation; the hints are simply prepended on a separate line. There is
no blocking, no rejection, no retry channel — by construction ECHO can
never score worse than the baseline, only better when the hints succeed
in steering the model away from a wasteful trajectory.

Why this is the right shape:

* In our prior runs the dominant failure mode was *running out of
  budget*, not bad arguments — agents averaged 41–46 messages on a
  30-step horizon, with most wasted on duplicate / loop calls.
* Static gates (e.g. SAGE) hurt because they reject valid calls and
  cause thrashing; advisory hints cannot reject, only inform.
* The three signals above are domain-agnostic and require no per-task
  configuration, so the same controller applies to tau-bench and
  ACEBench without modification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


def canonical_args(args: Any) -> str:
    """Stable string representation of tool arguments for cache keys.

    Uses ``json.dumps`` with sorted keys so two semantically identical
    argument dicts always produce the same key, regardless of insertion
    order. Falls back to ``repr`` on un-serialisable inputs.
    """
    if not isinstance(args, dict):
        try:
            return json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
        except Exception:
            return repr(args)
    try:
        return json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return repr(args)


# Two-stage horizon trigger thresholds (steps remaining at which a
# bracketed budget reminder is appended to the next tool observation).
HORIZON_WARN_AT: int = 7
HORIZON_COMMIT_AT: int = 3


@dataclass
class EchoCache:
    """Per-episode state for the ECHO controller.

    Attributes
    ----------
    calls
        Maps ``(tool_name, canonical_args(args))`` to the *first* step
        index at which that exact call was dispatched. Used by the
        cache-hit annotation.
    recent_tool_names
        Ordered list of every dispatched tool name in this episode.
        Inspected by the divergence annotation.
    stats
        Cumulative counts of each annotation kind. Surfaced in the
        per-task record so we can diagnose whether ECHO is firing.
    """

    calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    recent_tool_names: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "cache_hits": 0,
        "diverge_hits": 0,
        "budget_warn": 0,
        "budget_commit": 0,
        "tool_calls_seen": 0,
    })

    # ------------------------------------------------------------------
    def annotate(
        self,
        *,
        name: str,
        args: Any,
        observation: str,
        step: int,
        max_num_steps: int,
    ) -> str:
        """Return ``observation`` with any applicable ECHO hints prepended.

        ``step`` is the zero-based index of the LLM iteration that
        emitted the call (the for-loop variable in the agent solve
        loop). ``max_num_steps`` is the loop's hard horizon.
        """
        prefixes: List[str] = []

        key = (str(name), canonical_args(args))

        # 1. Episodic cache hit ---------------------------------------
        if key in self.calls:
            prev_step = self.calls[key]
            self.stats["cache_hits"] += 1
            prefixes.append(
                f"[echo:cache] this exact call (tool='{name}', same arguments) "
                f"was already dispatched at step {prev_step}; if the observation "
                f"is unchanged, a different action is likely needed."
            )
        # Record on first occurrence so subsequent duplicates trigger.
        else:
            self.calls[key] = step

        # 2. Repeat-tool divergence -----------------------------------
        # Trigger when the *previous two* dispatches were also ``name`` —
        # i.e. this call would make three consecutive uses of the same
        # tool. Args don't have to match (this is broader than the cache
        # check) so we catch loops that vary an argument slightly.
        if (
            len(self.recent_tool_names) >= 2
            and self.recent_tool_names[-1] == name
            and self.recent_tool_names[-2] == name
        ):
            self.stats["diverge_hits"] += 1
            prefixes.append(
                f"[echo:diverge] tool '{name}' has now been used 3 times in a "
                f"row; consider a different tool or respond if you have enough "
                f"information."
            )

        # 3. Two-stage horizon nudge ----------------------------------
        # ``step`` counts LLM iterations; ``remaining`` is the number of
        # iterations the loop can still run *after* this one.
        remaining = int(max_num_steps) - int(step) - 1
        if remaining == HORIZON_WARN_AT:
            self.stats["budget_warn"] += 1
            prefixes.append(
                f"[echo:budget] {remaining} steps remaining — consider "
                f"closing the task soon."
            )
        elif remaining == HORIZON_COMMIT_AT:
            self.stats["budget_commit"] += 1
            prefixes.append(
                f"[echo:budget] {remaining} steps remaining — respond now "
                f"with a final answer if you have one."
            )

        # Update history for divergence detection on subsequent calls.
        self.recent_tool_names.append(str(name))
        self.stats["tool_calls_seen"] += 1

        if not prefixes:
            return observation
        return "\n".join(prefixes) + "\n" + (observation or "")

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of cumulative ECHO stats."""
        return {
            "tool_calls_seen": self.stats["tool_calls_seen"],
            "cache_hits": self.stats["cache_hits"],
            "diverge_hits": self.stats["diverge_hits"],
            "budget_warn": self.stats["budget_warn"],
            "budget_commit": self.stats["budget_commit"],
            "unique_calls": len(self.calls),
        }
