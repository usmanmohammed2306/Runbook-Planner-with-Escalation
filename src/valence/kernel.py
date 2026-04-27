"""AffordanceKernel — the single enforcement point for VALENCE.

The kernel is the only object that can produce a CompiledAction. Outside
this module, nothing is allowed to construct mutation arguments from
free-form model output. The agent calls ``parse_choice``,
``compile_action`` and ``validate_mutation`` in that order; if any step
fails, the kernel returns a fail-closed result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .event_log import EventLog
from .handles import (
    Handle,
    _IdMinter,
    mint_handles_from_observation,
    mint_handles_from_user_text,
)
from .lattice import (
    Affordance,
    build_affordances,
    classify_tool,
    rank_and_truncate,
    render_menu_text,
)
from .transaction import CompiledAction, TransactionValidator, ValidationResult


# Compact tiny prompt — never embed VALENCE theory.
TINY_SYSTEM_PROMPT = (
    "You must choose exactly one verified action_id from the menu. "
    "Do not invent tool arguments. If no executable mutation is available, "
    "choose read/search/ask/final. Return only JSON: {\"action_id\":\"A1\"}."
)


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_action_id(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    # Strip code fences
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    # Try direct JSON parse first.
    for candidate in (s,) + tuple(_JSON_OBJ_RE.findall(s)):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            aid = obj.get("action_id") or obj.get("id") or obj.get("action")
            if isinstance(aid, str) and aid:
                return aid.strip()
    # Last resort: regex for An token.
    m = re.search(r"\bA\d+\b", s)
    return m.group(0) if m else None


@dataclass
class KernelStats:
    action_menu_sizes: List[int] = field(default_factory=list)
    grounded_mutations: int = 0
    rejected_ungrounded: int = 0
    duplicate_rejections: int = 0
    compile_failures: int = 0
    mutation_attempts: int = 0
    total_actions: int = 0

    def snapshot(self, validator_stats: Dict[str, int]) -> Dict[str, Any]:
        n_attempts = max(self.mutation_attempts, 1)
        return {
            "action_menu_size_mean": (sum(self.action_menu_sizes) / len(self.action_menu_sizes))
                if self.action_menu_sizes else 0.0,
            "grounded_mutation_rate": self.grounded_mutations / n_attempts,
            "rejected_ungrounded_mutations": validator_stats.get(
                "rejected_ungrounded_mutations", 0),
            "duplicate_mutation_rejections": validator_stats.get(
                "duplicate_mutation_rejections", 0),
            "valence_compile_failures": self.compile_failures,
            "mutation_attempts": self.mutation_attempts,
            "total_actions": self.total_actions,
        }


class AffordanceKernel:
    """Single source of truth for what the LLM may execute.

    Lifecycle per step::

        kernel.ingest_user_message(text)             # once at start; also for re-prompts
        kernel.ingest_tool_result(name, kwargs, obs) # after each env.step
        affs = kernel.build_affordances(tool_schemas, remaining_steps)
        menu_text = kernel.render_menu(affs, k=8)
        # ... LLM call returns text ...
        action_id = kernel.parse_choice(text)
        compiled  = kernel.compile_action(action_id)
        result    = kernel.validate_mutation(compiled)
    """

    def __init__(self, *, respond_tool_name: str = "respond") -> None:
        self.event_log = EventLog()
        self.handles: List[Handle] = []
        self.user_text_handles: List[Handle] = []
        self.validator = TransactionValidator()
        self._id_minter = _IdMinter()
        self._last_menu: List[Affordance] = []
        self._action_index: Dict[str, Affordance] = {}
        self.respond_tool_name = respond_tool_name
        self.stats = KernelStats()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_user_message(self, text: str) -> None:
        ev = self.event_log.add_user_message(text)
        new = mint_handles_from_user_text(text, ev.event_id, self._id_minter)
        # User-text handles are kept in their own list so binding can
        # distinguish typed tool-evidence from raw user spans.
        self.user_text_handles.extend(new)

    def ingest_assistant_choice(self, action_id: str, raw: str = "") -> None:
        self.event_log.add_assistant_choice(action_id, raw)

    def ingest_tool_result(self, tool_name: str, kwargs: Dict[str, Any],
                           observation: Any) -> None:
        ev = self.event_log.add_tool_observation(tool_name, kwargs, observation)
        new = mint_handles_from_observation(observation, ev.event_id, self._id_minter)
        self.handles.extend(new)

    def mint_handles(self) -> List[Handle]:
        """Return the union of typed handles + user-text handles (read-only view)."""
        return list(self.handles) + list(self.user_text_handles)

    # ------------------------------------------------------------------
    # Lattice / menu
    # ------------------------------------------------------------------
    def build_affordances(self, tool_schemas: List[Dict[str, Any]],
                          remaining_steps: int = 30) -> List[Affordance]:
        all_affs = build_affordances(
            tool_schemas=tool_schemas,
            handles=self.handles,
            user_text_handles=self.user_text_handles,
            executed_signatures=self.validator.executed_signatures,
            remaining_steps=remaining_steps,
            respond_tool_name=self.respond_tool_name,
        )
        return all_affs

    def render_menu(self, affordances: List[Affordance], *,
                    k: int = 8, remaining_steps: int = 30) -> str:
        top = rank_and_truncate(affordances, k=k, remaining_steps=remaining_steps)
        self._last_menu = list(top)
        self._action_index = {a.action_id: a for a in top}
        self.stats.action_menu_sizes.append(len(top))
        return render_menu_text(top, remaining_steps)

    # ------------------------------------------------------------------
    # Choice → CompiledAction
    # ------------------------------------------------------------------
    @staticmethod
    def parse_choice(model_response: str) -> Optional[str]:
        return _extract_action_id(model_response)

    def compile_action(self, action_id: Optional[str]) -> Optional[CompiledAction]:
        """Compile a model-chosen action_id into a CompiledAction.

        Hallucinated IDs (not present in the rendered menu) cannot compile.
        Non-executable affordances also cannot compile.
        """
        if action_id is None:
            self.stats.compile_failures += 1
            return None
        aff = self._action_index.get(action_id)
        if aff is None:
            self.stats.compile_failures += 1
            return None
        if not aff.is_executable:
            self.stats.compile_failures += 1
            return None
        compiled = CompiledAction(
            action_id=aff.action_id,
            tool_name=aff.tool_name or self.respond_tool_name,
            kwargs=dict(aff.arguments),
            argument_refs=dict(aff.argument_refs),
            kind=aff.kind,
        )
        return compiled

    def validate_mutation(self, compiled: Optional[CompiledAction]) -> ValidationResult:
        if compiled is None:
            return ValidationResult(False, "no_compiled_action")
        if compiled.kind == "mutation":
            self.stats.mutation_attempts += 1
        result = self.validator.validate(compiled)
        if result.ok and compiled.kind == "mutation":
            self.stats.grounded_mutations += 1
        return result

    def record_execution(self, compiled: CompiledAction) -> None:
        self.validator.record_execution(compiled)
        self.event_log.add_translated_tool_call(
            compiled.action_id, compiled.tool_name, compiled.kwargs)
        self.stats.total_actions += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return self.stats.snapshot(self.validator.stats())
