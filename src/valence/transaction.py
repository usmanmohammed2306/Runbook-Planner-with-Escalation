"""Transaction validation for VALENCE-compiled mutation calls.

Single chokepoint: every mutation about to hit the env passes through
``validate_mutation``. If any kwarg lacks provenance or the call has
already been executed in this episode, the validator rejects it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class CompiledAction:
    action_id: str
    tool_name: str
    kwargs: Dict[str, Any]
    argument_refs: Dict[str, str]   # arg -> handle_id or resolver name
    kind: str                        # mutation | read | search | ask | final


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TransactionValidator:
    """Stateful validator: tracks executed mutation signatures."""

    def __init__(self) -> None:
        self._executed_mutations: Set[str] = set()
        self.rejected_ungrounded: int = 0
        self.rejected_duplicate: int = 0
        self.rejected_mismatch: int = 0
        self.rejected_compile: int = 0

    @staticmethod
    def signature(tool_name: str, kwargs: Dict[str, Any]) -> str:
        parts = [f"{k}={kwargs[k]!r}" for k in sorted(kwargs.keys())]
        return tool_name + "(" + ",".join(parts) + ")"

    def validate(self, compiled: CompiledAction,
                 *, expected_action_id: Optional[str] = None) -> ValidationResult:
        if compiled is None:
            self.rejected_compile += 1
            return ValidationResult(False, "no_compiled_action")
        if expected_action_id is not None and compiled.action_id != expected_action_id:
            self.rejected_mismatch += 1
            return ValidationResult(False,
                                    f"action_id_mismatch:{expected_action_id}!={compiled.action_id}")
        # Read/search/ask/final never reach mutation enforcement, but we still
        # record their execution. The hard invariant only applies to mutations.
        if compiled.kind != "mutation":
            return ValidationResult(True, "non_mutation_passthrough")

        # 1. Every kwarg must have a provenance ref.
        ungrounded = [k for k in compiled.kwargs.keys() if not compiled.argument_refs.get(k)]
        if ungrounded:
            self.rejected_ungrounded += 1
            return ValidationResult(False, f"ungrounded_args:{ungrounded}")

        # 2. Duplicate mutation rejection.
        sig = self.signature(compiled.tool_name, compiled.kwargs)
        if sig in self._executed_mutations:
            self.rejected_duplicate += 1
            return ValidationResult(False, f"duplicate_mutation:{sig}")

        return ValidationResult(True, "ok", diagnostics={"signature": sig})

    def record_execution(self, compiled: CompiledAction) -> None:
        if compiled.kind == "mutation":
            self._executed_mutations.add(self.signature(compiled.tool_name, compiled.kwargs))

    @property
    def executed_signatures(self) -> List[str]:
        return list(self._executed_mutations)

    def stats(self) -> Dict[str, int]:
        return {
            "rejected_ungrounded_mutations": self.rejected_ungrounded,
            "duplicate_mutation_rejections": self.rejected_duplicate,
            "rejected_mismatch": self.rejected_mismatch,
            "valence_compile_failures": self.rejected_compile,
            "executed_mutations": len(self._executed_mutations),
        }
