"""Baseline controllers used for the four-way comparison.

The package contains three reference policies that share a single in-process
tool-calling loop. Differences live ONLY in the system prompt, so the
comparison against IG-RPE varies a single axis (the controller) and keeps
the model, tools, max-steps, temperature and truncation identical.

- ``ToolCallingAgent``: vanilla tool-calling. The system prompt only states
  the role and the policy; no reasoning style is enforced. This corresponds
  to tau-bench's ``tool-calling`` strategy at the prompt level.
- ``ActAgent``: action-only. The model is instructed to skip reasoning prose
  and emit tool calls directly. Mirrors the "Act" ablation from the ReAct
  paper (Yao et al., 2022).
- ``ReActAgent``: thought-then-action. The model is instructed to write one
  short ``Thought:`` line in its assistant content before each tool call,
  mirroring "ReAct" interleaved reasoning + acting.
"""
from __future__ import annotations

from .agents import ActAgent, ReActAgent, ToolCallingAgent

__all__ = ["ActAgent", "ReActAgent", "ToolCallingAgent"]
