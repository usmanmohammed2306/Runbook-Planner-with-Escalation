"""ECHO — Episodic Cache + Horizon Orientation.

A lightweight, training-free, prompt-free *augmentation* of vanilla
tool-calling. ECHO never blocks a tool call; it only **annotates** tool
observations with three deterministic, domain-agnostic hints:

* ``[echo:cache]``    — this exact ``(tool, normalized_args)`` was already
  dispatched at an earlier step; if the observation is unchanged, a
  different action is likely needed.
* ``[echo:diverge]``  — the same tool name has been used three times in a
  row; consider a different tool or respond.
* ``[echo:budget]``   — fewer than ``H`` steps remain; consider closing
  soon (``H=7``) or respond now if confident (``H=3``).

The annotation is appended to the existing observation content, so the
agent is **never** denied information and the loop **cannot** dead-lock.
ECHO is therefore a strict superset of vanilla tool-calling: in the worst
case, the model ignores the hints and behaves identically to baseline.

This package contains the full controller — there is no separate gate
module, by design.
"""
from .cache import EchoCache, canonical_args

__all__ = ["EchoCache", "canonical_args"]
