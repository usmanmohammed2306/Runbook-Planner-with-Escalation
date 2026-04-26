"""sitecustomize hook for the tau-bench baseline subprocess.

Python imports any module named ``sitecustomize`` on startup if it can find
one on ``sys.path``. We add this directory to PYTHONPATH only when launching
the baseline subprocess (see ``src/runners/tau_runner.py``). This file then
monkey-patches ``litellm.completion`` so that any per-message ``content``
field longer than ``MAX_TOOL_CHARS`` is truncated *before* the request is
sent to the model.

Why this is needed
------------------
tau-bench's reference ``ToolCallingAgent`` accumulates every tool observation
in the messages array verbatim. tau-retail tools such as
``list_all_product_types``, ``get_product_details`` and ``get_order_details``
can return many KB of JSON; over the default 30-step horizon the prompt can
exceed even the 32 K context of Qwen2.5-7B-Instruct, killing the run with a
``ContextWindowExceededError``.

The truncation is conservative (5000 chars per tool/user message, leaves the
system prompt untouched, never modifies assistant tool_calls) and only
activates when the total estimated prompt length is over a soft budget.
"""
from __future__ import annotations

import os
import sys

MAX_TOOL_CHARS = int(os.environ.get("TAU_PATCH_MAX_TOOL_CHARS", "3000"))
# Soft budget: only truncate individual oversized messages below this total.
# Hard budget: iteratively shrink until content fits (must leave headroom for
# tool schemas + system prompt + decode).  Tau-retail ships ~20 tool specs at
# ~6 000 tokens; at 1.5 chars/token that is ~9 000 chars NOT in the message
# content.  With a 16 384-token model:
#   available_content_tokens ≈ 16384 - 6000 (tools) - 1500 (system) - 1500 (decode)
#                            = 7384 tokens × 1.75 chars/token ≈ 12 922 chars
# We use 12 000 as the default to give a comfortable buffer.
SOFT_TOTAL_CHAR_BUDGET = int(os.environ.get("TAU_PATCH_SOFT_BUDGET", "20000"))
HARD_TOTAL_CHAR_BUDGET = int(os.environ.get("TAU_PATCH_HARD_BUDGET", "12000"))
KEEP_SYSTEM_FULL = True


def _truncate_to(content, cap: int):
    if not isinstance(content, str) or len(content) <= cap:
        return content
    head = max(200, cap - 200)
    return content[:head] + f"\n...[truncated {len(content) - head} chars to fit context]"


def _total_content_chars(messages):
    return sum(len(m.get("content") or "") if isinstance(m, dict) else 0 for m in messages)


def _shrink_pass(messages, recent_cap: int, old_cap: int, recent_keep: int):
    """Single truncation pass. System messages are preserved verbatim. The last
    ``recent_keep`` non-system messages keep ``recent_cap`` chars; older
    tool/user messages are capped to ``old_cap``."""
    last_keep_idx = max(0, len(messages) - recent_keep)
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        if role == "system" and KEEP_SYSTEM_FULL:
            out.append(m)
            continue
        if role in ("tool", "user"):
            cap = recent_cap if i >= last_keep_idx else old_cap
            content = m.get("content")
            if isinstance(content, str) and len(content) > cap:
                m = {**m, "content": _truncate_to(content, cap)}
        out.append(m)
    return out


def _shrink_messages(messages, extra_chars: int = 0):
    """Iteratively shrink until under the hard budget.

    Parameters
    ----------
    messages : list
        Chat message dicts to shrink.
    extra_chars : int
        Additional chars already "spent" by content outside the message list
        (e.g. the serialised ``tools`` JSON passed to the API).  This is
        subtracted from both budgets so the caller does not need to know about
        the internal thresholds.

    Strategy
    --------
    1. If total content is under the *effective* soft budget, only cap
       individual oversized tool/user observations.
    2. Normal pass: last 6 messages at ``MAX_TOOL_CHARS``, older at
       ``MAX_TOOL_CHARS // 5``.
    3. Aggressive pass: last 4 messages at ``MAX_TOOL_CHARS // 2``, older at
       300 chars.
    4. Last-resort: last 3 messages at 800 chars, everything else at 150.
    5. Extreme guard: keep system + last 2 messages at 400 chars each.

    The hard budget is sized so that message content + tool schemas + system
    prompt + decode stay under the model's 16 384-token limit.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    # Clamp extra_chars to avoid negative budgets on very large tool schemas.
    extra = max(0, int(extra_chars))
    soft = max(3000, SOFT_TOTAL_CHAR_BUDGET - extra)
    hard = max(2000, HARD_TOTAL_CHAR_BUDGET - extra)

    total = _total_content_chars(messages)
    if total <= soft:
        out = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            role = m.get("role")
            if role in ("tool", "user") and isinstance(m.get("content"), str) and len(m["content"]) > MAX_TOOL_CHARS:
                m = {**m, "content": _truncate_to(m["content"], MAX_TOOL_CHARS)}
            out.append(m)
        return out

    old_cap_normal = max(300, MAX_TOOL_CHARS // 5)
    out = _shrink_pass(messages, recent_cap=MAX_TOOL_CHARS, old_cap=old_cap_normal, recent_keep=6)
    if _total_content_chars(out) <= hard:
        return out

    aggressive_recent = max(600, MAX_TOOL_CHARS // 2)
    out = _shrink_pass(out, recent_cap=aggressive_recent, old_cap=300, recent_keep=4)
    if _total_content_chars(out) <= hard:
        return out

    # Last-resort: floor everything.
    out = _shrink_pass(out, recent_cap=800, old_cap=150, recent_keep=3)
    if _total_content_chars(out) <= hard:
        return out

    # Extreme guard: keep system + last 2 assistant+tool pairs only.
    out = _shrink_pass(out, recent_cap=400, old_cap=80, recent_keep=2)
    return out


def _install():
    try:
        import litellm  # noqa: F401
    except Exception:
        return  # litellm not installed — nothing to patch

    try:
        import litellm.main as _llmain
    except Exception:
        return

    if getattr(_llmain, "_tau_patched", False):
        return

    real_completion = _llmain.completion

    def patched_completion(*args, **kwargs):
        try:
            msgs = kwargs.get("messages")
            if msgs is not None:
                kwargs["messages"] = _shrink_messages(msgs)
        except Exception as e:  # never let the patch itself break the call
            print(f"[tau_patch] truncation skipped: {e}", file=sys.stderr)
        return real_completion(*args, **kwargs)

    _llmain.completion = patched_completion
    # litellm re-exports completion at the top level — also patch there.
    try:
        import litellm as _ll
        _ll.completion = patched_completion
    except Exception:
        pass
    _llmain._tau_patched = True
    print("[tau_patch] litellm.completion patched (max_tool_chars="
          f"{MAX_TOOL_CHARS}, soft_budget={SOFT_TOTAL_CHAR_BUDGET}, "
          f"hard_budget={HARD_TOTAL_CHAR_BUDGET})", file=sys.stderr)


_install()
