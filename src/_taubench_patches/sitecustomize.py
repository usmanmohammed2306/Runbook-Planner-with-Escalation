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

MAX_TOOL_CHARS = int(os.environ.get("TAU_PATCH_MAX_TOOL_CHARS", "5000"))
SOFT_TOTAL_CHAR_BUDGET = int(os.environ.get("TAU_PATCH_SOFT_BUDGET", "80000"))
KEEP_SYSTEM_FULL = True


def _truncate_one(content):
    if not isinstance(content, str):
        return content
    if len(content) <= MAX_TOOL_CHARS:
        return content
    head = MAX_TOOL_CHARS - 200
    return content[:head] + f"\n...[truncated {len(content) - head} chars to fit context]"


def _shrink_messages(messages):
    if not isinstance(messages, list) or not messages:
        return messages
    total = sum(len(m.get("content") or "") if isinstance(m, dict) else 0 for m in messages)
    if total <= SOFT_TOTAL_CHAR_BUDGET:
        # Only enforce per-message cap on tool/user messages even when
        # under budget — a single 200KB observation must always be cut.
        out = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            role = m.get("role")
            if role in ("tool", "user") and isinstance(m.get("content"), str) and len(m["content"]) > MAX_TOOL_CHARS:
                m = {**m, "content": _truncate_one(m["content"])}
            out.append(m)
        return out

    # Over budget — be more aggressive. Keep system message untouched, keep
    # the last 12 messages, truncate every other tool/user content hard.
    out = []
    last_keep = len(messages) - 12
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        if role == "system" and KEEP_SYSTEM_FULL:
            out.append(m)
            continue
        if role in ("tool", "user"):
            cap = MAX_TOOL_CHARS if i >= last_keep else max(800, MAX_TOOL_CHARS // 4)
            content = m.get("content")
            if isinstance(content, str) and len(content) > cap:
                head = cap - 200
                m = {**m, "content": content[:head] + f"\n...[truncated {len(content) - head} chars]"}
        out.append(m)
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
          f"{MAX_TOOL_CHARS}, soft_budget={SOFT_TOTAL_CHAR_BUDGET})", file=sys.stderr)


_install()
