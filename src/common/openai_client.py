"""Thin wrapper around the OpenAI SDK pointed at our local vLLM server.

Adds two pragmatic safeguards that the bare ``OpenAI`` client doesn't have:

1. **Pre-flight message truncation** — every controller (baseline / Act /
   ReAct / ECHO) calls the OpenAI SDK directly here, not via ``litellm``,
   so the sitecustomize patch that protects tau-bench's user simulator
   does NOT protect these direct calls. We monkey-patch
   ``client.chat.completions.create`` to call the same ``_shrink_messages``
   helper before sending, keeping the prompt under the model's context cap.
2. **Tighter default timeout** — the SDK's default is 600 s; on a busy
   single-GPU vLLM server, a stuck request that can't possibly fit (context
   overflow that bypassed truncation) blocks the loop for the full timeout.
   60 s is enough headroom for a normal Qwen2.5-7B response on 1×A100 and
   short enough that genuine hangs surface quickly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


_PATCH_LOADED = False
_shrink_messages = None  # populated lazily from sitecustomize


def _load_shrink_helper():
    """Import the ``_shrink_messages`` helper from the sitecustomize patch.

    We use ``importlib`` rather than a normal import so this works regardless
    of whether the sitecustomize directory is on sys.path at import time.
    """
    global _PATCH_LOADED, _shrink_messages
    if _PATCH_LOADED:
        return _shrink_messages
    _PATCH_LOADED = True
    try:
        repo_root = Path(__file__).resolve().parents[2]
        patch_file = repo_root / "src" / "_taubench_patches" / "sitecustomize.py"
        if not patch_file.exists():
            return None
        spec = importlib.util.spec_from_file_location("_taubench_sitecustomize", patch_file)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _shrink_messages = getattr(mod, "_shrink_messages", None)
        return _shrink_messages
    except Exception as exc:  # never let this break client construction
        print(f"[openai_client] shrink helper not loaded: {exc}", file=sys.stderr)
        return None


def _tools_chars(tools: Any) -> int:
    """Estimate the char footprint of the tools JSON sent to the API."""
    if not tools:
        return 0
    try:
        return len(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0


def _wrap_create(real_create):
    shrink = _load_shrink_helper()
    if shrink is None:
        return real_create

    def patched_create(*args: Any, **kwargs: Any):
        try:
            msgs = kwargs.get("messages")
            if msgs is not None:
                # Pass the tools JSON size so _shrink_messages can reduce its
                # effective budget accordingly.  This prevents overflows caused
                # by large tool schemas (tau-retail ships ~20 specs at ~6K
                # tokens) that are NOT counted in the message content chars.
                extra = _tools_chars(kwargs.get("tools"))
                kwargs["messages"] = shrink(msgs, extra_chars=extra)
        except Exception as exc:
            print(f"[openai_client] truncation skipped: {exc}", file=sys.stderr)
        return real_create(*args, **kwargs)

    return patched_create


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
        timeout=float(os.environ.get("OPENAI_TIMEOUT", "60")),
    )
    # Patch the bound method so callers don't need to know.
    completions = client.chat.completions
    if not getattr(completions, "_rpe_truncation_wrapped", False):
        completions.create = _wrap_create(completions.create)  # type: ignore[assignment]
        completions._rpe_truncation_wrapped = True  # type: ignore[attr-defined]
    return client
