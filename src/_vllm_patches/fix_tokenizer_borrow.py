"""Patch vLLM source to fix RuntimeError: Already borrowed and JSONDecodeError.

Root cause 1 — RuntimeError: Already borrowed
----------------------------------------------
vLLM creates a **new** tool-parser instance for every chat-completion request.
Each instantiation calls ``tokenizer.encode()`` to resolve special-token IDs.
The HuggingFace fast tokenizer (backed by the Rust ``tokenizers`` crate) uses a
``RefCell`` internally: ``no_truncation()`` / ``set_truncation_and_padding()``
require a *mutable* borrow, but vLLM's async engine may already hold a borrow
(e.g. while running prefill tokenization in the background).  Result:

    RuntimeError: Already borrowed

The bug appears at max-concurrency ≥ 1 because even a single request triggers
both the engine borrow and the parser-init borrow before the first one releases.

Root cause 2 — JSONDecodeError: Extra data
------------------------------------------
Qwen2.5-7B-Instruct (and other models) sometimes generates valid JSON followed
by extra text inside the ``<tool_call>`` block.  ``json.loads()`` rejects the
entire string with ``JSONDecodeError: Extra data``.

Three-pronged fix (idempotent — safe to apply multiple times)
-------------------------------------------------------------
1. **Tool parsers** (hermes, mistral, llama3_json, …): wrap every
   ``self.model_tokenizer.encode(...)`` call in a ``threading.Lock`` with a
   short exponential-backoff retry loop so the call waits for the other borrow
   to finish.

2. **engine/serving.py**: add a retry loop around ``tool_parser_cls(tokenizer)``
   itself, because errors can also surface there when the parser's ``__init__``
   calls encode indirectly.

3. **hermes_tool_parser.py** ``extract_tool_calls``: replace ``json.loads()``
   with a ``raw_decode()``-based helper that stops at the end of the first valid
   JSON object and silently discards any trailing content.

Usage (from run_project.sh)
---------------------------
    python src/_vllm_patches/fix_tokenizer_borrow.py /path/to/vllm-src-0.18.0
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "# _SAGE_BORROW_FIX_APPLIED"
MARKER_JSON = "# _SAGE_JSON_FIX_APPLIED"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_after_imports(src: str, block: str) -> str:
    """Insert ``block`` after the last top-level import/from statement."""
    lines = src.splitlines(keepends=True)
    last_import = 0
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            last_import = i
    lines.insert(last_import + 1, block)
    return "".join(lines)


# Code injected at module level in each tool-parser file.
_LOCK_BLOCK = """\

{marker}
import threading as _sage_threading
_SAGE_ENCODE_LOCK = _sage_threading.Lock()


def _sage_encode(tokenizer, *args, **kwargs):
    \"\"\"Thread-safe wrapper for tokenizer.encode() — retries on RefCell borrow conflicts.\"\"\"
    for _i in range(30):
        try:
            with _SAGE_ENCODE_LOCK:
                return tokenizer.encode(*args, **kwargs)
        except RuntimeError as _e:
            if "Already borrowed" not in str(_e) or _i >= 29:
                raise
            import time as _t
            _t.sleep(0.01 * (_i + 1))

""".format(marker=MARKER)


# ---------------------------------------------------------------------------
# Patch 1: tool parser files
# ---------------------------------------------------------------------------

def patch_tool_parser(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"  skip (already patched): {path.name}")
        return

    if "self.model_tokenizer.encode(" not in src:
        print(f"  skip (no encode pattern): {path.name}")
        return

    src = _insert_after_imports(src, _LOCK_BLOCK)
    src = src.replace(
        "self.model_tokenizer.encode(",
        "_sage_encode(self.model_tokenizer,",
    )
    path.write_text(src, encoding="utf-8")
    print(f"  patched: {path.name}")


# ---------------------------------------------------------------------------
# Patch 2: engine/serving.py — retry around tool_parser_cls(tokenizer)
# ---------------------------------------------------------------------------

# Pattern: one or more leading spaces, then "tool_parser = tool_parser_cls(tokenizer)"
_SERVING_PATTERN = re.compile(
    r"^( +)(tool_parser\s*=\s*tool_parser_cls\(tokenizer\))",
    re.MULTILINE,
)

def _serving_replacement(m: re.Match) -> str:
    indent = m.group(1)
    orig = m.group(2)
    i2 = indent + "    "
    i3 = indent + "        "
    return (
        f"{indent}{MARKER}\n"
        f"{indent}for _sage_retry in range(30):\n"
        f"{i2}try:\n"
        f"{i3}{orig}\n"
        f"{i3}break\n"
        f"{i2}except RuntimeError as _sage_e:\n"
        f"{i3}if 'Already borrowed' not in str(_sage_e) or _sage_retry >= 29:\n"
        f"{i3}    raise\n"
        f"{i3}import time as _sage_t; _sage_t.sleep(0.01 * (_sage_retry + 1))\n"
    )


def patch_serving(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"  skip (already patched): {path.name}")
        return

    new_src, count = _SERVING_PATTERN.subn(_serving_replacement, src)
    if count == 0:
        print(f"  skip (pattern not found): {path.name}")
        return
    path.write_text(new_src, encoding="utf-8")
    print(f"  patched: {path.name} ({count} site(s))")


# ---------------------------------------------------------------------------
# Patch 3: hermes_tool_parser.py — tolerate trailing content after JSON
# ---------------------------------------------------------------------------

# Helper injected at module level in hermes_tool_parser.py.
_JSON_HELPER_BLOCK = """\

{marker}
import json as _sage_json_mod


def _sage_json_load(s):
    \"\"\"Parse the first valid JSON object from *s*, ignoring trailing content.\"\"\"
    s = (s or "").strip()
    try:
        obj, _ = _sage_json_mod.JSONDecoder().raw_decode(s)
        return obj
    except _sage_json_mod.JSONDecodeError:
        return _sage_json_mod.loads(s)

""".format(marker=MARKER_JSON)

# Patterns: json.loads(match[0] if match[0] else match[1])
# and similar single-argument json.loads() calls inside extract_tool_calls.
# We replace every occurrence of json.loads( that follows the hermes regex
# match variable with _sage_json_load( — the helper is module-level so
# it's always in scope.
_JSON_LOADS_PATTERN = re.compile(
    r"\bjson\.loads\(\s*(match\[(?:0|1)\](?:\s+if\s+match\[(?:0|1)\]\s+else\s+match\[(?:0|1)\])?)\s*\)"
)


def patch_json_loads(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    if MARKER_JSON in src:
        print(f"  skip (already patched json): {path.name}")
        return

    new_src, count = _JSON_LOADS_PATTERN.subn(
        lambda m: f"_sage_json_load({m.group(1)})", src
    )
    if count == 0:
        print(f"  skip (json.loads pattern not found): {path.name}")
        return

    new_src = _insert_after_imports(new_src, _JSON_HELPER_BLOCK)
    path.write_text(new_src, encoding="utf-8")
    print(f"  patched json.loads→_sage_json_load: {path.name} ({count} site(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_PARSER_NAMES = (
    "hermes_tool_parser.py",
    "mistral_tool_parser.py",
    "llama3_json_tool_parser.py",
    "pythonic_tool_parser.py",
    "jamba_tool_parser.py",
    "internlm_tool_parser.py",
    "granite_20b_fc_tool_parser.py",
)

_SERVING_CANDIDATES = (
    "vllm/entrypoints/openai/engine/serving.py",
    "vllm/entrypoints/openai/serving_chat.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vllm_src_dir>")
        sys.exit(1)

    vllm_src = Path(sys.argv[1])
    if not vllm_src.exists():
        print(f"vLLM source not found at {vllm_src}; skipping patches.")
        sys.exit(0)

    print(f"Patching vLLM source at {vllm_src} ...")

    parsers_dir = vllm_src / "vllm" / "tool_parsers"
    if parsers_dir.exists():
        for name in _PARSER_NAMES:
            p = parsers_dir / name
            if p.exists():
                patch_tool_parser(p)
        # JSON fix only needed for hermes (the only parser that calls json.loads
        # on raw regex match groups).
        hermes = parsers_dir / "hermes_tool_parser.py"
        if hermes.exists():
            patch_json_loads(hermes)
    else:
        print(f"  tool_parsers dir not found: {parsers_dir}")

    for rel in _SERVING_CANDIDATES:
        p = vllm_src / rel
        if p.exists():
            patch_serving(p)

    print("Done.")


if __name__ == "__main__":
    main()
