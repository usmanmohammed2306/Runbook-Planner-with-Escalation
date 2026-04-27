"""Patch vLLM source to fix two runtime errors observed with Qwen2.5-7B.

Errors fixed
============
1. ``RuntimeError: Already borrowed`` raised from the HuggingFace fast
   tokenizer (Rust ``RefCell``) when vLLM creates a new tool-parser instance
   per request. We wrap ``self.model_tokenizer.encode(...)`` calls in a
   ``threading.Lock`` + retry loop, and we also retry around
   ``tool_parser_cls(tokenizer)`` in ``serving.py``.

2. ``json.JSONDecodeError: Extra data`` from
   ``hermes_tool_parser.extract_tool_calls`` when the model emits valid JSON
   followed by trailing text inside a ``<tool_call>`` block. We replace the
   bare ``json.loads(match[...])`` with ``raw_decode()``-based parsing that
   tolerates trailing content.

Design (this version is robust against multi-line ``from … import (…)``)
========================================================================
Earlier patcher revisions injected a multi-line helper block after "the last
import line", which silently corrupted files whose last import is a
parenthesised multi-line one. To eliminate that class of bugs entirely:

* All helper code lives in a **sibling helper module**
  ``vllm/tool_parsers/_sage_patch_helpers.py`` that the patcher writes
  fresh on every run.
* Each target parser file is modified with **only** two safe operations:
    1. ``str.replace(...)`` for the call sites (preserves indentation),
    2. **append** a single ``from … import …`` line at the end of the file.
  Both are immune to whatever import structure exists above.
* For ``serving.py`` we keep the regex line-rewrite, since it operates on a
  single line at fixed indentation.

Self-healing
============
Every target file has a ``.sage_orig`` backup. On every run the patcher:

1. Tries the local file first; if it doesn't parse and the backup parses,
   we use the backup as the clean source.
2. If neither parses, we download the canonical v0.18.0 source from
   GitHub raw (no auth needed) and use that as the clean source.
3. We always overwrite the backup with the verified-clean source so a
   future run cannot inherit a corrupted backup.
4. We apply edits, validate with ``ast.parse`` and only commit on success.

Usage
=====
    python src/_vllm_patches/fix_tokenizer_borrow.py /path/to/vllm-src-0.18.0
"""
from __future__ import annotations

import ast
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER = "# _SAGE_FIX_APPLIED"  # appears in the appended import line
SERVING_MARKER = "# _SAGE_BORROW_FIX_APPLIED"
BACKUP_SUFFIX = ".sage_orig"
HELPER_MODULE_NAME = "_sage_patch_helpers"

# The pinned vLLM tag whose tool_parsers/ files we know are syntactically
# valid. Used as a last-resort recovery source.
VLLM_TAG = "v0.18.0"
GITHUB_RAW_BASE = (
    f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_TAG}/vllm"
)

# Files we patch — name → relative-path-under-vllm/
PARSER_FILES = {
    "hermes_tool_parser.py": "tool_parsers/hermes_tool_parser.py",
    "mistral_tool_parser.py": "tool_parsers/mistral_tool_parser.py",
    "pythonic_tool_parser.py": "tool_parsers/pythonic_tool_parser.py",
    "jamba_tool_parser.py": "tool_parsers/jamba_tool_parser.py",
    "granite_20b_fc_tool_parser.py": "tool_parsers/granite_20b_fc_tool_parser.py",
    # llama3_json_tool_parser.py and internlm_tool_parser.py do not exist
    # at v0.18.0 and may be local-only forks; we skip them silently if absent.
}

SERVING_CANDIDATES = (
    "vllm/entrypoints/openai/engine/serving.py",
    "vllm/entrypoints/openai/serving_chat.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
)

# ---------------------------------------------------------------------------
# Helper module written into vllm/tool_parsers/_sage_patch_helpers.py
# ---------------------------------------------------------------------------

HELPER_MODULE_SRC = '''\
"""Runtime helpers used by SAGE patches to vLLM's tool parsers.

Defined in a sibling module so we never have to inject multi-line code into
parser files (which historically corrupted files with multi-line imports).
"""
from __future__ import annotations

import json as _json
import threading as _threading
import time as _time

_ENCODE_LOCK = _threading.Lock()


def _sage_encode(tokenizer, *args, **kwargs):
    """Thread-safe wrapper for tokenizer.encode().

    HuggingFace fast tokenizers (Rust ``RefCell``) raise ``RuntimeError:
    Already borrowed`` when vLLM's async engine and a parser-init both touch
    the tokenizer. Lock + short exponential backoff resolves the race.
    """
    last_exc = None
    for i in range(30):
        try:
            with _ENCODE_LOCK:
                return tokenizer.encode(*args, **kwargs)
        except RuntimeError as exc:
            last_exc = exc
            if "Already borrowed" not in str(exc):
                raise
            _time.sleep(0.01 * (i + 1))
    raise last_exc  # type: ignore[misc]


def _sage_json_load(s):
    """Parse the first valid JSON object in *s*, ignoring trailing garbage.

    Qwen2.5-7B-Instruct sometimes emits trailing text inside <tool_call>
    blocks. ``json.loads`` rejects the whole string with
    ``JSONDecodeError: Extra data``; ``raw_decode`` returns at the end of the
    first valid JSON value.
    """
    s = (s or "").strip()
    try:
        obj, _ = _json.JSONDecoder().raw_decode(s)
        return obj
    except _json.JSONDecodeError:
        return _json.loads(s)
'''

# Single-line import we append at the end of each patched parser file.
APPEND_IMPORT = (
    f"from vllm.tool_parsers.{HELPER_MODULE_NAME} import "
    f"_sage_encode, _sage_json_load  {MARKER}\n"
)


# ---------------------------------------------------------------------------
# Source-acquisition helpers
# ---------------------------------------------------------------------------

def _parses(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def _strip_sage_imports(text: str) -> str:
    """Remove any previously appended SAGE import lines so re-runs idempotently
    rebuild the file from a clean state."""
    return "\n".join(
        line for line in text.splitlines() if MARKER not in line
    ) + ("\n" if text.endswith("\n") else "")


def _download_from_github(rel_path: str) -> Optional[str]:
    url = f"{GITHUB_RAW_BASE}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            if resp.status != 200:
                return None
            data = resp.read().decode("utf-8")
        if not _parses(data):
            return None
        return data
    except Exception as exc:
        print(f"  warn: github fetch failed for {rel_path}: {exc}")
        return None


def _acquire_clean_source(path: Path, rel_path: str) -> Optional[str]:
    """Return verified-parseable *original* source for ``path``.

    Priority order — picked so re-runs always rebuild from the same pristine
    base, never from an already-patched file (which would lose the helper
    import on the next pass):

    1. Existing ``.sage_orig`` backup if it parses cleanly.
    2. Local file if it parses cleanly AND does not contain SAGE markers
       (i.e. this is the first ever run); save it as the backup.
    3. Download from GitHub at the pinned vLLM tag; save as backup.
    """
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)

    # 1. Trust the backup first. It's the only file we ever guarantee is the
    # untouched original, so subsequent runs always rebuild from here.
    if backup.exists():
        bak = backup.read_text(encoding="utf-8")
        if _parses(bak) and MARKER not in bak:
            return bak

    # 2. First run: take the local file IF it's clean and unmarked.
    if path.exists():
        local = path.read_text(encoding="utf-8")
        if MARKER not in local and _parses(local):
            backup.write_text(local, encoding="utf-8")
            return local

    # 3. Recover from GitHub.
    print(f"  recovering {path.name} from GitHub ({VLLM_TAG}) …")
    fresh = _download_from_github(rel_path)
    if fresh is None:
        print(f"  ERROR: could not obtain a clean source for {path.name}")
        return None
    backup.write_text(fresh, encoding="utf-8")
    return fresh


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------

def _apply_parser_edits(src: str) -> tuple[str, int, int]:
    """Apply call-site rewrites + append the helper import.

    Returns (new_src, encode_count, json_count).
    """
    encode_count = src.count("self.model_tokenizer.encode(")
    new_src = src.replace(
        "self.model_tokenizer.encode(",
        "_sage_encode(self.model_tokenizer,",
    )

    # json.loads(match[...] [if … else match[...]]) — only the hermes pattern.
    json_pattern = re.compile(
        r"\bjson\.loads\(\s*"
        r"(match\[(?:0|1)\](?:\s+if\s+match\[(?:0|1)\]\s+else\s+match\[(?:0|1)\])?)"
        r"\s*\)"
    )
    new_src, json_count = json_pattern.subn(
        lambda m: f"_sage_json_load({m.group(1)})", new_src
    )

    if encode_count + json_count == 0:
        return new_src, 0, 0

    if not new_src.endswith("\n"):
        new_src += "\n"
    new_src += APPEND_IMPORT
    return new_src, encode_count, json_count


def write_helper_module(parsers_dir: Path) -> None:
    target = parsers_dir / f"{HELPER_MODULE_NAME}.py"
    target.write_text(HELPER_MODULE_SRC, encoding="utf-8")
    print(f"  wrote helper module: {target.name}")


def patch_parser(path: Path, rel_path: str) -> None:
    src = _acquire_clean_source(path, rel_path)
    if src is None:
        return

    new_src, ec, jc = _apply_parser_edits(src)
    if ec + jc == 0:
        # Nothing to patch; still rewrite the file to the clean source so the
        # backup and live file are in sync.
        path.write_text(src, encoding="utf-8")
        print(f"  no-op (no patch sites): {path.name}")
        return

    path.write_text(new_src, encoding="utf-8")
    if not _parses(path.read_text(encoding="utf-8")):
        # Revert from backup on validation failure.
        backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  ERROR: validation failed, reverted: {path.name}")
        return
    print(f"  patched: {path.name} (encode={ec}, json={jc})")


# ---------------------------------------------------------------------------
# serving.py: regex-rewrite around tool_parser_cls(tokenizer)
# ---------------------------------------------------------------------------

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
        f"{indent}{SERVING_MARKER}\n"
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
    if SERVING_MARKER in src:
        print(f"  skip (already patched): {path.name}")
        return
    new_src, count = _SERVING_PATTERN.subn(_serving_replacement, src)
    if count == 0:
        print(f"  skip (pattern not found): {path.name}")
        return
    path.write_text(new_src, encoding="utf-8")
    if not _parses(path.read_text(encoding="utf-8")):
        path.write_text(src, encoding="utf-8")
        print(f"  ERROR: validation failed, reverted: {path.name}")
        return
    print(f"  patched: {path.name} ({count} site(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vllm_src_dir>")
        sys.exit(1)

    vllm_src = Path(sys.argv[1])
    if not vllm_src.exists():
        print(f"vLLM source not found at {vllm_src}; skipping patches.")
        sys.exit(0)

    print(f"Patching vLLM source at {vllm_src} …")

    parsers_dir = vllm_src / "vllm" / "tool_parsers"
    if parsers_dir.exists():
        write_helper_module(parsers_dir)
        for fname, rel in PARSER_FILES.items():
            p = parsers_dir / fname
            if not p.exists():
                # Files like llama3_json/internlm may not exist in this
                # vLLM build — silently skip.
                continue
            patch_parser(p, rel)
    else:
        print(f"  tool_parsers dir not found: {parsers_dir}")

    for rel in SERVING_CANDIDATES:
        p = vllm_src / rel
        if p.exists():
            patch_serving(p)

    print("Done.")


if __name__ == "__main__":
    main()
