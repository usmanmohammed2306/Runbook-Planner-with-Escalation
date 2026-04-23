#!/usr/bin/env bash
# ============================================================================
# setup_env.sh
#
# Prepares a self-contained venv for the Runbook Planner with Escalation (RPE)
# project. Clones the tau-bench and ACEBench upstream repositories into
# ./external/ and installs their Python runtimes alongside vLLM.
#
# Run ONCE before run_project.sh. Re-running is safe (idempotent).
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "[setup] $*"; }

# ---------------------------------------------------------------------------
# Scratch / cache exports
# ---------------------------------------------------------------------------
: "${PROJECT_SCRATCH:=${REPO_ROOT}/.scratch}"
export PROJECT_SCRATCH
export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_SCRATCH}/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_SCRATCH}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TMPDIR="${TMPDIR:-${PROJECT_SCRATCH}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_SCRATCH}/pip_cache}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TMPDIR" "$PIP_CACHE_DIR"

# ---------------------------------------------------------------------------
# Python + venv
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' not found on PATH." >&2
  exit 1
fi

log "Python: $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  log "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Reusing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# Install core requirements (vLLM pulls torch with a compatible CUDA wheel)
# ---------------------------------------------------------------------------
log "Installing core requirements"
pip install -r "$REPO_ROOT/requirements.txt"

# ---------------------------------------------------------------------------
# Clone + install tau-bench
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/external"
TAU_DIR="$REPO_ROOT/external/tau-bench"
if [[ ! -d "$TAU_DIR/.git" ]]; then
  log "Cloning tau-bench -> $TAU_DIR"
  git clone --depth 1 https://github.com/sierra-research/tau-bench.git "$TAU_DIR"
else
  log "tau-bench already present at $TAU_DIR"
fi
log "Installing tau-bench (editable)"
pip install -e "$TAU_DIR"

# ---------------------------------------------------------------------------
# Clone + install ACEBench
# ---------------------------------------------------------------------------
ACE_DIR="$REPO_ROOT/external/ACEBench"
if [[ ! -d "$ACE_DIR/.git" ]]; then
  log "Cloning ACEBench -> $ACE_DIR"
  # Primary URL is the one cited in the project PDF; mirror is a fallback.
  git clone --depth 1 https://github.com/chenchen0103/ACEBench.git "$ACE_DIR" \
    || git clone --depth 1 https://github.com/ACEBench/ACEBench.git "$ACE_DIR"
else
  log "ACEBench already present at $ACE_DIR"
fi

if [[ -f "$ACE_DIR/requirements.txt" ]]; then
  log "Installing ACEBench requirements"
  pip install -r "$ACE_DIR/requirements.txt" \
    || log "WARNING: ACEBench requirements.txt install had issues; continuing"
fi

# ---------------------------------------------------------------------------
# Sanity / version checks
# ---------------------------------------------------------------------------
log "Version + compatibility check"
python - <<'PY'
import importlib, sys

def try_ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"IMPORT_FAIL: {e}"

for name in ["torch", "transformers", "vllm", "openai", "litellm", "tau_bench"]:
    print(f"  {name}={try_ver(name)}")

try:
    import torch
    print("  cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        print("  capability:", f"{cap[0]}.{cap[1]}")
except Exception as e:
    print("  torch probe failed:", e, file=sys.stderr)
PY

log "Setup complete."
log "Next: bash run_project.sh"
