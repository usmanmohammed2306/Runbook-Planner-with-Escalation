#!/usr/bin/env bash
# ============================================================================
# setup_env.sh
#
# One-time environment setup for the Runbook Planner with Escalation (RPE)
# project. Targets an HPC node where the system Python is 3.6 and we have
# Environment Modules / Lmod available. We use `uv` with a managed Python 3.12
# so we don't depend on whichever Python happens to be on PATH.
#
# Build recipe (validated working combination):
#   - gcc-13.2.0-gcc-12.1.0
#   - cuda-13.0.1-gcc-13.2.0
#   - Python 3.12 (uv --managed-python)
#   - torch 2.10.0+cu130 / torchvision 0.25.0+cu130 / torchaudio 2.10.0+cu130
#     (installed from https://download.pytorch.org/whl/cu130)
#   - vLLM v0.18.0 built from source, editable, --no-build-isolation
#
# Critical: we unset LD_LIBRARY_PATH after module load so the PyTorch cu130
# wheel's bundled NCCL / CUDA shared libs aren't shadowed by cluster libraries.
#
# Re-run friendly:
#   - reuses venv if present (RESET_VENV=1 wipes it)
#   - repairs torch stack in place if the trio versions drift
#   - rebuilds vLLM only when FORCE_REBUILD=1, dist metadata is missing, the
#     installed vLLM isn't pointing at external/vllm, or the build signature
#     changed
#
# First clean run:   RESET_VENV=1 FORCE_REBUILD=1 bash setup_env.sh
# Normal re-run:     bash setup_env.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "[setup] $*"; }

# ---------------------------------------------------------------------------
# Pinned versions (edit here, not inline)
# ---------------------------------------------------------------------------
PY_VER="${PY_VER:-3.12}"

TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TORCH_WHL_INDEX="${TORCH_WHL_INDEX:-https://download.pytorch.org/whl/cu130}"

VLLM_VERSION="${VLLM_VERSION:-0.18.0}"

GCC_MODULE="${GCC_MODULE:-gcc-13.2.0-gcc-12.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda-13.0.1-gcc-13.2.0}"

MAX_JOBS="${MAX_JOBS:-6}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
RESET_VENV="${RESET_VENV:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
: "${PROJECT_SCRATCH:=${REPO_ROOT}/.scratch}"
export PROJECT_SCRATCH

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${REPO_ROOT}/external}"
TAU_DIR="${TAU_DIR:-${EXTERNAL_DIR}/tau-bench}"
ACE_DIR="${ACE_DIR:-${EXTERNAL_DIR}/ACEBench}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-${EXTERNAL_DIR}/vllm-src-${VLLM_VERSION}}"

export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_SCRATCH}/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_SCRATCH}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch/inductor}"
export TMPDIR="${TMPDIR:-${PROJECT_SCRATCH}/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROJECT_SCRATCH}/torch_extensions}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_SCRATCH}/pip_cache}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" \
  "$PIP_CACHE_DIR" "$EXTERNAL_DIR"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MAX_JOBS

# ---------------------------------------------------------------------------
# HF token (gated model downloads). Override by exporting HF_TOKEN.
# ---------------------------------------------------------------------------
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
require_cmd () {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

ensure_module_cmd () {
  if command -v module >/dev/null 2>&1; then return 0; fi
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
    if [[ -f "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"
      break
    fi
  done
  command -v module >/dev/null 2>&1
}

ensure_uv () {
  if command -v uv >/dev/null 2>&1; then return 0; fi
  log "Installing uv (astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || { echo "uv install failed" >&2; exit 1; }
}

have_req () {
  python - "$1" <<'PY' >/dev/null 2>&1
import sys, importlib.metadata as md
try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(1)
req = Requirement(sys.argv[1])
if req.marker is not None and not req.marker.evaluate():
    raise SystemExit(0)
try:
    v = md.version(req.name)
except md.PackageNotFoundError:
    raise SystemExit(1)
if req.specifier and v not in req.specifier:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_uv_pkg () {
  if have_req "$1"; then log "Already satisfied: $1"; else log "Installing: $1"; uv pip install "$1"; fi
}

req_file_satisfied () {
  python - "$1" <<'PY' >/dev/null 2>&1
import sys, pathlib, importlib.metadata as md
try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(1)
p = pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
for raw in p.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("-"): continue
    r = Requirement(line)
    if r.marker is not None and not r.marker.evaluate(): continue
    try: v = md.version(r.name)
    except md.PackageNotFoundError: raise SystemExit(1)
    if r.specifier and v not in r.specifier: raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_req_file () {
  if req_file_satisfied "$1"; then log "Requirement file already satisfied: $1"; else log "Installing requirement file: $1"; uv pip install -r "$1"; fi
}

make_filtered_requirements () {
  # Strip packages from an upstream requirements file so installing it can't
  # clobber our cu130 torch stack or our from-source vLLM build.
  #
  #   $1 = source requirements file
  #   $2 = destination filtered file
  #   $3 = space-separated exact package names to skip (normalized: lowercase, _→-)
  #   $4 = space-separated name-prefixes to skip (e.g. "nvidia-")
  #   $5 = space-separated line-prefixes to drop (e.g. "--extra-index-url -i")
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import pathlib, re, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
skip_names = set(sys.argv[3].split())
skip_name_prefixes = tuple(sys.argv[4].split())
skip_line_prefixes = tuple(sys.argv[5].split())
def nn(s):
    m = re.match(r'^\s*([A-Za-z0-9_.-]+)', s)
    return m.group(1).lower().replace("_","-") if m else ""
out = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        out.append(raw); continue
    if any(line.startswith(p) for p in skip_line_prefixes): continue
    n = nn(line)
    if n in skip_names: continue
    if any(n.startswith(p) for p in skip_name_prefixes): continue
    out.append(raw)
dst.write_text("\n".join(out) + "\n")
PY
}

# Skip-set used for vLLM build.txt: strip torch + nvidia-* + triton + --extra-index-url
# so the build resolver can't overwrite our cu130 stack.
filter_vllm_build_requirements () {
  make_filtered_requirements "$1" "$2" \
    "torch torchvision torchaudio triton" \
    "nvidia-" \
    "--extra-index-url -i"
}

# Skip-set used for ACEBench's requirements.txt: same torch-family block, plus
# vllm (we build from source at a different version), xformers (torch-version
# sensitive), and any nvidia-*-cu12 packages (we're on cu13).
filter_acebench_requirements () {
  make_filtered_requirements "$1" "$2" \
    "torch torchvision torchaudio triton vllm vllm-flash-attn xformers" \
    "nvidia-" \
    "--extra-index-url -i"
}

# CUDA 13 removed libnvToolsExt.so (nvtx is now a header-only library). The
# torch 2.10.0+cu130 wheel ships a Caffe2/public/cuda.cmake that still calls
# FATAL_ERROR when find_library(nvToolsExt) fails, which breaks any project
# building against torch. Downgrade that failure to a soft STATUS and leave
# CUDA_nvToolsExt_LIBRARY empty — the downstream target torch::nvtoolsext
# then becomes a no-op INTERFACE target, which is the right behavior on
# CUDA 13+.
patch_torch_nvtx_for_cuda13 () {
  local cuda_cmake
  cuda_cmake="$(find "$VENV_DIR/lib" -path '*/torch/share/cmake/Caffe2/public/cuda.cmake' 2>/dev/null | head -n1)"
  if [[ -z "$cuda_cmake" || ! -f "$cuda_cmake" ]]; then
    log "No torch cuda.cmake found; skipping nvtx CUDA-13 patch"
    return 0
  fi
  if ! grep -qE 'FATAL_ERROR[[:space:]]+"Failed to find nvToolsExt"' "$cuda_cmake"; then
    log "torch cuda.cmake: already patched or no nvToolsExt FATAL_ERROR"
    return 0
  fi
  log "Patching $cuda_cmake for CUDA 13 (libnvToolsExt.so removed upstream)"
  python - "$cuda_cmake" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
needle = 'if(NOT CUDA_nvToolsExt_LIBRARY)\n    message(FATAL_ERROR "Failed to find nvToolsExt")\nendif()'
replacement = (
    'if(NOT CUDA_nvToolsExt_LIBRARY)\n'
    '    message(STATUS "nvToolsExt not found; CUDA 13 removed libnvToolsExt.so. "\n'
    '                   "Using empty stub; torch::nvtoolsext becomes a no-op target.")\n'
    '    set(CUDA_nvToolsExt_LIBRARY "")\n'
    'endif()'
)
if needle in s:
    p.write_text(s.replace(needle, replacement))
    print("patched (exact)")
    sys.exit(0)
s2 = re.sub(r'message\(FATAL_ERROR\s+"Failed to find nvToolsExt"\s*\)',
            'message(STATUS "nvToolsExt missing on CUDA 13+; continuing")', s)
if s2 != s:
    p.write_text(s2)
    print("patched (regex)")
else:
    print("no-op (pattern not found)")
PY
}

pytorch_stack_ok () {
  python - "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCHAUDIO_VERSION" <<'PY' >/dev/null 2>&1
import sys, torch, torchvision, torchaudio
from torchvision.ops import nms  # noqa: F401
assert torch.__version__.startswith(sys.argv[1]), torch.__version__
assert torchvision.__version__.startswith(sys.argv[2]), torchvision.__version__
assert torchaudio.__version__.startswith(sys.argv[3]), torchaudio.__version__
PY
}

repair_pytorch_stack () {
  log "Repairing PyTorch stack from $TORCH_WHL_INDEX"
  uv pip install --reinstall --index-url "$TORCH_WHL_INDEX" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
}

ensure_pytorch_stack () {
  log "Ensuring PyTorch ${TORCH_VERSION}+cu130 / torchvision ${TORCHVISION_VERSION} / torchaudio ${TORCHAUDIO_VERSION}"
  if pytorch_stack_ok; then
    log "PyTorch stack is already healthy"
  else
    repair_pytorch_stack
  fi
  python - <<'PY'
import torch, torchvision, torchaudio
from torchvision.ops import nms  # noqa: F401
print(f"  torch={torch.__version__} torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}")
print(f"  torch.version.cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
}

vllm_distribution_present () {
  python -c 'import importlib.metadata as md; md.version("vllm")' >/dev/null 2>&1
}

vllm_package_path_under_src () {
  python - "$1" <<'PY' >/dev/null 2>&1
import importlib.util, os, sys
src = os.path.realpath(sys.argv[1])
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin: raise SystemExit(1)
raise SystemExit(0 if os.path.realpath(spec.origin).startswith(src) else 1)
PY
}

build_signature () {
  python - "$1" "$2" <<'PY'
import json, os, subprocess, sys, torch
src_dir = os.path.realpath(sys.argv[1])
vllm_version = sys.argv[2]
def run(c):
    try: return subprocess.check_output(c, text=True).strip()
    except Exception: return ""
def git_head(p):
    if not os.path.isdir(os.path.join(p, ".git")): return ""
    return run(["git","-C",p,"rev-parse","HEAD"])
device, cap = "", ""
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
nvcc = run(["nvcc","--version"])
sig = {
    "python": sys.version.split()[0], "torch": torch.__version__,
    "torchvision": run([sys.executable,"-c","import torchvision;print(torchvision.__version__)"]),
    "torchaudio": run([sys.executable,"-c","import torchaudio;print(torchaudio.__version__)"]),
    "torch_cuda": str(torch.version.cuda), "cuda_available": bool(torch.cuda.is_available()),
    "device_name": device, "capability": cap,
    "gcc": run(["gcc","-dumpfullversion","-dumpversion"]),
    "gxx": run(["g++","-dumpfullversion","-dumpversion"]),
    "nvcc": nvcc.splitlines()[-1] if nvcc else "",
    "cc": os.environ.get("CC",""), "cxx": os.environ.get("CXX",""),
    "cuda_home": os.environ.get("CUDA_HOME",""),
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST",""),
    "source_git_head": git_head(src_dir),
    "requested_vllm_version": vllm_version,
}
print(json.dumps(sig, sort_keys=True))
PY
}

detect_torch_cuda_arch () {
  python - <<'PY'
import sys, torch
if not torch.cuda.is_available(): sys.exit(2)
M,m = torch.cuda.get_device_capability(0)
print(f"{M}.{m}")
PY
}

# ---------------------------------------------------------------------------
# Pre-module tool checks
# ---------------------------------------------------------------------------
require_cmd git
require_cmd curl
ensure_uv

# ---------------------------------------------------------------------------
# Load toolchain (no-op off-cluster)
# ---------------------------------------------------------------------------
if ensure_module_cmd; then
  log "Loading cluster modules: $GCC_MODULE + $CUDA_MODULE"
  module purge || true
  module load "$GCC_MODULE" || log "WARNING: module load $GCC_MODULE failed"
  module load "$CUDA_MODULE" || log "WARNING: module load $CUDA_MODULE failed"
else
  log "No 'module' command available; using system toolchain"
fi

unset PYTHONPATH
unset TRANSFORMERS_CACHE
unset VLLM_CACHE_DIR
# Critical: prevent the cluster's NCCL/CUDA shared libs from shadowing the
# cu130 wheels that torch ships.
unset LD_LIBRARY_PATH
hash -r

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  log "CUDA_HOME=$CUDA_HOME"
else
  log "WARNING: nvcc not found; vLLM build will fail without CUDA toolkit"
fi

if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
  export CC="$(which gcc)"
  export CXX="$(which g++)"
  export CUDAHOSTCXX="$CXX"
  export CMAKE_C_COMPILER="$CC"
  export CMAKE_CXX_COMPILER="$CXX"
  export CMAKE_CUDA_HOST_COMPILER="$CXX"
  export CMAKE_GENERATOR="Ninja"
fi

log "Toolchain check:"
command -v gcc  >/dev/null 2>&1 && { echo -n "  "; gcc --version | head -n1; }
command -v g++  >/dev/null 2>&1 && { echo -n "  "; g++ --version | head -n1; }
command -v nvcc >/dev/null 2>&1 && { echo -n "  "; nvcc --version | tail -n1; }

# ---------------------------------------------------------------------------
# Create / reuse venv (Python 3.12 via uv managed-python)
# ---------------------------------------------------------------------------
if [[ "$RESET_VENV" == "1" && -d "$VENV_DIR" ]]; then
  log "RESET_VENV=1 -> removing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  log "Creating uv venv at $VENV_DIR (Python $PY_VER)"
  uv venv "$VENV_DIR" --python "$PY_VER" --seed --managed-python
else
  log "Reusing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd /tmp  # so uv doesn't treat $REPO_ROOT as a project by accident

log "Python: $(python --version) at $(command -v python)"

# ---------------------------------------------------------------------------
# Base build tooling
# ---------------------------------------------------------------------------
uv self update || true
ensure_uv_pkg "packaging>=24.2"
ensure_uv_pkg "wheel"
ensure_uv_pkg "setuptools>=77.0.3,<81.0.0"
ensure_uv_pkg "setuptools-scm>=8.0"
ensure_uv_pkg "numpy"
ensure_uv_pkg "ninja"
ensure_uv_pkg "cmake"
ensure_uv_pkg "jinja2"
ensure_uv_pkg "gitpython"

# ---------------------------------------------------------------------------
# PyTorch cu130 trio (must be pinned BEFORE any torch-aware libs so they
# pick up the already-installed torch instead of pulling another variant).
# ---------------------------------------------------------------------------
ensure_pytorch_stack

# ---------------------------------------------------------------------------
# Project requirements (transformers / accelerate / openai / litellm / etc.)
# requirements.txt is torch-free by design.
# ---------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  log "Installing repo requirements.txt"
  uv pip install -r "$REPO_ROOT/requirements.txt"
fi

# Re-check after transformers/accelerate — they don't pin torch, but just in case.
ensure_pytorch_stack

# Detect current GPU arch for vLLM build
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  if DETECTED_ARCH="$(detect_torch_cuda_arch 2>/dev/null)"; then
    export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCH"
    log "Detected TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  fi
else
  export TORCH_CUDA_ARCH_LIST
  log "Using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

# ---------------------------------------------------------------------------
# Clone tau-bench and ACEBench
# ---------------------------------------------------------------------------
if [[ ! -d "$TAU_DIR/.git" ]]; then
  log "Cloning tau-bench -> $TAU_DIR"
  git clone --depth 1 https://github.com/sierra-research/tau-bench.git "$TAU_DIR"
else
  log "tau-bench already at $TAU_DIR"
fi

if [[ -f "$TAU_DIR/pyproject.toml" || -f "$TAU_DIR/setup.py" ]]; then
  log "Editable install of tau-bench"
  (cd "$TAU_DIR" && uv pip install -e .) || log "WARNING: editable tau-bench install failed"
fi
[[ -f "$TAU_DIR/requirements.txt"     ]] && ensure_req_file "$TAU_DIR/requirements.txt"
[[ -f "$TAU_DIR/requirements-dev.txt" ]] && ensure_req_file "$TAU_DIR/requirements-dev.txt"
ensure_uv_pkg "litellm"

if [[ ! -d "$ACE_DIR/.git" ]]; then
  log "Cloning ACEBench -> $ACE_DIR"
  git clone --depth 1 https://github.com/chenchen0103/ACEBench.git "$ACE_DIR" \
    || git clone --depth 1 https://github.com/ACEBench/ACEBench.git "$ACE_DIR" \
    || log "WARNING: ACEBench clone failed (both primary and mirror)"
else
  log "ACEBench already at $ACE_DIR"
fi
if [[ -f "$ACE_DIR/requirements.txt" ]]; then
  # ACEBench's requirements.txt pins torch==2.4.0 / torchvision==0.19.0 /
  # triton==3.0.0 / vllm==0.6.1.post1 / xformers and a full nvidia-*-cu12
  # stack. Installing it as-is clobbers our cu130 torch build. Filter to
  # drop those before install, then re-check the torch stack afterwards.
  FILTERED_ACE_REQ="$TMPDIR/acebench-requirements.txt"
  filter_acebench_requirements "$ACE_DIR/requirements.txt" "$FILTERED_ACE_REQ"
  log "Installing ACEBench requirements.txt (filtered: torch/vllm/nvidia-* stripped)"
  uv pip install -r "$FILTERED_ACE_REQ" || log "WARNING: ACEBench filtered requirements install had issues"
  # Defensive: ensure_pytorch_stack re-pins torch to our cu130 trio if a
  # transitive dep pulled in a different variant anyway.
  ensure_pytorch_stack
fi

# ---------------------------------------------------------------------------
# vLLM source + build-if-needed
# ---------------------------------------------------------------------------
if [[ ! -d "$VLLM_SRC_DIR/.git" ]]; then
  log "Cloning vLLM v$VLLM_VERSION -> $VLLM_SRC_DIR"
  git clone --branch "v${VLLM_VERSION}" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
fi

VLLM_BUILD_SIG_FILE="$VENV_DIR/.vllm-build-signature"
REBUILD_REASON=""
need_rebuild=0
if [[ "$FORCE_REBUILD" == "1" ]]; then
  REBUILD_REASON="FORCE_REBUILD=1"; need_rebuild=1
elif ! vllm_distribution_present; then
  REBUILD_REASON="vLLM distribution metadata is missing"; need_rebuild=1
elif ! vllm_package_path_under_src "$VLLM_SRC_DIR"; then
  REBUILD_REASON="Installed vLLM is not from $VLLM_SRC_DIR"; need_rebuild=1
elif [[ ! -f "$VLLM_BUILD_SIG_FILE" ]]; then
  REBUILD_REASON="Missing build signature file"; need_rebuild=1
else
  cur_sig="$(build_signature "$VLLM_SRC_DIR" "$VLLM_VERSION" || true)"
  saved_sig="$(cat "$VLLM_BUILD_SIG_FILE" 2>/dev/null || true)"
  if [[ "$cur_sig" != "$saved_sig" ]]; then
    REBUILD_REASON="Build signature changed"
    printf '%s\n' "$saved_sig" > "${VLLM_BUILD_SIG_FILE}.saved"
    printf '%s\n' "$cur_sig"   > "${VLLM_BUILD_SIG_FILE}.current"
    need_rebuild=1
  fi
fi

if (( need_rebuild )); then
  log "Rebuild required: $REBUILD_REASON"
  python -m pip uninstall -y vllm >/dev/null 2>&1 || true
  git -C "$VLLM_SRC_DIR" fetch --tags origin
  git -C "$VLLM_SRC_DIR" checkout -f "v${VLLM_VERSION}"
  git -C "$VLLM_SRC_DIR" reset --hard "v${VLLM_VERSION}"
  git -C "$VLLM_SRC_DIR" clean -xfd >/dev/null 2>&1 || true
  rm -rf "$TORCH_EXTENSIONS_DIR"/* 2>/dev/null || true

  FILTERED_BUILD_REQ="$TMPDIR/vllm-build-requirements.txt"
  filter_vllm_build_requirements "$VLLM_SRC_DIR/requirements/build.txt" "$FILTERED_BUILD_REQ"
  ensure_req_file "$FILTERED_BUILD_REQ"

  # torch 2.10.0+cu130's Caffe2/public/cuda.cmake still FATAL_ERRORs when it
  # can't find libnvToolsExt.so, but CUDA 13 removed that library (nvtx is
  # now header-only). Patch it in place before the build so cmake treats
  # nvtx as a no-op target instead of aborting.
  patch_torch_nvtx_for_cuda13

  log "Building vLLM (editable, --no-build-isolation) with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset}"
  (cd "$VLLM_SRC_DIR" && uv pip install --no-build-isolation -e .)

  # uv can sometimes pull in a torch variant via vLLM's metadata — re-pin.
  ensure_pytorch_stack

  build_signature "$VLLM_SRC_DIR" "$VLLM_VERSION" > "$VLLM_BUILD_SIG_FILE"
else
  log "Skipping vLLM rebuild: existing build matches"
fi

# ---------------------------------------------------------------------------
# Persist HF token for huggingface_hub / vLLM
# ---------------------------------------------------------------------------
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token" || true

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
python - "$VLLM_SRC_DIR" <<'PY'
import importlib.metadata as md, os, shutil, sys
import torch, torchvision, torchaudio
from torchvision.ops import nms  # noqa: F401
import vllm
src = os.path.realpath(sys.argv[1])
print(f"  python={sys.version.split()[0]}")
print(f"  torch={torch.__version__} torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}")
print(f"  torch.version.cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"  vllm dist={md.version('vllm')} file={os.path.realpath(vllm.__file__)}")
print(f"  vllm exe={shutil.which('vllm')}")
PY

log "Setup complete."
log "Next: bash run_project.sh"
