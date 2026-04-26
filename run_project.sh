#!/usr/bin/env bash
# ============================================================================
# run_project.sh
#
# One-shot driver for the RPE experiment. Loads the same cluster toolchain
# that setup_env.sh used, activates the .venv (Python 3.12 + cu130 torch
# stack + vLLM 0.18.0 from source), serves a single local vLLM instance, and
# runs the six evaluations (baseline + RPE over tau-retail, tau-airline,
# ACEBench Agent). Produces outputs/summary/{summary.json,summary.md}.
#
# The venv created by setup_env.sh must already exist.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "[run] $*"; }

# ---------------------------------------------------------------------------
# Paths (must match setup_env.sh)
# ---------------------------------------------------------------------------
: "${PROJECT_SCRATCH:=${REPO_ROOT}/.scratch}"
export PROJECT_SCRATCH

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${REPO_ROOT}/external}"
TAU_DIR="${TAU_DIR:-${EXTERNAL_DIR}/tau-bench}"
ACE_DIR="${ACE_DIR:-${EXTERNAL_DIR}/ACEBench}"

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

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# HF token (gated model downloads). Override by exporting HF_TOKEN.
# ---------------------------------------------------------------------------
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"

# ---------------------------------------------------------------------------
# Cluster modules (same pinning as setup_env.sh)
# ---------------------------------------------------------------------------
GCC_MODULE="${GCC_MODULE:-gcc-13.2.0-gcc-12.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda-13.0.1-gcc-13.2.0}"

ensure_module_cmd () {
  if command -v module >/dev/null 2>&1; then return 0; fi
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
    if [[ -f "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"; break
    fi
  done
  command -v module >/dev/null 2>&1
}

if ensure_module_cmd; then
  log "Loading cluster modules: $GCC_MODULE + $CUDA_MODULE"
  module purge || true
  module load "$GCC_MODULE" || log "WARNING: module load $GCC_MODULE failed"
  module load "$CUDA_MODULE" || log "WARNING: module load $CUDA_MODULE failed"
fi

unset PYTHONPATH
unset TRANSFORMERS_CACHE
unset VLLM_CACHE_DIR
# Don't leak cluster NCCL/CUDA libs onto the cu130 wheels.
unset LD_LIBRARY_PATH
hash -r

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log "Python: $(python --version) at $(command -v python)"
log "vllm:   $(command -v vllm || echo not-found)"

# ---------------------------------------------------------------------------
# Runtime configuration (overridable)
# ---------------------------------------------------------------------------
SERVED_NAME="${SERVED_NAME:-qwen-agent}"
PORT="${PORT:-8001}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-bfloat16}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
export VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_READY_TIMEOUT"

# Matched-compile-off config from the known-good baseline run.
VLLM_COMPILATION_CONFIG='{"mode":0,"custom_ops":["none"],"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false}}'

# Primary + fallback (max_model_len, gpu_memory_utilization, model-impl, eager).
# Rung 1 (eager, primary context):  the known-good config. Default.
# Rung 2 (eager, smaller context):  for OOM at primary context.
# Rung 3 (transformers backend):    last-resort path with no CUDA-graph deps.
#
# Why not try CUDA graphs first?
#   We pin compilation_config.mode=0 (NO_COMPILATION) for stability. vLLM 0.18
#   then auto-overrides cudagraph_mode -> NONE anyway, so dropping
#   --enforce-eager gains zero speed but routes through a different warmup
#   path (_dummy_run) that hits cudaErrorNoKernelImageForDevice in FlashAttn
#   on cu130 builds. Net: eager is strictly the safe and equivalent choice.
#   Set ENFORCE_EAGER=0 to opt back into the experimental fast rung.
PRIMARY_MAX_LEN="${MAX_MODEL_LEN:-16384}"
PRIMARY_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
PRIMARY_IMPL="${MODEL_IMPL:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
FALLBACK1_IMPL="auto";         FALLBACK1_MAX_LEN="12288"; FALLBACK1_MEM_UTIL="0.75"
FALLBACK2_IMPL="transformers"; FALLBACK2_MAX_LEN="8192";  FALLBACK2_MEM_UTIL="0.65"

# Truncation patch is sized for the 16 K context. The hard budget triggers
# the patch's iterative shrink: if a normal pass leaves the prompt over
# HARD_BUDGET chars, the patch keeps shrinking until it fits.
export TAU_PATCH_MAX_TOOL_CHARS="${TAU_PATCH_MAX_TOOL_CHARS:-2000}"
export TAU_PATCH_SOFT_BUDGET="${TAU_PATCH_SOFT_BUDGET:-28000}"
export TAU_PATCH_HARD_BUDGET="${TAU_PATCH_HARD_BUDGET:-32000}"

# Qwen2.5-7B first — stable 32K context, best tool-calling quality at this size.
MODEL_CANDIDATES=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-4B-Instruct-2507-FP8"
  "Qwen/Qwen3-4B-Instruct-2507"
)

TAU_TASK_SPLIT="${TAU_TASK_SPLIT:-test}"
TAU_START_INDEX="${TAU_START_INDEX:-0}"
TAU_END_INDEX="${TAU_END_INDEX:-20}"
TAU_NUM_TRIALS="${TAU_NUM_TRIALS:-1}"
TAU_MAX_CONCURRENCY="${TAU_MAX_CONCURRENCY:-1}"
TAU_TEMPERATURE="${TAU_TEMPERATURE:-0.0}"
TAU_MAX_STEPS="${TAU_MAX_STEPS:-30}"

ACE_LIMIT="${ACE_LIMIT:-30}"
ACE_MAX_STEPS="${ACE_MAX_STEPS:-20}"
ACE_LANGUAGE="${ACE_LANGUAGE:-en}"

OUTPUTS_DIR="${OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "$OUTPUTS_DIR/summary"

for cmd in curl lsof vllm python; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: missing command: $cmd" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------
VLLM_PID=""
ACTIVE_MODEL=""
ACTIVE_IMPL=""
ACTIVE_MAX_LEN=""

wait_health () {
  local url="$1" timeout="$2" slept=0 step=3
  while (( slept < timeout )); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep "$step"; slept=$(( slept + step ))
  done
  return 1
}

kill_port () {
  local pids; pids="$(lsof -ti tcp:"$1" 2>/dev/null || true)"
  [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
}

kill_vllm () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  kill_port "$PORT"
  VLLM_PID=""
}
trap 'kill_vllm' EXIT

start_vllm_once () {
  local model="$1" impl="$2" max_len="$3" mem_util="$4" eager="$5" label="$6"
  local log_file="$OUTPUTS_DIR/vllm.log"
  kill_vllm
  rm -f "$log_file"
  # `set -u` makes empty arrays brittle in older bash. Use a plain string flag.
  local eager_flag=""
  if [[ "$eager" == "1" ]]; then
    eager_flag="--enforce-eager"
  fi
  log "Starting vLLM [$label] model=$model impl=$impl max_len=$max_len mem_util=$mem_util eager=$eager"
  # NOTE: $eager_flag is intentionally unquoted so the empty case expands to
  #       nothing (vllm sees no extra arg). When set, it's a single flag with
  #       no spaces, so word-splitting is safe.
  CUDA_VISIBLE_DEVICES="$GPU" \
    vllm serve "$model" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --model-impl "$impl" \
      --dtype "$DTYPE" \
      --max-model-len "$max_len" \
      --gpu-memory-utilization "$mem_util" \
      --enable-auto-tool-choice \
      --tool-call-parser "$TOOL_CALL_PARSER" \
      --disable-log-stats \
      $eager_flag \
      --compilation-config "$VLLM_COMPILATION_CONFIG" \
      > "$log_file" 2>&1 &
  VLLM_PID=$!

  if wait_health "http://127.0.0.1:${PORT}/health" "$VLLM_READY_TIMEOUT"; then
    log "vLLM healthy [$label] with $model"
    ACTIVE_MODEL="$model"; ACTIVE_IMPL="$impl"; ACTIVE_MAX_LEN="$max_len"
    printf '%s\n' "$model" > "$OUTPUTS_DIR/active_model.txt"
    return 0
  fi
  log "vLLM failed [$label]; tail of log:"
  tail -n 80 "$log_file" >&2 || true
  kill_vllm
  return 1
}

# ---------------------------------------------------------------------------
# Try (model candidate × launch config) combos until one serves.
# Per-model rungs:
#   1. fast:   CUDA graphs ON, primary context  (skipped if ENFORCE_EAGER=1)
#   2. safe:   eager,         primary context  (previous known-good config)
#   3. small:  eager,         12K context
#   4. cpu-ish: transformers,  8K  context
# ---------------------------------------------------------------------------
STARTED=0
for m in "${MODEL_CANDIDATES[@]}"; do
  if [[ "$ENFORCE_EAGER" != "1" ]]; then
    if start_vllm_once "$m" "$PRIMARY_IMPL"   "$PRIMARY_MAX_LEN"   "$PRIMARY_MEM_UTIL"   "0" "fast/$m";    then STARTED=1; break; fi
  fi
  if start_vllm_once   "$m" "$PRIMARY_IMPL"   "$PRIMARY_MAX_LEN"   "$PRIMARY_MEM_UTIL"   "1" "safe/$m";    then STARTED=1; break; fi
  if start_vllm_once   "$m" "$FALLBACK1_IMPL" "$FALLBACK1_MAX_LEN" "$FALLBACK1_MEM_UTIL" "1" "small/$m";   then STARTED=1; break; fi
  if start_vllm_once   "$m" "$FALLBACK2_IMPL" "$FALLBACK2_MAX_LEN" "$FALLBACK2_MEM_UTIL" "1" "transformers/$m"; then STARTED=1; break; fi
  log "All configs failed for $m; trying next candidate"
done

if [[ "$STARTED" != "1" ]]; then
  log "ERROR: no model candidate could be served"
  exit 1
fi

log "Using $ACTIVE_MODEL (impl=$ACTIVE_IMPL max_len=$ACTIVE_MAX_LEN) for BOTH baseline and RPE"

# ---------------------------------------------------------------------------
# OpenAI-compatible client config
# ---------------------------------------------------------------------------
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
run_tau () {
  local env_name="$1" agent_kind="$2"
  local out="$OUTPUTS_DIR/tau_${env_name}_${agent_kind}"
  mkdir -p "$out"
  log "tau-bench: env=$env_name agent=$agent_kind -> $out"
  if python -m src.runners.tau_runner \
      --env "$env_name" --agent "$agent_kind" \
      --model "$SERVED_NAME" --user-model "$SERVED_NAME" \
      --task-split "$TAU_TASK_SPLIT" \
      --start-index "$TAU_START_INDEX" --end-index "$TAU_END_INDEX" \
      --num-trials "$TAU_NUM_TRIALS" \
      --max-concurrency "$TAU_MAX_CONCURRENCY" \
      --temperature "$TAU_TEMPERATURE" \
      --max-num-steps "$TAU_MAX_STEPS" \
      --output-dir "$out"; then
    log "tau-bench OK: $env_name/$agent_kind"
  else
    log "WARNING: tau-bench FAILED: $env_name/$agent_kind (continuing)"
  fi
}

run_ace () {
  local agent_kind="$1"
  local out="$OUTPUTS_DIR/acebench_agent_${agent_kind}"
  mkdir -p "$out"
  log "ACEBench: agent=$agent_kind -> $out"
  if python -m src.runners.ace_runner \
      --agent "$agent_kind" --model "$SERVED_NAME" \
      --language "$ACE_LANGUAGE" \
      --limit "$ACE_LIMIT" --max-num-steps "$ACE_MAX_STEPS" \
      --output-dir "$out"; then
    log "ACEBench OK: $agent_kind"
  else
    log "WARNING: ACEBench FAILED: $agent_kind (continuing)"
  fi
}

run_tau retail  baseline
run_tau retail  rpe
run_tau retail  igrpe
run_tau airline baseline
run_tau airline rpe
run_tau airline igrpe
run_ace baseline
run_ace rpe
run_ace igrpe

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Stopping vLLM before summary"
kill_vllm

log "Building summary"
python -m src.summary.build_summary \
  --outputs-dir "$OUTPUTS_DIR" \
  --active-model "$ACTIVE_MODEL" \
  --served-name "$SERVED_NAME"

log "Done."
log "  summary.json: $OUTPUTS_DIR/summary/summary.json"
log "  summary.md:   $OUTPUTS_DIR/summary/summary.md"
