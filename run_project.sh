#!/usr/bin/env bash
# ============================================================================
# run_project.sh
#
# One-shot driver for the Runbook Planner with Escalation (RPE) experiment.
# Starts a single local vLLM OpenAI-compatible server with a fixed base model
# (same model for baseline AND RPE) and runs, in order:
#   1. tau-bench retail  baseline
#   2. tau-bench retail  rpe
#   3. tau-bench airline baseline
#   4. tau-bench airline rpe
#   5. ACEBench Agent    baseline
#   6. ACEBench Agent    rpe
# Then emits outputs/summary/summary.json and outputs/summary/summary.md.
#
# The venv created by setup_env.sh must already exist.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "[run] $*"; }

# ---------------------------------------------------------------------------
# Scratch / cache exports (mirror setup_env.sh)
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
mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Make `python -m src.*` work without an install
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ---------------------------------------------------------------------------
SERVED_NAME="${SERVED_NAME:-qwen-agent}"
PORT="${PORT:-8001}"
GPU="${GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
DTYPE="${DTYPE:-auto}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

MODEL_CANDIDATES=(
  "Qwen/Qwen3-4B-Instruct-2507-FP8"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen2.5-7B-Instruct"
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

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
for cmd in curl lsof vllm python; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $cmd" >&2
    exit 1
  }
done

# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------
VLLM_PID=""
ACTIVE_MODEL=""

wait_health () {
  local url="$1" timeout="$2" slept=0 step=3
  while (( slept < timeout )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$step"
    slept=$(( slept + step ))
  done
  return 1
}

kill_port () {
  local port="$1" pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
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

start_vllm () {
  local model="$1"
  local log_file="$OUTPUTS_DIR/vllm.log"
  kill_vllm
  rm -f "$log_file"
  log "Starting vLLM (model=$model, served_name=$SERVED_NAME, port=$PORT)"
  CUDA_VISIBLE_DEVICES="$GPU" \
    vllm serve "$model" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --enable-auto-tool-choice \
      --tool-call-parser "$TOOL_CALL_PARSER" \
      --disable-log-stats \
      > "$log_file" 2>&1 &
  VLLM_PID=$!

  if wait_health "http://127.0.0.1:${PORT}/health" "$VLLM_READY_TIMEOUT"; then
    log "vLLM is healthy with $model"
    ACTIVE_MODEL="$model"
    printf '%s\n' "$model" > "$OUTPUTS_DIR/active_model.txt"
    return 0
  fi

  log "vLLM did not become healthy with $model; last log lines:"
  tail -n 60 "$log_file" >&2 || true
  kill_vllm
  return 1
}

# ---------------------------------------------------------------------------
# Try model candidates in priority order (same model for baseline + RPE)
# ---------------------------------------------------------------------------
STARTED=0
for m in "${MODEL_CANDIDATES[@]}"; do
  if start_vllm "$m"; then
    STARTED=1
    break
  fi
  log "Falling back from $m to next candidate"
done
if [[ "$STARTED" != "1" ]]; then
  log "ERROR: no model candidate could be served"
  exit 1
fi

log "Same base model will be used for ALL baseline and RPE runs: $ACTIVE_MODEL"

# ---------------------------------------------------------------------------
# OpenAI-compatible client config
# ---------------------------------------------------------------------------
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------
run_tau () {
  local env_name="$1" agent_kind="$2"
  local out="$OUTPUTS_DIR/tau_${env_name}_${agent_kind}"
  mkdir -p "$out"
  log "tau-bench: env=$env_name agent=$agent_kind -> $out"
  if python -m src.runners.tau_runner \
      --env "$env_name" \
      --agent "$agent_kind" \
      --model "$SERVED_NAME" \
      --user-model "$SERVED_NAME" \
      --task-split "$TAU_TASK_SPLIT" \
      --start-index "$TAU_START_INDEX" \
      --end-index "$TAU_END_INDEX" \
      --num-trials "$TAU_NUM_TRIALS" \
      --max-concurrency "$TAU_MAX_CONCURRENCY" \
      --temperature "$TAU_TEMPERATURE" \
      --max-num-steps "$TAU_MAX_STEPS" \
      --output-dir "$out"; then
    log "tau-bench run completed: $env_name/$agent_kind"
  else
    log "WARNING: tau-bench run failed: $env_name/$agent_kind (continuing)"
  fi
}

run_ace () {
  local agent_kind="$1"
  local out="$OUTPUTS_DIR/acebench_agent_${agent_kind}"
  mkdir -p "$out"
  log "ACEBench Agent: agent=$agent_kind -> $out"
  if python -m src.runners.ace_runner \
      --agent "$agent_kind" \
      --model "$SERVED_NAME" \
      --language "$ACE_LANGUAGE" \
      --limit "$ACE_LIMIT" \
      --max-num-steps "$ACE_MAX_STEPS" \
      --output-dir "$out"; then
    log "ACEBench run completed: agent=$agent_kind"
  else
    log "WARNING: ACEBench run failed: agent=$agent_kind (continuing)"
  fi
}

# ---------------------------------------------------------------------------
# Required 6 runs
# ---------------------------------------------------------------------------
run_tau retail  baseline
run_tau retail  rpe
run_tau airline baseline
run_tau airline rpe
run_ace baseline
run_ace rpe

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Stopping vLLM before building summary"
kill_vllm

log "Building summary"
python -m src.summary.build_summary \
  --outputs-dir "$OUTPUTS_DIR" \
  --active-model "$ACTIVE_MODEL" \
  --served-name "$SERVED_NAME"

log "Done."
log "  summary.json: $OUTPUTS_DIR/summary/summary.json"
log "  summary.md:   $OUTPUTS_DIR/summary/summary.md"
