#!/usr/bin/env bash
# Run TerminalBench baseline: standalone vLLM + TerminalBench's own agent.
#
# This uses TerminalBench's native tb CLI with a local vLLM endpoint,
# providing an apples-to-apples comparison against the PIE agent.
#
# Required env vars:
#   TBENCH_DIR   — path to terminal-bench repo clone
#   MODEL_REPO   — HuggingFace model repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TBENCH_DIR="${TBENCH_DIR:?Set TBENCH_DIR}"
MODEL_REPO="${MODEL_REPO:?Set MODEL_REPO}"

VLLM_PORT="${VLLM_PORT:-8899}"
VLLM_VENV="${VLLM_VENV:-/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TASK_LIST="${TASK_LIST:-easy}"
NUM_TASKS="${NUM_TASKS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/baseline}"

export HF_HOME="${HF_HOME:-/nfs/roberts/scratch/pi_ql324/ly337/hf_cache}"
export PYTHONPATH=""

TBENCH_VENV="$TBENCH_DIR/.venv"

mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

echo "=== TerminalBench Baseline ==="
echo "  Model:    $MODEL_REPO"
echo "  Tasks:    $TASK_LIST"
echo "  vLLM:     127.0.0.1:$VLLM_PORT"
echo ""

# --- 1. Start vLLM server ---
echo "[1/2] Starting vLLM server..."
VLLM_LOG="$SCRIPT_DIR/logs/vllm_baseline_$$.log"

"$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_REPO" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

echo "  Waiting for vLLM..."
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null | grep -q "$MODEL_REPO"; then
        echo "  Ready ($((i * 5))s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done

# --- 2. Run TerminalBench ---
echo "[2/2] Running TerminalBench (baseline)..."

# Collect task IDs
TASK_ARGS=()
if [[ "$TASK_LIST" == "easy" || "$TASK_LIST" == "medium" || "$TASK_LIST" == "hard" ]]; then
    while IFS= read -r task_name; do
        TASK_ARGS+=(--task-ids "$task_name")
    done < <(
        grep -l "difficulty: $TASK_LIST" "$TBENCH_DIR"/original-tasks/*/task.yaml 2>/dev/null | \
        xargs -I{} dirname {} | xargs -I{} basename {} | \
        sort | head -${NUM_TASKS:-999}
    )
fi

# Set up LLM config for TerminalBench
export OPENAI_API_KEY="none"
export OPENAI_API_BASE="http://127.0.0.1:$VLLM_PORT/v1"

cd "$TBENCH_DIR"

"$TBENCH_VENV/bin/tb" run \
    --agent naive \
    --model "openai/$MODEL_REPO" \
    --output "$OUTPUT_DIR" \
    "${TASK_ARGS[@]}"

echo ""
echo "Done. Results in $OUTPUT_DIR"
