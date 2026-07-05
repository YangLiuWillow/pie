#!/usr/bin/env bash
# Phase 1 SWE-Bench run — LiteLLM baseline (direct vLLM serve, no Pie)
# Usage: bash run_litellm_baseline.sh [--subset-size N] [--instance-id ID]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=$SCRIPT_DIR/.venv
PIE_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
LOG_DIR=$SCRIPT_DIR/logs
PRED_DIR=$SCRIPT_DIR/predictions
VLLM_PORT=18000

MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

mkdir -p "$LOG_DIR" "$PRED_DIR"

EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--subset-size 50)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$PRED_DIR/litellm_qwen25_coder_7b_${TIMESTAMP}.jsonl"
VLLM_LOG="$LOG_DIR/vllm_serve_${TIMESTAMP}.log"

echo "=== Phase 1 SWE-Bench: LiteLLM baseline ==="
echo "  Model:    $MODEL"
echo "  Output:   $OUTPUT"
echo "  vLLM log: $VLLM_LOG"
echo "  Args:     ${EXTRA_ARGS[*]}"

# Boot vllm serve
echo ""
echo "[1/2] Starting vllm serve on port $VLLM_PORT..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE_VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "echo 'Stopping vllm (PID $VLLM_PID)'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

echo "  Waiting for vllm to be ready (model load ~60s)..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/health" 2>/dev/null | grep -q "{}"; then
        echo "  vllm is up (${i}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vllm died. Check $VLLM_LOG"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    sleep 2
done

# Run the benchmark
echo ""
echo "[2/2] Running SWE-Bench harness (LiteLLM baseline)..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python -m benchmarks.run_swe_bench \
    --backend litellm \
    --model "openai/$MODEL" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --api-key dummy \
    --output "$OUTPUT" \
    --label "litellm+qwen2.5-coder-7b" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Predictions: $OUTPUT"
