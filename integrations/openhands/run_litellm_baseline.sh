#!/usr/bin/env bash
# Phase 1 baseline — LiteLLM + direct vLLM serve (no Pie)
# Usage: bash run_litellm_baseline.sh [--subset-size N] [--instance-id ID]
# Set HARNESS=benchmarks.run_humanevalfix for HumanEvalFix instead of SWE-Bench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=$SCRIPT_DIR/.venv
PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
LOG_DIR=$SCRIPT_DIR/logs
PRED_DIR=$SCRIPT_DIR/predictions
VLLM_PORT=18000
HARNESS=${HARNESS:-benchmarks.run_swe_bench}

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
LABEL=${LABEL:-litellm+$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# Auto-select tool-call parser based on model name
if [ -z "${TOOL_CALL_PARSER:-}" ]; then
    case "$MODEL" in
        *Qwen3-Coder*) TOOL_CALL_PARSER=qwen3_coder ;;
        *Qwen3*)       TOOL_CALL_PARSER=qwen3_xml ;;
        *)             TOOL_CALL_PARSER=hermes ;;
    esac
fi

mkdir -p "$LOG_DIR" "$PRED_DIR"

EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--subset-size 50)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$PRED_DIR/litellm_${TIMESTAMP}.jsonl"}
VLLM_LOG="$LOG_DIR/vllm_serve_${TIMESTAMP}.log"

echo "=== Baseline: LiteLLM + vLLM (no Pie) ==="
echo "  Harness:  $HARNESS"
echo "  Model:    $MODEL"
echo "  Label:    $LABEL"
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
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enforce-eager \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    $VLLM_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "echo 'Stopping vllm (PID $VLLM_PID)'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

echo "  Waiting for vllm to be ready..."
READY=0
for i in $(seq 1 1800); do
    if curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q "200"; then
        echo "  vllm is up ($((i * 2))s)"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vllm died. Check $VLLM_LOG"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    sleep 2
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vllm did not report ready within the timeout. Check $VLLM_LOG"
    tail -30 "$VLLM_LOG"
    exit 1
fi

# Run the benchmark
echo ""
echo "[2/2] Running $HARNESS (LiteLLM baseline)..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python -m "$HARNESS" \
    --backend litellm \
    --model "openai/$MODEL" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --api-key dummy \
    --output "$OUTPUT" \
    --label "$LABEL" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Predictions: $OUTPUT"
