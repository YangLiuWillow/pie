#!/usr/bin/env bash
# =============================================================================
# FAIR litellm+vLLM baseline arm (boot fair vLLM, then run the OpenHands harness).
#
# Faithful to the upstream run_litellm_baseline.sh harness invocation, with the
# two crippling conditions removed and a fairness gate added:
#   - NO --enforce-eager  (CUDA graphs ON)
#   - refuses to run the benchmark unless assert_vllm_fair.sh passes on the banner
# Everything below the LLM boundary (backend=litellm, base_url, parser) is
# unchanged, so this stays a clean drop-in swap of the serving config only.
#
# Usage (mirrors the original):
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct OUTPUT=predictions/ab_litellm.jsonl \
#     bash run_litellm_baseline_fair.sh --instance-id django__django-12276 ... \
#          --temperature 0 --max-iterations 100
# =============================================================================
set -euo pipefail

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Harness tree = integrations/openhands (this bundle lives in runpod/ under it).
HARNESS_DIR=${HARNESS_DIR:-$(cd "$RUNPOD_DIR/.." && pwd)}
VENV=${HARNESS_VENV:-$HARNESS_DIR/.venv}
PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
LOG_DIR=$HARNESS_DIR/logs
PRED_DIR=$HARNESS_DIR/predictions
VLLM_PORT=${VLLM_PORT:-18000}
HARNESS=${HARNESS:-benchmarks.run_swe_bench}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
LABEL=${LABEL:-litellm-fair+$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')}
VLLM_TIER=${VLLM_TIER:-fair}   # crippled | graphs-only | fair (see README)
STRICT_FAIR=${STRICT_FAIR:-1}
EAGER_FLAG=""; [ "$VLLM_TIER" = "crippled" ] && EAGER_FLAG="--enforce-eager"

case "$MODEL" in
    *Qwen3-Coder*) TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder} ;;
    *Qwen3*)       TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml} ;;
    *)             TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-hermes} ;;
esac

mkdir -p "$LOG_DIR" "$PRED_DIR"
EXTRA_ARGS=("$@"); [ ${#EXTRA_ARGS[@]} -eq 0 ] && EXTRA_ARGS=(--subset-size 50)
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$PRED_DIR/litellm_fair_${TS}.jsonl"}
VLLM_LOG="$LOG_DIR/vllm_serve_fair_${TS}.log"

echo "=== FAIR baseline: litellm + vLLM (CUDA graphs ON, tuned MoE) ==="
echo "  Model: $MODEL   Output: $OUTPUT   vLLM log: $VLLM_LOG"

# --- boot fair vLLM (NOTE: no --enforce-eager) -------------------------------
PYTHONPATH="" HF_HOME=$HF_HOME \
  "$PIE_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    $EAGER_FLAG \
    ${VLLM_EXTRA_ARGS:-} \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

echo "  waiting for vllm..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -40 "$VLLM_LOG"; exit 1; }
    sleep 2
done

# --- fairness gate: banner must match the declared tier ----------------------
if ! bash "$RUNPOD_DIR/assert_vllm_fair.sh" "$VLLM_LOG" "$VLLM_TIER"; then
    [ "$STRICT_FAIR" = "1" ] && { echo "ABORT: vLLM banner does not meet tier '$VLLM_TIER' (see above)."; exit 3; }
    echo "WARN: tier assertion failed (STRICT_FAIR=0) — continuing."
fi

# --- run the harness (identical to upstream below the LLM boundary) ----------
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  "$VENV/bin/python" -m "$HARNESS" \
    --backend litellm \
    --model "openai/$MODEL" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --api-key dummy \
    --output "$OUTPUT" \
    --label "$LABEL" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo "Done. Predictions: $OUTPUT   (fair vLLM banner: $VLLM_LOG)"
