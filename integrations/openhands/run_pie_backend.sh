#!/usr/bin/env bash
# Phase 1 benchmark run — Pie backend (PieLLM → pie serve → vllm driver → GPU)
# Usage: bash run_pie_backend.sh [--subset-size N] [--instance-id ID]
# Set HARNESS=benchmarks.run_humanevalfix for the lighter smoke-test harness
# instead of the default SWE-Bench one (in which case pass --task-id instead
# of --instance-id, and note --subset-size defaults to 20 there, not 50).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIE=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
BACKEND=${BACKEND:-pie}
CFG=${CFG:-$SCRIPT_DIR/tests/fixtures/pie_cuda_vllm_config.toml}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
LABEL=${LABEL:-pie+qwen2.5-coder-7b}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-pie_qwen25_coder_7b}
HARNESS=${HARNESS:-benchmarks.run_swe_bench}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1800}
VENV=$SCRIPT_DIR/.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
LOG_DIR=$SCRIPT_DIR/logs
PRED_DIR=$SCRIPT_DIR/predictions
PIE_PORT=18080

if [ "$BACKEND" = "pie-agent" ]; then
    WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/target/wasm32-wasip2/release/openhands_agent.wasm
    MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/Pie.toml
    INFERLET_NAME="openhands-agent@0.1.0"
else
    WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm
    MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-completion/Pie.toml
    INFERLET_NAME="openhands-completion@0.1.0"
fi

mkdir -p "$LOG_DIR" "$PRED_DIR"

# Pass remaining args through to $HARNESS
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--subset-size 50)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$PRED_DIR/${OUTPUT_PREFIX}_${TIMESTAMP}.jsonl"}
PIE_LOG="$LOG_DIR/pie_serve_${TIMESTAMP}.log"

echo "=== Phase 1 SWE-Bench: Pie backend ==="
echo "  Config:   $CFG"
echo "  Output:   $OUTPUT"
echo "  Pie log:  $PIE_LOG"
echo "  Args:     ${EXTRA_ARGS[*]}"

# Boot pie serve
echo ""
echo "[1/3] Starting pie serve on port $PIE_PORT..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE serve --config "$CFG" --port $PIE_PORT --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap "echo 'Stopping pie serve (PID $PIE_PID)'; kill $PIE_PID 2>/dev/null; wait $PIE_PID 2>/dev/null" EXIT

# Wait for server ready (polls up to 60 min — a cold model download/load can
# take a long time; requires [server].verbose = true in $CFG so pie-server
# actually prints "pie-server serving on ..." once it's bound and ready).
echo "  Waiting for pie serve to be ready..."
READY=0
for i in $(seq 1 1800); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  pie serve is up ($((i * 2))s)"
        READY=1
        break
    fi
    if ! kill -0 $PIE_PID 2>/dev/null; then
        echo "ERROR: pie serve died. Check $PIE_LOG"
        tail -30 "$PIE_LOG"
        exit 1
    fi
    sleep 2
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: pie serve did not report ready within the timeout. Check $PIE_LOG"
    tail -30 "$PIE_LOG"
    exit 1
fi

# Install the inferlet
echo ""
echo "[2/3] Installing $INFERLET_NAME inferlet..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 \
  WASM="$WASM" MANIFEST="$MANIFEST" \
  $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import PieClient

async def install():
    async with PieClient("ws://127.0.0.1:18080") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  inferlet installed")

asyncio.run(install())
PY

# Run the benchmark
echo ""
echo "[3/3] Running $HARNESS ($BACKEND backend)..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python -m "$HARNESS" \
    --backend "$BACKEND" \
    --pie-uri ws://127.0.0.1:$PIE_PORT \
    --pie-inferlet "$INFERLET_NAME" \
    --model "$MODEL" \
    --pie-request-timeout-s "$REQUEST_TIMEOUT_S" \
    --output "$OUTPUT" \
    --label "$LABEL" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Predictions: $OUTPUT"
