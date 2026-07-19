#!/usr/bin/env bash
# Phase 1 benchmark run — Pie backend (PieLLM → pie serve → vllm driver → GPU)
# Usage: bash run_pie_backend.sh [--subset-size N] [--instance-id ID]
# Set HARNESS=benchmarks.run_humanevalfix for the lighter smoke-test harness
# instead of the default SWE-Bench one (in which case pass --task-id instead
# of --instance-id, and note --subset-size defaults to 20 there, not 50).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIE=${PIE_BIN:-/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie}
BACKEND=${BACKEND:-pie}
CFG=${CFG:-$SCRIPT_DIR/tests/fixtures/pie_cuda_vllm_config.toml}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
LABEL=${LABEL:-pie+qwen2.5-coder-7b}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-pie_qwen25_coder_7b}
HARNESS=${HARNESS:-benchmarks.run_swe_bench}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1800}
VENV=$SCRIPT_DIR/.venv
HF_HOME=${HF_HOME:-/nfs/roberts/scratch/pi_ql324/ly337/hf_cache}
LOG_DIR=$SCRIPT_DIR/logs
PRED_DIR=$SCRIPT_DIR/predictions
PIE_PORT=${PIE_PORT:-18080}
MAX_SERVER_RESTARTS=${MAX_SERVER_RESTARTS:-3}

if [ "$BACKEND" = "pie-agent" ]; then
    WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/target/wasm32-wasip2/release/openhands_agent.wasm
    MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/Pie.toml
    INFERLET_NAME="openhands-agent@0.1.0"
elif [ "$BACKEND" = "pie-session" ]; then
    WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm
    MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session/Pie.toml
    INFERLET_NAME="openhands-coder-session@0.1.0"
else
    WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm
    MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-completion/Pie.toml
    INFERLET_NAME="openhands-completion@0.1.0"
fi

# "pie-session" is a script-level alias: same PieLLM harness backend as
# "pie", plus the session inferlet and the --pie-session flag (and
# --kv-verify when KV_VERIFY=1).
HARNESS_BACKEND=$BACKEND
SESSION_ARGS=()
if [ "$BACKEND" = "pie-session" ]; then
    HARNESS_BACKEND=pie
    SESSION_ARGS+=(--pie-session)
    if [ "${KV_VERIFY:-0}" = "1" ]; then
        SESSION_ARGS+=(--kv-verify)
    fi
fi

mkdir -p "$LOG_DIR" "$PRED_DIR"

# Pass remaining args through to $HARNESS
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--subset-size 50)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-"$PRED_DIR/${OUTPUT_PREFIX}_${TIMESTAMP}.jsonl"}

echo "=== Phase 1 SWE-Bench: Pie backend ==="
echo "  Config:   $CFG"
echo "  Output:   $OUTPUT"
echo "  Args:     ${EXTRA_ARGS[*]}"

PIE_PID=""

cleanup() {
    if [ -n "$PIE_PID" ]; then
        echo "Stopping pie serve (PID $PIE_PID)"
        kill "$PIE_PID" 2>/dev/null
        wait "$PIE_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

start_server() {
    local pie_log="$1"

    # Kill any leftover server from a prior iteration.
    if [ -n "$PIE_PID" ]; then
        kill "$PIE_PID" 2>/dev/null || true
        wait "$PIE_PID" 2>/dev/null || true
        PIE_PID=""
    fi

    echo ""
    echo "  Starting pie serve on port $PIE_PORT..."
    PYTHONPATH="" HF_HOME=$HF_HOME \
      $PIE serve --config "$CFG" --port $PIE_PORT --no-auth > "$pie_log" 2>&1 &
    PIE_PID=$!

    echo "  Waiting for pie serve to be ready (PID $PIE_PID)..."
    local ready=0
    for i in $(seq 1 1800); do
        if grep -q "pie-server serving on" "$pie_log" 2>/dev/null; then
            echo "  pie serve is up ($((i * 2))s)"
            ready=1
            break
        fi
        if ! kill -0 $PIE_PID 2>/dev/null; then
            echo "ERROR: pie serve died during startup. Check $pie_log"
            tail -30 "$pie_log"
            return 1
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "ERROR: pie serve did not report ready within the timeout. Check $pie_log"
        tail -30 "$pie_log"
        return 1
    fi
    return 0
}

install_inferlet() {
    echo "  Installing $INFERLET_NAME inferlet..."
    PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 \
      WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
      $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import PieClient

async def install():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  inferlet installed")

asyncio.run(install())
PY
}

# --- Initial server boot ------------------------------------------------
PIE_LOG="$LOG_DIR/pie_serve_${TIMESTAMP}.log"
echo ""
echo "[1/3] Booting pie serve..."
start_server "$PIE_LOG"

echo ""
echo "[2/3] Installing inferlet..."
install_inferlet

# --- Benchmark loop with auto-restart -----------------------------------
# The harness exits with code 42 when the Pie server dies mid-run. When
# that happens, restart the server, reinstall the inferlet, and re-run
# with --resume so already-completed instances are skipped.
echo ""
echo "[3/3] Running $HARNESS ($BACKEND backend)..."

RESTARTS=0
while true; do
    set +e
    PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
      $VENV/bin/python -m "$HARNESS" \
        --backend "$HARNESS_BACKEND" \
        "${SESSION_ARGS[@]}" \
        --pie-uri ws://127.0.0.1:$PIE_PORT \
        --pie-inferlet "$INFERLET_NAME" \
        --model "$MODEL" \
        --pie-request-timeout-s "$REQUEST_TIMEOUT_S" \
        --output "$OUTPUT" \
        --label "$LABEL" \
        --verbose \
        --resume \
        "${EXTRA_ARGS[@]}"
    RC=$?
    set -e

    if [ $RC -eq 0 ]; then
        break
    fi

    if [ $RC -ne 42 ]; then
        echo "ERROR: harness exited with code $RC (not a server-death restart)"
        exit $RC
    fi

    # Exit code 42 = server died.
    RESTARTS=$((RESTARTS + 1))
    if [ $RESTARTS -gt $MAX_SERVER_RESTARTS ]; then
        echo "ERROR: server died $RESTARTS times, giving up"
        exit 1
    fi

    echo ""
    echo "=== Server died — restart $RESTARTS/$MAX_SERVER_RESTARTS ==="
    PIE_LOG="$LOG_DIR/pie_serve_${TIMESTAMP}_restart${RESTARTS}.log"
    start_server "$PIE_LOG"
    install_inferlet
    echo "  Resuming benchmark..."
done

echo ""
echo "Done. Predictions: $OUTPUT"
