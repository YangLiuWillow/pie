#!/usr/bin/env bash
# Orchestration script for the autoresearch agent.
#
# Starts pie serve, installs the autoresearch-agent inferlet, starts the
# tool server, then launches the agent with the given program.md.
#
# Required env vars:
#   AUTORESEARCH_DIR  — path to the autoresearch repo (contains train.py)
#   PROGRAM_MD        — path to program.md (research instructions)
#
# Optional env vars:
#   CFG               — pie config TOML (default: tests/fixtures/pie_cuda_vllm_config_32b.toml
#                       from the openhands integration — reuse the same model config)
#   MAX_EXPERIMENTS   — max experiment iterations (default: 100)
#   TOOL_SERVER_PORT  — port for the tool server (default: 9876)
#   PIE_BIN           — path to pie binary (default: ../../target/release/pie)
#   WASM_PATH         — path to built WASM (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Required
: "${AUTORESEARCH_DIR:?Set AUTORESEARCH_DIR to the autoresearch repo path}"
: "${PROGRAM_MD:?Set PROGRAM_MD to the path to program.md}"

# Optional with defaults
CFG="${CFG:-$PIE_ROOT/integrations/openhands/tests/fixtures/pie_cuda_vllm_config_32b.toml}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-100}"
TOOL_SERVER_PORT="${TOOL_SERVER_PORT:-9876}"
PIE_BIN="${PIE_BIN:-$PIE_ROOT/target/release/pie}"
WASM_PATH="${WASM_PATH:-$PIE_ROOT/inferlets/autoresearch-agent/target/wasm32-wasip2/release/autoresearch_agent.wasm}"
MANIFEST_PATH="$PIE_ROOT/inferlets/autoresearch-agent/Pie.toml"

# Scratch/cache dirs
HF_HOME="${HF_HOME:-/nfs/roberts/scratch/pi_ql324/ly337/hf_cache}"
export HF_HOME

# Clear HPC PYTHONPATH pollution
export PYTHONPATH=""

echo "=== Autoresearch Agent ==="
echo "Config:          $CFG"
echo "Autoresearch dir: $AUTORESEARCH_DIR"
echo "Program.md:      $PROGRAM_MD"
echo "Max experiments: $MAX_EXPERIMENTS"
echo "Tool server:     127.0.0.1:$TOOL_SERVER_PORT"
echo "PIE binary:      $PIE_BIN"
echo "WASM:            $WASM_PATH"
echo ""

# Validate inputs
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config not found: $CFG" >&2
    exit 1
fi
if [[ ! -d "$AUTORESEARCH_DIR" ]]; then
    echo "ERROR: Autoresearch dir not found: $AUTORESEARCH_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROGRAM_MD" ]]; then
    echo "ERROR: program.md not found: $PROGRAM_MD" >&2
    exit 1
fi
if [[ ! -f "$PIE_BIN" ]]; then
    echo "ERROR: pie binary not found: $PIE_BIN" >&2
    exit 1
fi
if [[ ! -f "$WASM_PATH" ]]; then
    echo "ERROR: WASM not found: $WASM_PATH" >&2
    echo "Build it with: cd $PIE_ROOT/inferlets/autoresearch-agent && cargo build --target wasm32-wasip2 --release"
    exit 1
fi

# --- 1. Start pie serve ---
echo "[1/4] Starting pie serve..."
PIE_LOG="$SCRIPT_DIR/logs/pie_serve_$$.log"
mkdir -p "$SCRIPT_DIR/logs"
"$PIE_BIN" serve --config "$CFG" > "$PIE_LOG" 2>&1 &
PIE_PID=$!
echo "  PID: $PIE_PID, log: $PIE_LOG"

# Wait for pie serve readiness
echo "  Waiting for pie serve to be ready..."
WAIT_START=$(date +%s)
MAX_WAIT=3600  # 60 minutes (cold model download can be slow)
while ! grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; do
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo "ERROR: pie serve exited unexpectedly. Log:" >&2
        tail -20 "$PIE_LOG" >&2
        exit 1
    fi
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if (( ELAPSED > MAX_WAIT )); then
        echo "ERROR: pie serve not ready after ${MAX_WAIT}s" >&2
        kill "$PIE_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done
echo "  Ready! ($(( $(date +%s) - WAIT_START ))s)"

# --- 2. Install inferlet ---
echo "[2/4] Installing autoresearch-agent inferlet..."
"$PIE_BIN" install --path "$WASM_PATH" --manifest "$MANIFEST_PATH" --force-overwrite
echo "  Installed."

# --- 3. Start tool server ---
echo "[3/4] Starting tool server on port $TOOL_SERVER_PORT..."
python3 "$SCRIPT_DIR/tool_server.py" \
    --working-dir "$AUTORESEARCH_DIR" \
    --port "$TOOL_SERVER_PORT" &
TOOL_PID=$!
sleep 1
if ! kill -0 "$TOOL_PID" 2>/dev/null; then
    echo "ERROR: Tool server failed to start" >&2
    kill "$PIE_PID" 2>/dev/null || true
    exit 1
fi
echo "  PID: $TOOL_PID"

# --- 4. Launch the agent ---
echo "[4/4] Launching autoresearch agent..."
echo ""

PROGRAM_CONTENT=$(cat "$PROGRAM_MD")
INPUT=$(python3 -c "
import json, sys
print(json.dumps({
    'program': sys.stdin.read(),
    'tool_server_url': 'http://127.0.0.1:$TOOL_SERVER_PORT',
    'max_experiments': $MAX_EXPERIMENTS,
}))" <<< "$PROGRAM_CONTENT")

"$PIE_BIN" run --name "autoresearch-agent@0.1.0" --input "$INPUT"

echo ""
echo "=== Done ==="

# Cleanup
kill "$TOOL_PID" 2>/dev/null || true
kill "$PIE_PID" 2>/dev/null || true
