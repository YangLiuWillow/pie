#!/usr/bin/env bash
# Run TerminalBench evaluation with PIE + Apptainer.
#
# Starts pie serve, installs the openhands-agent inferlet, then runs
# TerminalBench tasks via Apptainer containers.
#
# Required env vars:
#   TBENCH_DIR   — path to terminal-bench repo clone
#
# Optional env vars:
#   CFG          — pie config TOML
#   MAX_STEPS    — max agent steps per task (default: 50)
#   TASK_LIST    — file with task names (one per line), or "easy" / "medium" / "hard"
#   NUM_TASKS    — max tasks to run (default: all)
#   OUTPUT       — output JSONL path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIE_BIN="$PIE_ROOT/target/release/pie"
OPENHANDS_VENV="$PIE_ROOT/integrations/openhands/.venv"

: "${TBENCH_DIR:?Set TBENCH_DIR to the terminal-bench repo path}"

CFG="${CFG:-$PIE_ROOT/integrations/openhands/tests/fixtures/pie_cuda_vllm_config_30b_moe.toml}"
MAX_STEPS="${MAX_STEPS:-50}"
TASK_LIST="${TASK_LIST:-easy}"
NUM_TASKS="${NUM_TASKS:-}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/results/pie_agent_$(date +%Y%m%d_%H%M%S).jsonl}"

export HF_HOME="${HF_HOME:-/nfs/roberts/scratch/pi_ql324/ly337/hf_cache}"
export PYTHONPATH=""

WASM="$PIE_ROOT/inferlets/openhands-agent/target/wasm32-wasip2/release/openhands_agent.wasm"
MANIFEST="$PIE_ROOT/inferlets/openhands-agent/Pie.toml"

mkdir -p "$(dirname "$OUTPUT")" "$SCRIPT_DIR/logs"

echo "=== TerminalBench + PIE Agent ==="
echo "  Config:     $CFG"
echo "  Tasks:      $TASK_LIST"
echo "  Max steps:  $MAX_STEPS"
echo "  Output:     $OUTPUT"
echo ""

# --- 1. Start pie serve ---
echo "[1/3] Starting pie serve..."
PIE_LOG="$SCRIPT_DIR/logs/pie_serve_$$.log"
"$PIE_BIN" serve --config "$CFG" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap "kill $PIE_PID 2>/dev/null; wait $PIE_PID 2>/dev/null" EXIT

echo "  Waiting for pie serve..."
for i in $(seq 1 720); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  Ready ($((i * 5))s)"
        break
    fi
    if ! kill -0 $PIE_PID 2>/dev/null; then
        echo "ERROR: pie serve died"
        tail -20 "$PIE_LOG"
        exit 1
    fi
    sleep 5
done

# --- 2. Install inferlet ---
echo "[2/3] Installing openhands-agent inferlet..."
WASM="$WASM" MANIFEST="$MANIFEST" \
  "$OPENHANDS_VENV/bin/python" -c "
import asyncio, os
from pie_client import PieClient
async def install():
    async with PieClient('ws://127.0.0.1:8080') as c:
        await c.authenticate('local-dev')
        await c.install_program(os.environ['WASM'], os.environ['MANIFEST'], force_overwrite=True)
        print('  Installed')
asyncio.run(install())
"

# --- 3. Collect task dirs ---
echo "[3/3] Running TerminalBench..."

TASK_ARGS=()
if [[ "$TASK_LIST" == "easy" || "$TASK_LIST" == "medium" || "$TASK_LIST" == "hard" ]]; then
    while IFS= read -r task_name; do
        task_dir="$TBENCH_DIR/original-tasks/$task_name"
        if [[ -d "$task_dir" ]]; then
            TASK_ARGS+=(--task-dir "$task_dir")
        fi
    done < <(
        grep -l "difficulty: $TASK_LIST" "$TBENCH_DIR"/original-tasks/*/task.yaml 2>/dev/null | \
        xargs -I{} dirname {} | xargs -I{} basename {} | \
        sort | head -${NUM_TASKS:-999}
    )
elif [[ -f "$TASK_LIST" ]]; then
    while IFS= read -r task_name; do
        [[ -z "$task_name" || "$task_name" == \#* ]] && continue
        task_dir="$TBENCH_DIR/original-tasks/$task_name"
        if [[ -d "$task_dir" ]]; then
            TASK_ARGS+=(--task-dir "$task_dir")
        else
            echo "  WARNING: task not found: $task_name"
        fi
    done < "$TASK_LIST"
else
    for task_dir in "$TBENCH_DIR"/original-tasks/*/; do
        TASK_ARGS+=(--task-dir "$task_dir")
    done
fi

echo "  ${#TASK_ARGS[@]} task args ($(( ${#TASK_ARGS[@]} / 2 )) tasks)"

"$OPENHANDS_VENV/bin/python" "$SCRIPT_DIR/apptainer_runner.py" \
    "${TASK_ARGS[@]}" \
    --pie-uri ws://127.0.0.1:8080 \
    --pie-inferlet "openhands-agent@0.1.0" \
    --max-steps "$MAX_STEPS" \
    --output "$OUTPUT"

echo ""
echo "Done. Results: $OUTPUT"
