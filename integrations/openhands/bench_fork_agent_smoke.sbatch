#!/usr/bin/env bash
#SBATCH --job-name=fork-agent-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/fork_agent_smoke_%j.out

# Step-2 GPU smoke: fork-based test-time scaling in the openhands-agent
# inferlet (Pattern A). Runs ONE instance with num_branches=2: trunk runs to
# branch_at_step, then the KV context AND the tool-server workspace fork 2 ways
# and each branch finishes independently. Validates: a `branches` array of
# length 2, distinct workspace_ids, per-branch diffs from isolated workspaces.
# temperature>0 so the forked contexts actually diverge.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_fork_smoke_ws_${SLURM_JOB_ID}
mkdir -p "$SWEB_WS_ROOT" logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
VENV=.venv
WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/target/wasm32-wasip2/release/openhands_agent.wasm
MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent/Pie.toml
PIE_LOG=logs/pie_serve_fork_smoke_${SLURM_JOB_ID}.log

# Smoke parameters
export INSTANCE_ID=${INSTANCE_ID:-django__django-14373}
export NUM_BRANCHES=${NUM_BRANCHES:-2}
export BRANCH_AT_STEP=${BRANCH_AT_STEP:-3}
export TEMPERATURE=${TEMPERATURE:-0.7}
export MAX_STEPS=${MAX_STEPS:-25}

# --- [1/4] Build the openhands-agent wasm (picks up the Step-2 fork code) ---
echo "[1/4] Building openhands-agent wasm..."
( cd /nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent \
    && cargo build --release --target wasm32-wasip2 )

PIE_PID=""
cleanup() {
    if [ -n "$PIE_PID" ]; then
        echo "Stopping pie serve (PID $PIE_PID)"
        kill "$PIE_PID" 2>/dev/null || true
        wait "$PIE_PID" 2>/dev/null || true
    fi
    rm -rf "$SWEB_WS_ROOT" 2>/dev/null || true
}
trap cleanup EXIT

# --- [2/4] Boot pie serve -----------------------------------------------
echo "[2/4] Booting pie serve on port $PIE_PORT..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE_BIN serve --config "$CFG" --port $PIE_PORT --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
ready=0
for i in $(seq 1 1800); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  pie serve up ($((i * 2))s)"; ready=1; break
    fi
    if ! kill -0 $PIE_PID 2>/dev/null; then
        echo "ERROR: pie serve died during startup"; tail -30 "$PIE_LOG"; exit 1
    fi
    sleep 2
done
[ "$ready" -eq 1 ] || { echo "ERROR: pie serve not ready"; tail -30 "$PIE_LOG"; exit 1; }

# --- [3/4] Install the inferlet -----------------------------------------
echo "[3/4] Installing openhands-agent inferlet..."
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

# --- [4/4] Run the fork smoke -------------------------------------------
echo "[4/4] Running fork_agent_smoke.py..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python fork_agent_smoke.py
RC=$?

echo ""
echo "=== smoke exit code: $RC ==="
exit $RC
