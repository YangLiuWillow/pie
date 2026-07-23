#!/usr/bin/env bash
#SBATCH --job-name=mask-condense-ab
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_condense_ab_%j.out

# Stage-2 A/B: mask-condense vs rebuild-condense on a REAL SWE-bench trajectory
# (low context limit forces several condensations). openhands-agent Pattern A.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_mask_ab_ws_${SLURM_JOB_ID}
mkdir -p "$SWEB_WS_ROOT" logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
VENV=.venv
INF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent
WASM=$INF/target/wasm32-wasip2/release/openhands_agent.wasm
MANIFEST=$INF/Pie.toml
PIE_LOG=logs/pie_serve_mask_ab_${SLURM_JOB_ID}.log

export INSTANCE_ID=${INSTANCE_ID:-sympy__sympy-19346}
export CONTEXT_LIMIT=${CONTEXT_LIMIT:-6000}
export MAX_STEPS=${MAX_STEPS:-50}
export KEEP_RECENT=${KEEP_RECENT:-10}

echo "[build] openhands-agent wasm..."
( cd "$INF" && cargo build --release --target wasm32-wasip2 )

PIE_PID=""
cleanup() { [ -n "$PIE_PID" ] && { kill "$PIE_PID" 2>/dev/null || true; wait "$PIE_PID" 2>/dev/null || true; }; rm -rf "$SWEB_WS_ROOT" 2>/dev/null || true; }
trap cleanup EXIT

echo "[boot] pie serve..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE_BIN serve --config "$CFG" --port $PIE_PORT --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
for i in $(seq 1 1800); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 $PIE_PID 2>/dev/null || { echo "ERROR: serve died"; tail -30 "$PIE_LOG"; exit 1; }
    sleep 2
done

echo "[install] inferlet..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import PieClient
async def install():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed")
asyncio.run(install())
PY

echo "[run] A/B..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python mask_condense_ab.py
RC=$?

echo ""
echo "=== condensation event counts ==="
echo "rebuild [condense] events + mask [condense-mask] events (interleaved across both runs):"
grep -cE "\[condense\] rebuilding" logs/mask_condense_ab_${SLURM_JOB_ID}.out 2>/dev/null | sed 's/^/  rebuild condensations: /' || true
grep -cE "\[condense-mask\] masked" logs/mask_condense_ab_${SLURM_JOB_ID}.out 2>/dev/null | sed 's/^/  mask condensations: /' || true
echo "=== A/B exit code: $RC ==="
exit $RC
