#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-genprobe
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_genprobe_%j.out

# Direct generation probe: does the native CUDA driver decode coherent text
# and stop on stop-tokens? Boots pie serve (cuda_native, 30B), installs the
# text-completion inferlet, runs a few prompts, prints raw output. Isolates
# driver decode from the OpenHands agent's tool-call parsing.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

PIE=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml
PORT=18085
VENV=.venv
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
WASM=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/text-completion/target/wasm32-wasip2/release/text_completion.wasm
MANIFEST=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/text-completion/Pie.toml
LOG=logs/genprobe_serve_${SLURM_JOB_ID}.log

echo "[1/3] booting pie serve (cuda_native)..."
PYTHONPATH="" $PIE serve --config "$CFG" --port $PORT --no-auth > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
  grep -q "pie-server serving on" "$LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  sleep 2
done

echo "[2/3] installing text-completion inferlet..."
PYTHONPATH="" WASM="$WASM" MANIFEST="$MANIFEST" PORT="$PORT" $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import PieClient
async def go():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed")
asyncio.run(go())
PY

echo "[3/3] running generation probe..."
PYTHONPATH="" PIE_URI="ws://127.0.0.1:$PORT" $VENV/bin/python probe_cuda_native_gen.py

echo "=== driver log tail ==="
grep -iE 'fire_batch failed|Buffer overflow|CUDA error|serving on|kv_cache' "$LOG" | tail -6
