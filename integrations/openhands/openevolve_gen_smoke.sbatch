#!/usr/bin/env bash
#SBATCH --job-name=oe-gen-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/oe_gen_smoke_%j.out

# GPU smoke for the openevolve-generation inferlet: boots pie serve (7B coder),
# installs the inferlet, and runs openevolve_gen_smoke.py, which asserts the
# analyze-then-diverge KV-sharing telemetry (L1p opened across calls with
# prefill 0, L1g generated/reused, shared_prefill_saved = (N-1)*prefix).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_vllm_config.toml

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation
mkdir -p logs

echo "[1/3] Building openevolve-generation inferlet (wasm32-wasip2)..."
( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )

echo "[2/3] Booting pie serve on port $PIE_PORT..."
PIE_LOG="logs/pie_serve_oe_gen_smoke_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'echo "Stopping pie serve (PID $PIE_PID)"; kill "$PIE_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  pie serve is up ($((i * 2))s)"
        break
    fi
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo "ERROR: pie serve died during startup. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
    sleep 2
    if [ "$i" -eq 120 ]; then
        echo "ERROR: pie serve not ready within timeout. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
done

echo "[3/3] Running openevolve-generation smoke..."
PYTHONPATH="" .venv/bin/python openevolve_gen_smoke.py
