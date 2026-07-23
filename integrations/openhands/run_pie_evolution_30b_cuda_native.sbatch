#!/usr/bin/env bash
#SBATCH --job-name=oe-cuda-native-30b
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:55:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/oe_cuda_native_30b_%j.out

# OpenEvolve-on-Pie on the NATIVE CUDA driver (in-process, no vLLM subprocess) +
# the content-addressed APC inferlet. cuda_native counterpart of
# run_pie_evolution_30b_atd.sbatch. Motivation: the native driver batches
# multi-request Qwen3-MoE decode (the vLLM driver did not — the fan-out loss
# root cause), and the inferlet now self-keys L1p/L1g by rendered-prompt content.
#
# Swaps vs the vLLM-driver runner: module-load CUDA/NCCL (the cuda_native pie
# binary links them), the repo pie binary built with --features driver-cuda, and
# the cuda_native config.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18096
export PIE_URI=ws://127.0.0.1:18096
export PIE_USER=local-dev
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation
export PIE_OE_WASM="$INFERLET_DIR/target/wasm32-wasip2/release/openevolve_generation.wasm"
export PIE_OE_MANIFEST="$INFERLET_DIR/Pie.toml"
# analyze-then-diverge arm by default (skip_analysis=0). Set 1 for direct diffs.
export PIE_OE_SKIP_ANALYSIS=${PIE_OE_SKIP_ANALYSIS:-0}
export OE_ITERS=${OE_ITERS:-12}
mkdir -p logs

echo "[1/3] Building openevolve-generation inferlet (wasm32-wasip2)..."
( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )

echo "[2/3] Booting pie serve (cuda_native) on port $PIE_PORT..."
PIE_LOG="logs/pie_serve_oe_cuda_native_30b_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'echo "Stopping pie serve (PID $PIE_PID)"; kill "$PIE_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  pie serve is up ($((i * 2))s)"; break
    fi
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo "ERROR: pie serve died. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
    sleep 2
    if [ "$i" -eq 120 ]; then
        echo "ERROR: pie serve not ready. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
done

echo "[3/3] Running OpenEvolve on Pie/cuda_native (skip_analysis=$PIE_OE_SKIP_ANALYSIS, iters=$OE_ITERS)..."
PYTHONPATH="" .venv/bin/python run_pie_evolution.py
