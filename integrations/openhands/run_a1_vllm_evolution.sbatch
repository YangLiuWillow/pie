#!/usr/bin/env bash
#SBATCH --job-name=oe-a1-vllm
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:55:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/oe_a1_vllm_%j.out

# Fairness arm A1: OpenEvolve analyze-then-diverge on a standalone vLLM
# OpenAI server with automatic prefix caching (APC) ON — the honest non-fork
# equivalent of the Pie B2 arm. Boots `vllm serve` (Qwen3-Coder-30B, same
# util/len/eager as the Pie config), then runs run_pie_evolution.py with
# PIE_OE_BACKEND=vllm + PIE_OE_A1_TWOPHASE=1.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VLLM_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export VLLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export PIE_OE_BACKEND=vllm
export PIE_OE_A1_TWOPHASE=${PIE_OE_A1_TWOPHASE:-1}
export PIE_OE_SKIP_ANALYSIS=0
export OE_ITERS=${OE_ITERS:-12}
export OE_MAX_TOKENS=${OE_MAX_TOKENS:-2048}
export OE_TIMEOUT=${OE_TIMEOUT:-300}
mkdir -p logs

echo "[1/2] Booting vllm serve (APC on)..."
VLLM_LOG="logs/vllm_serve_a1_${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --port 8000 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enforce-eager \
    --served-model-name "$VLLM_MODEL" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "Stopping vllm (PID $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 240); do
    if grep -qE "Application startup complete|Uvicorn running on" "$VLLM_LOG" 2>/dev/null; then
        echo "  vllm serve is up ($((i * 3))s)"; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vllm serve died. Check $VLLM_LOG"; tail -50 "$VLLM_LOG"; exit 1
    fi
    sleep 3
    if [ "$i" -eq 240 ]; then
        echo "ERROR: vllm serve not ready in time. Check $VLLM_LOG"; tail -50 "$VLLM_LOG"; exit 1
    fi
done

echo "[2/2] Running OpenEvolve A1 (two-phase, APC) ..."
PYTHONPATH="" .venv/bin/python run_pie_evolution.py
