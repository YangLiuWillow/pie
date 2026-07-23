#!/usr/bin/env bash
#SBATCH --job-name=oe-meas-vllm-cp
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/oe_meas_vllm_cp_%j.out

# MEASUREMENT — vLLM baseline (APC on, A1 two-phase), function_minimization.
# Head-to-head partner = oe_measure_pie_fm.sbatch. Same example/iters/seeds/
# max_tokens/timeout as the Pie arm so time + accuracy are comparable.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VLLM_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export VLLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export PIE_OE_BACKEND=vllm
export PIE_OE_A1_TWOPHASE=1
export PIE_OE_SKIP_ANALYSIS=0

export OE_EXAMPLE=circle_packing
export OE_CONFIG=config_phase_1.yaml
export OE_ITERS=8
export OE_MAX_TOKENS=8192
export OE_TIMEOUT=900
SEEDS="0 1"
mkdir -p logs

VLLM_LOG="logs/vllm_serve_meas_cp_${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --port 8000 --enable-prefix-caching --gpu-memory-utilization 0.90 \
    --max-model-len 32768 --enforce-eager --served-model-name "$VLLM_MODEL" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
    grep -qE "Application startup complete|Uvicorn running on" "$VLLM_LOG" 2>/dev/null && { echo "vllm up (${i}x3s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm died"; tail -50 "$VLLM_LOG"; exit 1; }
    sleep 3
done

for SEED in $SEEDS; do
    echo "########## SEED $SEED ##########"
    OE_SEED=$SEED PYTHONPATH="" .venv/bin/python run_pie_evolution.py || echo "seed $SEED FAILED"
done

echo "=== OE_MEASURE lines (this job) ==="
grep -h "^OE_MEASURE " "logs/oe_meas_vllm_cp_${SLURM_JOB_ID}.out" || true
