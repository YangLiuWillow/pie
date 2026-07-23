#!/usr/bin/env bash
#SBATCH --job-name=abp-vllm
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:20:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/abp_vllm_%j.out

# MEMORY-PRESSURE A/B — vLLM A1 two-phase + APC arm. Same sweep as the Pie job.
# APC hit% should fall as concurrency pushes the distinct working set past the
# 244,704-token cache (eviction window between phase-1 and phase-2). Model via
# VLLM_MODEL (default 30B-A3B). Weights download to HF_HOME on scratch.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VLLM_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

export BENCH_EXAMPLES=${BENCH_EXAMPLES:-function_minimization}
export BENCH_N=${BENCH_N:-4}
export BENCH_CONC=${BENCH_CONC:-8,16,32,64}
export BENCH_REPEATS=${BENCH_REPEATS:-2}
export BENCH_PREFIX_PAD_TOKENS=${BENCH_PREFIX_PAD_TOKENS:-8000}
export BENCH_MAXTOK=${BENCH_MAXTOK:-16}
export BENCH_ANALYSIS_MAXTOK=${BENCH_ANALYSIS_MAXTOK:-64}
GPU_UTIL=${GPU_UTIL:-0.90}
mkdir -p logs

echo "[1/2] Booting vllm serve ($VLLM_MODEL, util=$GPU_UTIL)..."
VLLM_LOG="logs/vllm_serve_abp_${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --port 8000 --enable-prefix-caching --enable-prompt-tokens-details \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len 32768 --enforce-eager \
    --served-model-name "$VLLM_MODEL" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!; trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
    grep -qE "Application startup complete|Uvicorn running on" "$VLLM_LOG" 2>/dev/null && { echo "  up ($((i*3))s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vllm died"; tail -50 "$VLLM_LOG"; exit 1; }
    sleep 3
done
echo "  KV cache: $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$VLLM_LOG" | tail -1)"

echo "[2/2] Running pressure sweep (vllm two-phase + APC)..."
BENCH_ARM=vllm_twophase BENCH_OUT=logs/abp_vllm_twophase_${SLURM_JOB_ID}.jsonl \
    PYTHONPATH="" .venv/bin/python ab_bench.py
echo "=== report (vllm) ==="
PYTHONPATH="" .venv/bin/python ab_bench_report.py logs/abp_vllm_twophase_${SLURM_JOB_ID}.jsonl
