#!/usr/bin/env bash
#SBATCH --job-name=ab-vllm
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:10:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/ab_vllm_%j.out

# A/B harness — vLLM arms (A1 two-phase+APC and A0 naive). Boots vllm serve
# (30B, prefix caching + prompt-token-details ON so cached_tokens is reported),
# then runs ab_bench.py for each vllm arm across the same sweep/examples.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VLLM_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export VLLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

export BENCH_EXAMPLES=${BENCH_EXAMPLES:-function_minimization,circle_packing,signal_processing}
export BENCH_N=${BENCH_N:-1,2,4,8}
export BENCH_CONC=${BENCH_CONC:-1,2,4,8}
export BENCH_REPEATS=${BENCH_REPEATS:-3}
export BENCH_MAXTOK=${BENCH_MAXTOK:-256}
mkdir -p logs

echo "[1/3] Booting vllm serve (APC + prompt-token-details)..."
VLLM_LOG="logs/vllm_serve_ab_${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --port 8000 --enable-prefix-caching --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --enforce-eager \
    --served-model-name "$VLLM_MODEL" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!; trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
    grep -qE "Application startup complete|Uvicorn running on" "$VLLM_LOG" 2>/dev/null && { echo "  up ($((i*3))s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vllm died"; tail -50 "$VLLM_LOG"; exit 1; }
    sleep 3
done

echo "[2/3] A1 (two-phase + APC)..."
BENCH_ARM=vllm_twophase BENCH_OUT=logs/ab_bench_vllm_twophase_${SLURM_JOB_ID}.jsonl \
    PYTHONPATH="" .venv/bin/python ab_bench.py

echo "[3/3] A0 (naive single-shot)..."
BENCH_ARM=vllm_naive BENCH_OUT=logs/ab_bench_vllm_naive_${SLURM_JOB_ID}.jsonl \
    PYTHONPATH="" .venv/bin/python ab_bench.py

echo "=== report (vllm arms) ==="
PYTHONPATH="" .venv/bin/python ab_bench_report.py \
    logs/ab_bench_vllm_twophase_${SLURM_JOB_ID}.jsonl logs/ab_bench_vllm_naive_${SLURM_JOB_ID}.jsonl
