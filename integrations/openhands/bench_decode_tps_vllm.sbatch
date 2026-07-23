#!/usr/bin/env bash
#SBATCH --job-name=dtps-vllm
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/dtps_vllm_%j.out

# Decode-throughput vs context-length — vLLM arm (standalone vllm serve).
# Same slope method, ignore_eos+min_tokens forces exact decode counts.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VLLM_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export VLLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

export BENCH_ARM=vllm
export DT_L=${DT_L:-2000,5000,20000,50000}
export DT_M=${DT_M:-64,512}
export DT_REPEATS=${DT_REPEATS:-3}
export BENCH_OUT=logs/decode_tps_vllm_${SLURM_JOB_ID}.jsonl
mkdir -p logs

echo "[1/2] Booting vllm serve (ctx 64k)..."
VLLM_LOG="logs/vllm_serve_dtps_${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --port 8000 --enable-prefix-caching --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.90 --max-model-len 65536 --enforce-eager \
    --served-model-name "$VLLM_MODEL" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!; trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
    grep -qE "Application startup complete|Uvicorn running on" "$VLLM_LOG" 2>/dev/null && { echo "  up ($((i*3))s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vllm died"; tail -50 "$VLLM_LOG"; exit 1; }
    sleep 3
done
echo "[2/2] Running decode-tps (vllm)..."
PYTHONPATH="" .venv/bin/python decode_tps_bench.py
