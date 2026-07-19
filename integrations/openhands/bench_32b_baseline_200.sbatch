#!/usr/bin/env bash
#SBATCH --job-name=baseline-32b-200
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/baseline_32b_200_%j.out

# Baseline: vanilla OpenHands + vLLM (no Pie)
# Qwen2.5-Coder-32B-Instruct (dense, 32B active) via vLLM native tool calling.
# hermes parser for Qwen2.5 family. 200 problems, temperature=0.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT=predictions/baseline_32b_200.jsonl \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=litellm+qwen2.5-coder-32b-t0 \
VLLM_EXTRA_ARGS="--max-model-len 32768" \
  bash run_litellm_baseline.sh \
    --subset-size 200 \
    --max-iterations 100 \
    --temperature 0 \
    --resume
