#!/usr/bin/env bash
#SBATCH --job-name=baseline-t0-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/baseline_t0_smoke_%j.out

# Baseline (LiteLLM + vLLM, no Pie) with temperature=0 on 3 problems
# for direct comparison against Pattern A (which uses Sampler::Argmax).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=baseline+qwen3-coder-30b-a3b-t0 \
OUTPUT=predictions/baseline_t0_smoke_$(date +%Y%m%d_%H%M%S).jsonl \
  bash run_litellm_baseline.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877 \
    --temperature 0 \
    --max-iterations 100
