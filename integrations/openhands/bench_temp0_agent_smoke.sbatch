#!/usr/bin/env bash
#SBATCH --job-name=agent-t0-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_t0_smoke_%j.out

# Pattern A (Pie agent inferlet, Sampler::Argmax = greedy) on same 3 problems
# for direct comparison against baseline at temperature=0.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-t0 \
OUTPUT_PREFIX=pie_agent_t0_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877 \
    --max-iterations 100
