#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-ef-retry
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_ef_retry_%j.out

# Retry the 2 instances that failed in job 18374000 (matplotlib timeout, sympy
# connection error). Uses a higher context_token_limit (48000 vs default 28000)
# to avoid the condensation-every-step loop that caused the matplotlib timeout.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-ef-retry \
OUTPUT_PREFIX=pie_agent_ef_retry \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id sympy__sympy-23413 \
    --max-iterations 100 \
    --context-token-limit 48000
