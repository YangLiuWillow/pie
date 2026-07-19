#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-rootcause
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_rootcause_%j.out

# Test the root-cause-analysis + test-verification improvements on the two
# instances that previously produced incorrect patches.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-rootcause \
OUTPUT_PREFIX=pie_agent_rootcause \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id matplotlib__matplotlib-25479 \
    --instance-id sympy__sympy-23413 \
    --max-iterations 100 \
    --context-token-limit 48000
