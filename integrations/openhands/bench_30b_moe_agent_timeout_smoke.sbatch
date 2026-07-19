#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-timeout-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_timeout_smoke_%j.out

# Smoke test for the condensation-cooldown + timeout fixes.
# Uses the same two instances that timed out in job 18391460, but with:
#   --instance-timeout-s 900  (15 min cap per instance — ensures both get attempted)
#   --idle-timeout-s 300      (5 min idle detection — catches hung generation)
#   --context-token-limit 48000

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-timeout-smoke \
OUTPUT_PREFIX=pie_agent_timeout_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id matplotlib__matplotlib-25479 \
    --instance-id sympy__sympy-23413 \
    --max-iterations 100 \
    --context-token-limit 48000 \
    --instance-timeout-s 900 \
    --idle-timeout-s 300
