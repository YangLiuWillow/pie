#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-ef-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_empty_finish_smoke_%j.out

# Smoke test for the empty-finish recovery feature.
# Targets the 5 instances that produced 0-byte patches in the fileeditor_50 run.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-ef-smoke \
OUTPUT_PREFIX=pie_agent_ef_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-11740 \
    --instance-id django__django-13028 \
    --instance-id django__django-13128 \
    --instance-id matplotlib__matplotlib-25479 \
    --instance-id sympy__sympy-23413 \
    --max-iterations 100
