#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-30b-fe-rerun5
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_fileeditor_rerun5_%j.out

# Re-run the 5 instances the baseline solves but PIE+FileEditor missed.
# With logging fixes: root logger level preserved across FileEditor import,
# inferlet step lines logged at INFO, inferlet result dict logged.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-fileeditor-rerun5 \
OUTPUT_PREFIX=pie_agent_30b_moe_fileeditor_rerun5 \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-11740 \
    --instance-id django__django-13028 \
    --instance-id django__django-14007 \
    --instance-id scikit-learn__scikit-learn-25973 \
    --instance-id sympy__sympy-16450 \
    --max-iterations 100
