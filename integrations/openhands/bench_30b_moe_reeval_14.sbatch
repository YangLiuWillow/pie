#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-reeval-14
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/reeval_14_%j.out

# Re-evaluate the 14 previously-empty-patch instances with the improved
# openhands-agent inferlet (read_file action, edit-size guardrails,
# fuzzy-match error hints, corrective nudges on edit failure).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

OUTPUT=/nfs/roberts/scratch/pi_ql324/ly337/swebench-results/pie_agent_30b_moe_reeval_14.jsonl \
BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-v2 \
OUTPUT_PREFIX=reeval_14 \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id astropy__astropy-7606 \
    --instance-id astropy__astropy-8707 \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id django__django-13128 \
    --instance-id django__django-14007 \
    --instance-id django__django-14373 \
    --instance-id django__django-15695 \
    --instance-id matplotlib__matplotlib-25479 \
    --instance-id pydata__xarray-4695 \
    --instance-id scikit-learn__scikit-learn-26194 \
    --instance-id sympy__sympy-13877 \
    --instance-id sympy__sympy-23413 \
    --instance-id sympy__sympy-24562 \
    --max-iterations 50 \
    --resume
