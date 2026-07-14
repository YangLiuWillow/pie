#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-30b-fe-50
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_fileeditor_50_%j.out

# Pattern A with FileEditor wrapping on the same 50 SWE-Bench instances
# that the baseline scored 20/50 on, for direct comparison.
# Changes: OpenHands FileEditor (whitespace fallback, view_range, insert,
# undo_edit), detect_stuck wired into agent loop.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-fileeditor \
OUTPUT_PREFIX=pie_agent_30b_moe_fileeditor_50 \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id astropy__astropy-7606 \
    --instance-id astropy__astropy-8707 \
    --instance-id django__django-7530 \
    --instance-id django__django-11333 \
    --instance-id django__django-11400 \
    --instance-id django__django-11532 \
    --instance-id django__django-11740 \
    --instance-id django__django-12262 \
    --instance-id django__django-12276 \
    --instance-id django__django-13028 \
    --instance-id django__django-13089 \
    --instance-id django__django-13128 \
    --instance-id django__django-13410 \
    --instance-id django__django-13590 \
    --instance-id django__django-13807 \
    --instance-id django__django-14007 \
    --instance-id django__django-14140 \
    --instance-id django__django-14373 \
    --instance-id django__django-14559 \
    --instance-id django__django-15368 \
    --instance-id django__django-15563 \
    --instance-id django__django-15569 \
    --instance-id django__django-15695 \
    --instance-id django__django-16485 \
    --instance-id django__django-16801 \
    --instance-id django__django-17084 \
    --instance-id matplotlib__matplotlib-22719 \
    --instance-id matplotlib__matplotlib-24149 \
    --instance-id matplotlib__matplotlib-25479 \
    --instance-id matplotlib__matplotlib-26208 \
    --instance-id pydata__xarray-4075 \
    --instance-id pydata__xarray-4695 \
    --instance-id pydata__xarray-4966 \
    --instance-id pylint-dev__pylint-4661 \
    --instance-id pytest-dev__pytest-7236 \
    --instance-id scikit-learn__scikit-learn-10908 \
    --instance-id scikit-learn__scikit-learn-12973 \
    --instance-id scikit-learn__scikit-learn-13496 \
    --instance-id scikit-learn__scikit-learn-14983 \
    --instance-id scikit-learn__scikit-learn-25973 \
    --instance-id scikit-learn__scikit-learn-26194 \
    --instance-id sphinx-doc__sphinx-7454 \
    --instance-id sphinx-doc__sphinx-7748 \
    --instance-id sphinx-doc__sphinx-9229 \
    --instance-id sphinx-doc__sphinx-11510 \
    --instance-id sympy__sympy-13877 \
    --instance-id sympy__sympy-16450 \
    --instance-id sympy__sympy-19346 \
    --instance-id sympy__sympy-23413 \
    --instance-id sympy__sympy-24562 \
    --max-iterations 100
