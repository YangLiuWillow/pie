#!/usr/bin/env bash
#SBATCH --job-name=pie-sweb32b-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/swebench_32b_v2_smoke_%j.out

# Quick 3-problem smoke test for v2 harness improvements before full eval.
# Uses the same problems that failed in earlier sessions for direct comparison.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie+qwen2.5-coder-32b-v2-smoke \
OUTPUT_PREFIX=pie_qwen25_coder_32b_v2_smoke \
REQUEST_TIMEOUT_S=1800 \
  bash run_pie_backend.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877 \
    --max-iterations 100 \
    --max-fake-responses 10
