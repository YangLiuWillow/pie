#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-32b-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/swebench_agent_32b_smoke_%j.out

# Pattern A smoke test: same 3 problems as v2 smoke for direct comparison.
# Uses openhands-agent inferlet (full loop inside WASM) instead of
# openhands-completion (Pattern B completion adapter).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-agent+qwen2.5-coder-32b-smoke \
OUTPUT_PREFIX=pie_agent_qwen25_coder_32b_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877 \
    --max-iterations 50
