#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-v2-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_v2_smoke_%j.out

# Smoke test for v2 inferlet changes:
#   - Greedy decoding (Sampler::Argmax)
#   - Persistent bash sessions (cwd/env preserved across steps)
#   - Stronger stuck detection (repeated bash, alternating patterns)
#   - Directory listing in read_file
#   - Inferlet step logging at INFO level
# Same 3 problems as prior smoke tests for direct comparison.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-v2-smoke \
OUTPUT_PREFIX=pie_agent_v2_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877 \
    --max-iterations 100
