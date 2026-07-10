#!/usr/bin/env bash
#SBATCH --job-name=pie-swebench-32b-v2
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/swebench_32b_v2_%j.out

# SWE-Bench Verified eval with Qwen2.5-Coder-32B-Instruct on H200
# Key improvements over v1:
#   - Fake-user-response loop (agent doesn't give up on content-only turns)
#   - LLMSummarizingCondenser (prevents context degradation on long runs)
#   - Structured 8-phase prompt (matching official OpenHands evaluation)
#   - max_iterations=100 (was 50)

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie+qwen2.5-coder-32b-v2 \
OUTPUT_PREFIX=pie_qwen25_coder_32b_v2 \
REQUEST_TIMEOUT_S=1800 \
  bash run_pie_backend.sh \
    --subset-size 50 \
    --max-iterations 100 \
    --max-fake-responses 10
