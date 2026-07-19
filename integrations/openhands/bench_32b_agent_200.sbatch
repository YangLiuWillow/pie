#!/usr/bin/env bash
#SBATCH --job-name=agent-32b-200
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_32b_200_%j.out

# Pattern A agent with Qwen2.5-Coder-32B-Instruct (dense, 32B active)
# on 1x RTX Pro 6000 Blackwell (96GB). 200 problems, temperature=0.
# --resume allows preempted jobs to continue from where they left off.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT=predictions/pie_agent_32b_200.jsonl \
BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-agent+qwen2.5-coder-32b-t0 \
OUTPUT_PREFIX=pie_agent_32b_200 \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --subset-size 200 \
    --max-iterations 100 \
    --temperature 0 \
    --resume
