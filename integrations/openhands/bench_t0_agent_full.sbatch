#!/usr/bin/env bash
#SBATCH --job-name=agent-t0-full
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_t0_full_%j.out

# Pattern A full 50-problem SWE-Bench eval at temperature=0.
# Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, 3B active) on 1x RTX Pro 6000 (96GB).
# Inferlet uses Sampler::Argmax (greedy). --temperature 0 passed for consistency.
# --resume allows preempted jobs to continue from where they left off.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

OUTPUT=predictions/pie_agent_t0_full_50.jsonl \
BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b-t0 \
OUTPUT_PREFIX=pie_agent_t0_full \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --subset-size 50 \
    --max-iterations 100 \
    --temperature 0 \
    --resume
