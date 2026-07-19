#!/usr/bin/env bash
#SBATCH --job-name=baseline-t0-full
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/baseline_t0_full_%j.out

# Baseline full 50-problem SWE-Bench eval at temperature=0.
# Vanilla OpenHands + vLLM (no Pie) with same model as agent run.
# Qwen3-Coder-30B-A3B-Instruct via vLLM native tool calling (qwen3_coder parser).
# --max-model-len 65536 matches the agent config (prior baseline used 32768).
# --resume allows preempted jobs to continue from where they left off.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

OUTPUT=predictions/baseline_t0_full_50.jsonl \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=litellm+qwen3-coder-30b-a3b-t0 \
VLLM_EXTRA_ARGS="--max-model-len 65536" \
  bash run_litellm_baseline.sh \
    --subset-size 50 \
    --max-iterations 100 \
    --temperature 0 \
    --resume
