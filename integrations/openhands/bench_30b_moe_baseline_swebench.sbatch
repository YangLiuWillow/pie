#!/usr/bin/env bash
#SBATCH --job-name=baseline-30b-moe
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/baseline_30b_moe_swebench_%j.out

# Baseline: vanilla OpenHands + vLLM (no Pie) with same model as the agent run.
# Qwen3-Coder-30B-A3B-Instruct via vLLM's native tool calling (qwen3_coder parser).
# Fresh output file (not --resume) since prior run's predictions are all 0-byte.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

OUTPUT=predictions/baseline_qwen3_coder_30b_moe_v2_50.jsonl \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=litellm+qwen3-coder-30b-a3b \
  bash run_litellm_baseline.sh \
    --subset-size 50 \
    --max-iterations 50
