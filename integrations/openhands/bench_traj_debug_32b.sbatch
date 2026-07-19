#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-32b
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_32b_%j.out

# Diagnose 0-byte patches on 32B (equivalence run 18793749: all 10 completed
# runs empty despite 17-35 LLM calls each; litellm baseline_32b_200 was ALSO
# 0/93 non-empty, so the stock-agent path fails independent of backend).
# Runs ONE instance with --log-completions to capture the full trajectory.
# django__django-13028 chosen: 31 calls, no stuck-retries, still 0-byte.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18094
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_32b

COMPLETIONS_DIR=logs/completions_debug_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-session+qwen2.5-coder-32b-t0-trajdebug \
OUTPUT=predictions/traj_debug_32b_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== completions captured ==="
ls "$COMPLETIONS_DIR" | head
