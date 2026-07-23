#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-32b-v6
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_32b_v6_%j.out

# v6: validate the full guard + two-phase stack (commits d094ed69 +
# 5b773c55) on an instance Qwen2.5-Coder-32B ACTUALLY solves. v2-v5 used
# django-13028, which the 32B model never solves (it hallucinates a
# nonexistent adminpricing app), so a 0-byte patch there couldn't
# distinguish machinery bugs from model incapacity. django-14373 was
# confirmed resolved=True under Pattern A by the cand12 scoring run
# (job 18905080): a clean one-line fix in django/utils/dateformat.py
# (DateFormat.Y -> f'{self.data.year:04d}'). Success = clean single-turn
# responses (no leaked role tags / token salad), parsed tool calls, and a
# NON-EMPTY, correct patch produced end-to-end through the pie session.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_32b_v6

COMPLETIONS_DIR=logs/completions_debug_v6_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-session+qwen2.5-coder-32b-t0-trajdebug-v6 \
OUTPUT=predictions/traj_debug_32b_v6_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-14373 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== completions captured ==="
ls "$COMPLETIONS_DIR" | head
