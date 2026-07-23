#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-32b-v5
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_32b_v5_%j.out

# Validation run for the anti-degeneration guards (commit d094ed69) on top
# of two-phase tool calling: <|im_start|> stop token, special-token
# sanitization of replayed history, duplicate-call dedupe. Same instance
# as v2-v4 — success = clean single-turn responses (no leaked role tags,
# no token salad), parsed tool calls, and a non-empty patch.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_32b_v5

COMPLETIONS_DIR=logs/completions_debug_v5_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-session+qwen2.5-coder-32b-t0-trajdebug-v5 \
OUTPUT=predictions/traj_debug_32b_v5_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== completions captured ==="
ls "$COMPLETIONS_DIR" | head
