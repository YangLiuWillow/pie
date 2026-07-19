#!/usr/bin/env bash
#SBATCH --job-name=traj-debug
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_%j.out

# Diagnose 0-byte patches persisting after the tool-call grammar fix
# (equivalence run 18778703: tool calls now work, arms patch-equal, but no
# instance produced any edit). Runs ONE instance with --log-completions to
# capture the full request/response trajectory.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18092

COMPLETIONS_DIR=logs/completions_debug_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"

BACKEND=pie \
CFG=tests/fixtures/pie_cuda_vllm_config.toml \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
LABEL=pie-stateless+qwen2.5-coder-7b-t0-trajdebug \
OUTPUT=predictions/traj_debug_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== completions captured ==="
ls "$COMPLETIONS_DIR" | head
