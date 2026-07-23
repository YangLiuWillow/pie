#!/usr/bin/env bash
#SBATCH --job-name=aligned-stateless-sub4
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/aligned_stateless_sub4_%j.out

# Acceleration eval — STATELESS control. SAME coder-session inferlet, same
# aligned vLLM config, same Option B parsing, but STATELESS=1 omits
# --pie-session so the inferlet rebuilds (full re-prefill) every call. Pie's
# driver has NO vLLM prefix caching, so this genuinely re-prefills the whole
# context each turn — the true "no acceleration" reference. Longer walltime
# expected (that's the point). Pair with bench_aligned_session_sub4.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_aligned_stateless_sub4

COMPLETIONS_DIR=logs/completions_aligned_stateless_sub4_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"
export PIE_DEBUG_LOG=logs/pie_debug_aligned_stateless_sub4_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
STATELESS=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe_aligned.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-stateless+qwen3-coder-30b-t0-aligned-stateless \
OUTPUT=predictions/aligned_stateless_sub4_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-15569 \
    --instance-id django__django-14373 \
    --instance-id pydata__xarray-4075 \
    --instance-id scikit-learn__scikit-learn-10908 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== stateless summary ==="
grep -E 'pie session —|→ ok' "logs/aligned_stateless_sub4_${SLURM_JOB_ID}.out" 2>/dev/null || true
