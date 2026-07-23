#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-32b-v7
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_32b_v7_%j.out

# v7: DIAGNOSTIC re-run of v6 (django-14373) with phase-1/phase-2
# instrumentation. The coder-session wasm now returns debug_full_text
# (phase-1 raw generation), debug_phase1_marker (did it contain a literal
# <tool_call> block), debug_phase2_fired (did the forced-call fallback
# run); pie_openhands/llm.py appends these to $PIE_DEBUG_LOG as JSONL.
# GOAL: determine whether phase 1 emits a valid multi-line str_replace
# <tool_call> that the strict serde_json decoder (qwen3.rs:615) silently
# drops — vs. genuinely narrating without acting. This decides the fix
# (newline-lenient decoder vs. phase-2/schema change). No fix applied yet.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_32b_v7

COMPLETIONS_DIR=logs/completions_debug_v7_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"
export PIE_DEBUG_LOG=logs/pie_debug_v7_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-session+qwen2.5-coder-32b-t0-trajdebug-v7 \
OUTPUT=predictions/traj_debug_32b_v7_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-14373 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== phase debug log ==="
wc -l "$PIE_DEBUG_LOG" 2>/dev/null || echo "(no debug log written)"
