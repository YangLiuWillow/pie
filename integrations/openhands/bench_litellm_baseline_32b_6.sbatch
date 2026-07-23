#!/usr/bin/env bash
#SBATCH --job-name=litellm-base-32b-6
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/litellm_base_32b_6_%j.out

# TRUE BASELINE: OpenHands + litellm (real harness, native tool calling)
# with Qwen2.5-Coder-32B served by vLLM (hermes tool-call parser). This is
# the capacity reference the coder-session integration must match — NOT
# Pattern A (openhands-agent, a different scaffold). Runs the 6 instances
# the cand12 scoring resolved. The earlier 32B litellm baseline (18599876)
# was invalidated by the /workspace prompt bug; that is now fixed
# (_format_user_prompt uses the real checkout path, swe_bench.py:538).
# --log-completions captures whether the model emits proper <tool_call>
# str_replace with old_str under vLLM native tool calling — the behavior
# coder-session's phase-1 fails to reproduce (job 18937308 v7 diagnostic).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
export LABEL=litellm+qwen2.5-coder-32b-baseline
export HARNESS=benchmarks.run_swe_bench
# The 32B weights are cached here (62G), not in the script's default hf_cache.
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface

TS=$(date +%Y%m%d_%H%M%S)
export OUTPUT=predictions/litellm_base_32b_6_${SLURM_JOB_ID}.jsonl
COMPLETIONS_DIR=logs/completions_litellm_base_32b_6_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR" logs predictions

bash run_litellm_baseline.sh \
    --instance-id django__django-14373 \
    --instance-id django__django-15368 \
    --instance-id django__django-15569 \
    --instance-id django__django-16485 \
    --instance-id django__django-16801 \
    --instance-id matplotlib__matplotlib-22719 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== predictions ==="
ls -l "$OUTPUT" 2>/dev/null || echo "(no output written)"
echo "=== completions captured ==="
ls "$COMPLETIONS_DIR" | wc -l
