#!/usr/bin/env bash
#SBATCH --job-name=session-equiv-32b
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/session_equiv_%j.out

# 5-instance, 2-arm t=0 equivalence run on Qwen2.5-Coder-32B (grammar-fix
# binary). The 7B runs (18784455) validated the transport end-to-end but the
# model itself cannot solve these instances at t=0 (fixates on paths from the
# issue text — traj job 18788787), so every patch was legitimately empty.
# 32B produces real patches, giving the equivalence comparison teeth:
# expect NON-empty byte-equal patches, 0 kv-verify errors, big prefill cut.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18093
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
# Pin the workspace path so both arms see byte-identical prompts.
export SWEB_WS_ROOT=/tmp/sweb_equiv_ws_32b

TS=$(date +%Y%m%d_%H%M%S)
STATELESS_OUT=predictions/session_equiv32b_stateless_${TS}.jsonl
SESSION_OUT=predictions/session_equiv32b_session_${TS}.jsonl

INSTANCES=(
    --instance-id django__django-11333
    --instance-id django__django-11532
    --instance-id django__django-13028
    --instance-id scikit-learn__scikit-learn-25973
    --instance-id sympy__sympy-13877
)

echo "=== arm A: stateless (openhands-completion) ==="
BACKEND=pie \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-stateless+qwen2.5-coder-32b-t0-grammarfix \
OUTPUT="$STATELESS_OUT" \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    "${INSTANCES[@]}" \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== arm B: session (openhands-coder-session, kv-verify) ==="
BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie-session+qwen2.5-coder-32b-t0-grammarfix \
OUTPUT="$SESSION_OUT" \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    "${INSTANCES[@]}" \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== equivalence comparison ==="
PYTHONPATH="" .venv/bin/python -m benchmarks.compare_equivalence \
    "$STATELESS_OUT" "$SESSION_OUT" || echo "comparison exit=$?"
