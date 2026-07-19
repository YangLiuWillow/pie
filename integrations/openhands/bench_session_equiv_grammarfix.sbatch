#!/usr/bin/env bash
#SBATCH --job-name=session-equiv-fix
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/session_equiv_%j.out

# Same 5-instance, 2-arm t=0 equivalence run as bench_session_equiv_smoke.sbatch,
# but served by the grammar-fix binary (schema-constrained tool-call arguments
# + whitespace tolerance, 2026-07-18). The live target/release/pie is left
# untouched while job 18690721 is still running. Expected outcome now that the
# tool-call grammar no longer degenerates: NON-empty patches, byte-equal
# across arms, 0 kv-verify errors, ~7-11x prefill cut in arm B.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
# Distinct port: this job may share a node with other pie servers (18777817
# died on "Address already in use" when co-scheduled with job 18690721).
export PIE_PORT=18091
# Pin the workspace path so both arms see byte-identical prompts (the path
# leaks into prompts via tool output; a randomized tempdir makes t=0
# trajectories diverge across arms by construction).
export SWEB_WS_ROOT=/tmp/sweb_equiv_ws

TS=$(date +%Y%m%d_%H%M%S)
STATELESS_OUT=predictions/session_equiv_stateless_${TS}.jsonl
SESSION_OUT=predictions/session_equiv_session_${TS}.jsonl

INSTANCES=(
    --instance-id django__django-11333
    --instance-id django__django-11532
    --instance-id django__django-13028
    --instance-id scikit-learn__scikit-learn-25973
    --instance-id sympy__sympy-13877
)

echo "=== arm A: stateless (openhands-completion) ==="
BACKEND=pie \
CFG=tests/fixtures/pie_cuda_vllm_config.toml \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
LABEL=pie-stateless+qwen2.5-coder-7b-t0-grammarfix \
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
CFG=tests/fixtures/pie_cuda_vllm_config.toml \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
LABEL=pie-session+qwen2.5-coder-7b-t0-grammarfix \
OUTPUT="$SESSION_OUT" \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    "${INSTANCES[@]}" \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== equivalence comparison ==="
# The comparison's exit code is informational (1 = trajectory divergence);
# don't let set -e turn a completed two-arm run into a FAILED job.
PYTHONPATH="" .venv/bin/python -m benchmarks.compare_equivalence \
    "$STATELESS_OUT" "$SESSION_OUT" || echo "comparison exit=$?"
