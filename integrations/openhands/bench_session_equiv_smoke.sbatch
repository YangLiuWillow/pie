#!/usr/bin/env bash
#SBATCH --job-name=session-equiv
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/session_equiv_%j.out

# Phase 2 of docs/OPENHANDS_CODER_SESSION_DESIGN.md — 5-instance equivalence
# run at temperature 0:
#
#   arm A: stateless openhands-completion   (Phase-0 baseline transport)
#   arm B: openhands-coder-session          (KV sessions, kv-verify ON)
#
# Same stock OpenHands agent, same model, same sampling. Expected outcome:
# identical patches/iteration counts per instance, with arm B prefilling a
# small fraction of the prompt tokens (see the pie_session metadata block),
# and zero kv-verify errors.
#
# Model note: must be a model whose chat template matches Pie's qwen ChatML
# JSON tool-call format. Qwen3-Coder does NOT qualify — its template uses the
# XML <function=...><parameter=...> format, which Pie's Instruct/tool decoder
# doesn't speak yet, so the agent never registers a tool call (job 18709682
# produced empty patches in both arms this way). Qwen2.5-Coder does qualify.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

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
LABEL=pie-stateless+qwen2.5-coder-7b-t0 \
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
LABEL=pie-session+qwen2.5-coder-7b-t0 \
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
