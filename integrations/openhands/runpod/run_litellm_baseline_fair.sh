#!/usr/bin/env bash
# =============================================================================
# FAIR litellm+vLLM baseline arm (boot fair vLLM, then run the OpenHands harness).
#
# Faithful to the upstream run_litellm_baseline.sh harness invocation, with the
# two crippling conditions removed and a fairness gate added:
#   - NO --enforce-eager  (CUDA graphs ON)
#   - refuses to run the benchmark unless assert_vllm_fair.sh passes on the banner
# Everything below the LLM boundary (backend=litellm, base_url, parser) is
# unchanged, so this stays a clean drop-in swap of the serving config only.
#
# Usage (mirrors the original):
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct OUTPUT=predictions/ab_litellm.jsonl \
#     bash run_litellm_baseline_fair.sh --instance-id django__django-12276 ... \
#          --temperature 0 --max-iterations 100
# =============================================================================
set -euo pipefail

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Harness tree = integrations/openhands (this bundle lives in runpod/ under it).
HARNESS_DIR=${HARNESS_DIR:-$(cd "$RUNPOD_DIR/.." && pwd)}
VENV=${HARNESS_VENV:-$HARNESS_DIR/.venv}
PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
LOG_DIR=$HARNESS_DIR/logs
PRED_DIR=$HARNESS_DIR/predictions
VLLM_PORT=${VLLM_PORT:-18000}
HARNESS=${HARNESS:-benchmarks.run_swe_bench}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
VLLM_TIER=${VLLM_TIER:-fair}   # crippled | graphs-only | fair (see README)
# Derive the label from the ACTUAL tier. Hardcoding "fair" here mislabels a
# crippled/graphs-only run as fair whenever this script is invoked directly
# rather than through 30_ab_run.sh (which sets LABEL/OUTPUT itself).
LABEL=${LABEL:-litellm-${VLLM_TIER}+$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')}
STRICT_FAIR=${STRICT_FAIR:-1}
EAGER_FLAG=""; [ "$VLLM_TIER" = "crippled" ] && EAGER_FLAG="--enforce-eager"

case "$MODEL" in
    *Qwen3-Coder*) TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder} ;;
    *Qwen3*)       TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml} ;;
    *)             TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-hermes} ;;
esac

# --- the serving context cap must be stated, never defaulted -------------------
# This used to be `${MAX_MODEL_LEN:-32768}` inline at the serve call. 32768 was
# the A100's value, forced by that box's 12.59 GiB of KV; every sm_90 arm has run
# 131072. The fallback is silent, and it is asymmetric in the one direction that
# corrupts the comparison: MAX_MODEL_LEN is the ONLY context limit in the vLLM
# arm (litellm has no registry entry for a self-hosted model, so nothing
# truncates client-side, and the condenser bounds history by MESSAGE COUNT, not
# tokens), while **Pie has no fixed cap at all** — its ceiling is memory-planned.
# So an unsourced env file silently gives vLLM a 4x smaller context than Pie and
# fails it on inputs Pie serves fine, which reads as an accuracy difference and
# is not one.
#
# Refuse to guess. `source /workspace/pie-bench-env.sh` sets it per GPU.
if [ -z "${MAX_MODEL_LEN:-}" ]; then
    echo "FATAL: MAX_MODEL_LEN is not set." >&2
    echo "       Run 'source /workspace/pie-bench-env.sh' first — it sets the value" >&2
    echo "       this GPU's KV budget was checked against (00_setup_h200.sh [0])." >&2
    echo "       Refusing to fall back to a hardcoded default: the vLLM arm's" >&2
    echo "       context cap is the one knob Pie has no equivalent of, so a wrong" >&2
    echo "       value silently biases the comparison instead of failing." >&2
    exit 2
fi

# --- the tuned MoE config must exist BEFORE we spend 3 minutes booting ---------
# The fairness gate (assert_vllm_fair.sh) already catches an untuned 'fair' run,
# but only from the banner, i.e. after a full model load. Whether a config will
# resolve is knowable in two seconds, so know it in two seconds.
#
# This is NOT the same on every sm_90 board, which is the trap:
#   H200 - vLLM 0.25.1 SHIPS E=128,N=768,device_name=NVIDIA_H200.json.
#          Nothing to export. START_HERE_H200.md says "don't set
#          VLLM_TUNED_CONFIG_FOLDER" and is right *for that box*.
#   H100 - vLLM ships NO bf16 E=128,N=768 config (verified 2026-07-29 against the
#          v0.25.1 tree). The 'fair' tier REQUIRES
#          VLLM_TUNED_CONFIG_FOLDER=/workspace/tuned_moe_h100 — see that folder's
#          PROVENANCE.md and VALIDATION.md, both of which must be cited with any
#          H100 fair-tier number.
if [ "$VLLM_TIER" = "fair" ]; then
    "$PIE_VENV/bin/python" - <<'PY' || exit 2
import os, sys, torch
import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
name = get_config_file_name(128, 768, None, None)
folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
cands = ([os.path.join(folder, name)] if folder else []) + \
        [os.path.join(os.path.dirname(fm.__file__), "configs", name)]
for p in cands:
    if os.path.exists(p):
        print(f"  [ ok ] tier 'fair': MoE config resolves -> {p}")
        sys.exit(0)
print(f"FATAL: tier 'fair' but no tuned MoE config for {torch.cuda.get_device_properties(0).name}.", file=sys.stderr)
print(f"       vLLM looks for: {name}", file=sys.stderr)
for p in cands:
    print(f"       tried: {p}", file=sys.stderr)
if not folder:
    print("       VLLM_TUNED_CONFIG_FOLDER is unset. On a board where vLLM ships no", file=sys.stderr)
    print("       config for this shape (e.g. H100) the fair tier needs one exported.", file=sys.stderr)
sys.exit(1)
PY
fi

mkdir -p "$LOG_DIR" "$PRED_DIR"
EXTRA_ARGS=("$@"); [ ${#EXTRA_ARGS[@]} -eq 0 ] && EXTRA_ARGS=(--subset-size 50)
TS=$(date +%Y%m%d_%H%M%S)
# summarize_ab.py selects arms by FILENAME glob, so a tier-agnostic name here
# would silently file a crippled run under the fair arm.
OUTPUT=${OUTPUT:-"$PRED_DIR/litellm_${VLLM_TIER}_${TS}.jsonl"}
VLLM_LOG="$LOG_DIR/vllm_serve_${VLLM_TIER}_${TS}.log"

echo "=== FAIR baseline: litellm + vLLM (CUDA graphs ON, tuned MoE) ==="
echo "  Model: $MODEL   Output: $OUTPUT   vLLM log: $VLLM_LOG"

# --- boot fair vLLM (NOTE: no --enforce-eager) -------------------------------
PYTHONPATH="" HF_HOME=$HF_HOME \
  "$PIE_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    $EAGER_FLAG \
    ${VLLM_EXTRA_ARGS:-} \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

echo "  waiting for vllm..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -40 "$VLLM_LOG"; exit 1; }
    sleep 2
done

# --- fairness gate: banner must match the declared tier ----------------------
if ! bash "$RUNPOD_DIR/assert_vllm_fair.sh" "$VLLM_LOG" "$VLLM_TIER"; then
    [ "$STRICT_FAIR" = "1" ] && { echo "ABORT: vLLM banner does not meet tier '$VLLM_TIER' (see above)."; exit 3; }
    echo "WARN: tier assertion failed (STRICT_FAIR=0) — continuing."
fi

# --- run the harness (identical to upstream below the LLM boundary) ----------
#
# MAX_OUTPUT_TOKENS matches PieLLM's per-call cap (pie_openhands/llm.py: "or
# 2048"). Without it the arms are not comparable and vLLM is the one that
# suffers: openhands.sdk resolves max_output_tokens from litellm's model
# registry, which has no entry for a self-hosted Qwen, so it stays None and
# generation is bounded only by max_model_len. Measured 2026-07-29: one call
# generated at ~200 tok/s for 15+ minutes and timed out the client twice,
# stalling the whole arm. The 303 s call in AGENT_HANDOVER_20260728.md §4b is
# the same bug, milder.
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-2048}
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  "$VENV/bin/python" -m "$HARNESS" \
    --backend litellm \
    --model "openai/$MODEL" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --api-key dummy \
    --output "$OUTPUT" \
    --label "$LABEL" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo "Done. Predictions: $OUTPUT   (fair vLLM banner: $VLLM_LOG)"
