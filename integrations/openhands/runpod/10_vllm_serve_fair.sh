#!/usr/bin/env bash
# =============================================================================
# FAIR vLLM baseline launch — Qwen3-Coder-30B-A3B on a single GPU.
#
# Originally written for an A100 SXM (sm_80); has since run on H200 and H100
# (both sm_90). Nothing here is board-specific EXCEPT the tuned-MoE question,
# which is board-specific in a way that is easy to get wrong — see the note
# above the flag set.
#
# This is the direct replacement for the crippled `run_litellm_baseline.sh`
# launch that the current writeup (docs/pie-vs-litellm-writeup.md, "Fairness"
# section) flags. The two crippling flags there were:
#     --enforce-eager          # CUDA graphs OFF  -> worst-case batch-1 decode
#     (no tuned MoE config)     # untuned fused-MoE Triton kernel
#
# Both are removed / addressed here. Nothing exotic is added: every optimization
# below is a documented vLLM flag or vLLM's own shipped autotuner. That is the
# fairness rule — give vLLM the config a competent operator following vLLM's docs
# would use, no hand-written kernels on either side. See runpod/README.md.
#
# After boot this script ASSERTS the engine banner proves the fair config took
# effect (CUDA graphs captured + a real MoE config found), and refuses to serve
# otherwise — so a silently-crippled baseline can never sneak into the numbers
# again.
#
# Usage:
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct bash 10_vllm_serve_fair.sh
# Leaves vLLM serving on :$VLLM_PORT in the foreground (Ctrl-C to stop).
# Point the harness / microbench at http://localhost:$VLLM_PORT/v1 .
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
VLLM_PORT=${VLLM_PORT:-18000}
PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}          # venv with vllm installed
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
# Stated, never defaulted — see the long note in run_litellm_baseline_fair.sh.
# 32768 was the A100's forced value; every sm_90 arm runs 131072. A silent
# fallback gives vLLM a smaller context than Pie (which has no fixed cap) and
# reads as an accuracy difference rather than a config mistake.
if [ -z "${MAX_MODEL_LEN:-}" ]; then
    echo "FATAL: MAX_MODEL_LEN is not set — run 'source /workspace/pie-bench-env.sh' first." >&2
    exit 2
fi
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
# Optimization-effort tier — the axis the rerun measures. See runpod/README.md.
#   crippled    : --enforce-eager + default MoE (reproduce the original writeup)
#   graphs-only : CUDA graphs ON, NO MoE autotuner (zero extra effort)
#   fair        : CUDA graphs ON + autotuned MoE  (one documented autotune step)
VLLM_TIER=${VLLM_TIER:-fair}
# Strictness: if the banner does not meet the declared tier, abort (1) vs warn (0).
STRICT_FAIR=${STRICT_FAIR:-1}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# The only flag the tier toggles at launch is --enforce-eager. The MoE-tuned vs
# default distinction is NOT a flag — it is whether a tuned json exists in vLLM's
# config dir (run / skip 11_autotune_moe.sh). assert_vllm_fair.sh checks the
# banner matches the declared tier either way.
EAGER_FLAG=""
if [ "$VLLM_TIER" = "crippled" ]; then
    EAGER_FLAG="--enforce-eager"
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
VLLM_LOG="$LOG_DIR/vllm_serve_fair_${TS}.log"

echo "=== vLLM baseline ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)) — tier: $VLLM_TIER ==="
echo "  Model:     $MODEL"
echo "  Port:      $VLLM_PORT"
echo "  GPU mem:   $GPU_MEM_UTIL   max_model_len: $MAX_MODEL_LEN"
echo "  Log:       $VLLM_LOG"
echo "  eager:     ${EAGER_FLAG:-<none> (CUDA graphs ON)}"
echo ""

# -----------------------------------------------------------------------------
# THE FAIR FLAG SET.  Deltas vs the crippled baseline are marked [FIX].
#   --enable-prefix-caching   APC on, the honest match for Pie's KV reuse.
#   (no --enforce-eager)      [FIX] CUDA graphs capture -> kills batch-1 launch
#                             overhead, the single biggest decode win in vLLM.
#   --generation-config vllm  use vLLM's own sampling defaults (unchanged).
#   Chunked prefill / V1 are default-on in current vLLM; not forced so we get
#   vLLM's own recommended defaults rather than second-guessing them.
# A tuned fused-MoE config is NOT a flag — vLLM auto-loads it from
# VLLM_TUNED_CONFIG_FOLDER first, then its own packaged config dir, IF a file
# matching this GPU + expert shape exists (fused_moe.py:1075-1109).
#
# WHETHER ONE SHIPS IS PER-BOARD, and this is the trap. For E=128,N=768 bf16,
# vLLM 0.25.1 ships H200 / B200 / H20 / MI308X and **not A100, not H100**
# (verified against the v0.25.1 tree, 2026-07-29). So:
#   H200 - nothing to do; do NOT set VLLM_TUNED_CONFIG_FOLDER.
#   H100 - the 'fair' tier REQUIRES VLLM_TUNED_CONFIG_FOLDER=/workspace/tuned_moe_h100
#          (borrowed config; see that folder's PROVENANCE.md + VALIDATION.md).
#   A100 - same situation; /workspace/tuned_moe, see runpod/tuned_moe/.
# If the banner says "Using default MoE config" on a board with no shipped
# config, exporting the folder is the fix — NOT 11_autotune_moe.sh, which was
# abandoned at a 15-24 h projection with no partial-progress artifact.
# -----------------------------------------------------------------------------
PYTHONPATH="" HF_HOME="$HF_HOME" \
  "$PIE_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    $EAGER_FLAG \
    $VLLM_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "Stopping vllm (PID $VLLM_PID)"; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

echo "  Waiting for vllm to be ready (PID $VLLM_PID)..."
READY=0
for i in $(seq 1 1800); do
    if curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q "200"; then
        echo "  vllm is up ($((i * 2))s)"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vllm died during startup. Check $VLLM_LOG"; tail -40 "$VLLM_LOG"; exit 1
    fi
    sleep 2
done
[ "$READY" -eq 1 ] || { echo "ERROR: vllm not ready in time. Check $VLLM_LOG"; tail -40 "$VLLM_LOG"; exit 1; }

# --- Prove the fair config actually took effect ------------------------------
echo ""
echo "=== Fairness banner check ($VLLM_LOG) ==="
bash "$SCRIPT_DIR/assert_vllm_fair.sh" "$VLLM_LOG" "$VLLM_TIER" || {
    rc=$?
    if [ "$STRICT_FAIR" = "1" ]; then
        echo "ABORT: banner does not meet the declared tier '$VLLM_TIER' (STRICT_FAIR=1)."
        echo "       For tier 'fair' with a default-MoE warning: run 11_autotune_moe.sh, relaunch."
        exit "$rc"
    fi
    echo "WARN: tier assertion failed but STRICT_FAIR=0 — continuing."
}

echo ""
echo "vLLM serving (tier=$VLLM_TIER) on http://localhost:$VLLM_PORT/v1 — Ctrl-C to stop."
wait "$VLLM_PID"
