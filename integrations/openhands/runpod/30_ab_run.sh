#!/usr/bin/env bash
# =============================================================================
# Wall-clock A/B on the A100, both arms, same instances, temperature 0.
#   - baseline arm: run_litellm_baseline_fair.sh  (FAIR vLLM — CUDA graphs + tuned MoE)
#   - pie arm:      run_pie_backend.sh             (native CUDA driver, A100 config)
# Reuses the existing harness scripts so the invocation matches prior runs; only
# the serving config differs. Report BOTH total wall time AND wall/iteration
# (the divergence-robust number the writeup relies on).
#
# Run each arm in its own process with a FRESH server (never both engines
# resident at once — 60 GB weights each won't co-reside on 80 GB anyway).
#   ARM=litellm bash 30_ab_run.sh
#   ARM=pie     bash 30_ab_run.sh
# =============================================================================
set -euo pipefail

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR=${HARNESS_DIR:-$(cd "$RUNPOD_DIR/.." && pwd)}
ARM=${ARM:?set ARM=litellm or ARM=pie}
export MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}
TS=$(date +%Y%m%d_%H%M%S)

# The 13-instance set from the writeup (baseline's own prior wins — ceiling is
# parity for accuracy, but the timing/per-iteration signal is clean). Swap in
# the neutral-50 ids for a set whose accuracy can move both ways.
INSTANCES=(
  --instance-id django__django-12276  --instance-id django__django-13028
  --instance-id django__django-13089  --instance-id django__django-14373
  --instance-id django__django-15569  --instance-id django__django-16485
  --instance-id matplotlib__matplotlib-22719 --instance-id pydata__xarray-4075
  --instance-id pydata__xarray-4966   --instance-id scikit-learn__scikit-learn-10908
  --instance-id scikit-learn__scikit-learn-12973 --instance-id scikit-learn__scikit-learn-13496
  --instance-id sympy__sympy-19346
)

cd "$HARNESS_DIR"

if [ "$ARM" = "litellm" ]; then
    # Optimization-effort tier for the vLLM baseline: crippled | graphs-only | fair.
    # Run all three for the effort-axis curve; each writes a distinct output.
    export VLLM_TIER=${VLLM_TIER:-fair}
    export OUTPUT=predictions/ab_a100_litellm_${VLLM_TIER}_${TS}.jsonl
    export LABEL=litellm-${VLLM_TIER}+qwen3-coder-30b-a3b-t0
    bash "$RUNPOD_DIR/run_litellm_baseline_fair.sh" \
        "${INSTANCES[@]}" --temperature 0 --max-iterations 100

elif [ "$ARM" = "pie" ]; then
    # Uses the existing run_pie_backend.sh, pointed at the A100 native config.
    export PIE_BIN=${PIE_BIN:-$(cd "$HARNESS_DIR/../.." && pwd)/target/release/pie}
    export PIE_PORT=${PIE_PORT:-18097}
    export BACKEND=pie-session
    export KV_VERIFY=1
    export CFG=$RUNPOD_DIR/pie_cuda_native_config_30b_moe_a100.toml
    export LABEL=pie-cuda-native-a100+qwen3-coder-30b-t0
    export OUTPUT=predictions/ab_a100_pie_${TS}.jsonl
    export REQUEST_TIMEOUT_S=900
    bash "$HARNESS_DIR/run_pie_backend.sh" \
        "${INSTANCES[@]}" --python-tool-parser --temperature 0 --max-iterations 100
else
    echo "ARM must be litellm or pie"; exit 2
fi

echo ""
echo "=== $ARM arm done. Compute wall/iteration with: python $RUNPOD_DIR/summarize_ab.py <pie.jsonl> <litellm.jsonl> ==="
