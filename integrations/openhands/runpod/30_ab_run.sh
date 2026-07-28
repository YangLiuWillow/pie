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

# This script does NOT source /workspace/pie-bench-env.sh — callers are expected
# to have done it (40_concurrency_sweep.sh and 41_moe_decode_sweep.sh do). Forget
# it and HF_HOME falls back to $HOME/.cache/huggingface, which on a fresh pod is
# empty, and the failure is a 57 GB re-download announced as
#   "hf: … not in local cache; downloading runtime artifacts only"
# several minutes after launch, from a log you are not tailing. Fail here instead.
_snap_root="$HF_HOME/hub/models--${MODEL//\//--}/snapshots"
if [ ! -d "$_snap_root" ] || [ -z "$(ls -A "$_snap_root" 2>/dev/null)" ]; then
    echo "FATAL: $MODEL is not in HF_HOME=$HF_HOME" >&2
    echo "       Run 'source /workspace/pie-bench-env.sh' first (it sets HF_HOME" >&2
    echo "       to the /workspace copy). Refusing to trigger a 57 GB download." >&2
    exit 2
fi
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

# Narrow the set without editing the array. Comma- or space-separated ids:
#   AB_INSTANCES=django__django-14373 ARM=pie bash 30_ab_run.sh
# Changing the instance SET is on the escalate list (AGENT_HANDOVER_H200.md §8);
# this switch exists so a config sweep can run one instance without touching the
# 13-instance definition that the collected arms used.
if [ -n "${AB_INSTANCES:-}" ]; then
    INSTANCES=()
    for _id in ${AB_INSTANCES//,/ }; do
        INSTANCES+=(--instance-id "$_id")
    done
    # Each id contributes TWO array elements (--instance-id, <id>), so report
    # the id count rather than ${#INSTANCES[@]}, which reads as double.
    echo "=== instance set overridden: $((${#INSTANCES[@]} / 2)) instance(s) — ${AB_INSTANCES} ==="
fi

# How many instances to solve in parallel against ONE server. 1 = the serial
# behaviour every arm before 2026-07-28 used; >1 is the throughput lever.
#
# The harness already implements this (`swe_bench.py:942`, a ThreadPoolExecutor
# whose workers each call `build_llm`, so every conversation gets its own PieLLM
# — and, in daemon mode, its own inferlet process and websocket). Nothing here
# had a way to reach it, which is the only reason the concurrency question was
# still open.
#
# The metric this exists to produce is instances/GPU-hour, NOT s/iter: at N>1
# the per-iteration number is contended by construction and stops meaning what
# it means at N=1. Compare a concurrent arm against the SAME arm at N=1.
CONCURRENCY=${CONCURRENCY:-1}
if [ "$CONCURRENCY" -lt 1 ] 2>/dev/null; then
    echo "CONCURRENCY must be a positive integer" >&2; exit 2
fi

echo "=== concurrency: $CONCURRENCY instance(s) in parallel against one server ==="

cd "$HARNESS_DIR"

if [ "$ARM" = "litellm" ]; then
    # Optimization-effort tier for the vLLM baseline: crippled | graphs-only | fair.
    # Run all three for the effort-axis curve; each writes a distinct output.
    export VLLM_TIER=${VLLM_TIER:-fair}
    export OUTPUT=${OUTPUT:-predictions/ab_h200_litellm_${VLLM_TIER}_c${CONCURRENCY}_${TS}.jsonl}
    export LABEL=litellm-${VLLM_TIER}+qwen3-coder-30b-a3b-t0-c${CONCURRENCY}
    bash "$RUNPOD_DIR/run_litellm_baseline_fair.sh" \
        "${INSTANCES[@]}" --temperature 0 --max-iterations 100 \
        --concurrency "$CONCURRENCY"

elif [ "$ARM" = "pie" ]; then
    # Uses the existing run_pie_backend.sh, pointed at the A100 native config.
    export PIE_BIN=${PIE_BIN:-$(cd "$HARNESS_DIR/../.." && pwd)/target/release/pie}
    export PIE_PORT=${PIE_PORT:-18097}
    export BACKEND=pie-session
    export KV_VERIFY=1
    # Which Pie memory/attention configuration to run. The three differ ONLY in
    # the planner profile and the KV page size, which is what decides whether
    # xqa_decode is on. See the config files' own headers.
    #
    #   auto_p32     CONFIG A — auto profile, page forced to 32, xqa ON
    #                (the shape the 2026-07-27 H200 arm ran)
    #   latency_p32  CONFIG B — latency profile, page forced to 32, xqa ON
    #   auto_p16     CONFIG C — auto profile, page 16, xqa OFF (planner's choice)
    #
    # PIE_CUDA_KV_PAGE_SIZE is the ONLY control over page size — the toml key is
    # inert on the planner path (cuda_memory_planner.cpp:155-172), so C works by
    # UNSETTING the variable that /workspace/pie-bench-env.sh exports. Setting
    # it here rather than relying on the sourced env also means each variant is
    # self-contained and cannot inherit the previous one's value.
    PIE_CFG_VARIANT=${PIE_CFG_VARIANT:-auto_p32}
    case "$PIE_CFG_VARIANT" in
        auto_p32)
            _cfg=pie_cuda_native_config_30b_moe_h200.toml
            export PIE_CUDA_KV_PAGE_SIZE=32 ;;
        latency_p32)
            _cfg=pie_cuda_native_config_30b_moe_h200_latency.toml
            export PIE_CUDA_KV_PAGE_SIZE=32 ;;
        auto_p16)
            _cfg=pie_cuda_native_config_30b_moe_h200_auto_p16.toml
            unset PIE_CUDA_KV_PAGE_SIZE ;;
        *)
            echo "PIE_CFG_VARIANT must be auto_p32 | latency_p32 | auto_p16" >&2
            exit 2 ;;
    esac
    export CFG=${CFG:-$RUNPOD_DIR/$_cfg}
    export LABEL=${LABEL:-pie-cuda-native-h200-${PIE_CFG_VARIANT}+qwen3-coder-30b-t0-c${CONCURRENCY}}
    # The variant is in the FILENAME because summarize_ab.py selects arms by
    # glob. Summarize these with explicit per-variant globs — a bare
    # `*_pie_*.jsonl` would merge all three into one meaningless average.
    export OUTPUT=${OUTPUT:-predictions/ab_h200_pie_${PIE_CFG_VARIANT}_c${CONCURRENCY}_${TS}.jsonl}
    echo "=== pie config variant: $PIE_CFG_VARIANT ($_cfg)"
    echo "===   PIE_CUDA_KV_PAGE_SIZE=${PIE_CUDA_KV_PAGE_SIZE:-<unset — planner picks, expect 16>}"
    export REQUEST_TIMEOUT_S=900
    bash "$HARNESS_DIR/run_pie_backend.sh" \
        "${INSTANCES[@]}" --python-tool-parser --temperature 0 --max-iterations 100 \
        --concurrency "$CONCURRENCY"
else
    echo "ARM must be litellm or pie"; exit 2
fi

echo ""
echo "=== $ARM arm done. Compute wall/iteration with: python $RUNPOD_DIR/summarize_ab.py <pie.jsonl> <litellm.jsonl> ==="
