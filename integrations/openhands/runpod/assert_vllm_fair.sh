#!/usr/bin/env bash
# =============================================================================
# Tier-aware fairness banner assertion for a vLLM serve log.
#
# The rerun measures an OPTIMIZATION-EFFORT AXIS, not a single "fair" point.
# Three tiers (see runpod/README.md):
#   crippled    : --enforce-eager + default MoE   (reproduce the original writeup)
#   graphs-only : CUDA graphs ON, default MoE      (zero extra effort — just don't
#                                                   pass a bad flag; NO autotuner)
#   fair        : CUDA graphs ON + autotuned MoE   (one documented autotune step)
#
# This checks the banner against what the DECLARED tier promises, and only FAILS
# on a condition that tier is supposed to satisfy. Paste the output into the
# writeup — it is the mechanical, symmetric counterpart to the crippled banner
# the current writeup quotes.
#
# Usage: bash assert_vllm_fair.sh <vllm_serve.log> [tier]   (tier default: fair)
# Exit 0 = meets the tier's promise; non-zero = a promised condition is missing.
# =============================================================================
set -euo pipefail
LOG=${1:?usage: assert_vllm_fair.sh <vllm_serve.log> [crippled|graphs-only|fair]}
TIER=${2:-fair}
[ -f "$LOG" ] || { echo "no such log: $LOG"; exit 2; }

fail=0

graphs_off() { grep -qiE "Enforce eager|Cudagraph is disabled|overriding optimization level to -O0" "$LOG"; }
moe_default() { grep -qiE "Using default MoE config|Performance might be sub-optimal|Config file not found.*\.json" "$LOG"; }
apc_off() { grep -qiE "Prefix caching is disabled|prefix.caching.*disabled" "$LOG"; }

echo "  Declared tier: $TIER"

# --- CUDA graphs ---
if graphs_off; then
    if [ "$TIER" = "crippled" ]; then
        echo "  [ ok ] CUDA graphs OFF — expected for the 'crippled' tier."
    else
        echo "  [FAIL] CUDA graphs are OFF but tier '$TIER' requires them ON."
        grep -iE "Enforce eager|Cudagraph is disabled|optimization level to -O0" "$LOG" | sed 's/^/         /' | head -2
        fail=1
    fi
else
    echo "  [ ok ] CUDA graphs not disabled."
    grep -qiE "Capturing cudagraph|graph captur" "$LOG" && echo "  [ ok ] graph capture observed."
    if [ "$TIER" = "crippled" ]; then
        # Over-delivery is as corrupting as under-delivery: a 'crippled' run with
        # graphs ON understates what --enforce-eager cost, which is the whole
        # point of the crippled->graphs-only gap. Fail, don't warn.
        echo "  [FAIL] tier 'crippled' but CUDA graphs are ON — not reproducing the original baseline."
        fail=1
    fi
fi

# --- MoE kernel ---
if moe_default; then
    case "$TIER" in
        fair)
            echo "  [FAIL] Untuned/default MoE config but tier 'fair' requires a tuned one."
            echo "         -> autotune once (benchmark_moe.py --tune --tp-size 1"
            echo "            --save-dir \$TUNED_DIR), then relaunch with"
            echo "            VLLM_TUNED_CONFIG_FOLDER=\$TUNED_DIR exported."
            fail=1 ;;
        graphs-only)
            echo "  [info] Default MoE config — EXPECTED for the 'graphs-only' (no-autotuner) tier." ;;
        crippled)
            echo "  [ ok ] Default MoE config — expected for the 'crippled' tier." ;;
    esac
else
    echo "  [ ok ] Tuned MoE config in use (no default-config warning)."
    if [ "$TIER" = "crippled" ]; then
        # Symmetric to the graphs-only check below. 'crippled' is defined as
        # "--enforce-eager + DEFAULT MoE" (see header): it reproduces the
        # original writeup's baseline, which had no tuned kernel. Harmless while
        # the autotune ran last, but the arms can be collected in any order --
        # run 'crippled' after the autotune and the tuned json silently gives it
        # the good kernel, understating what --enforce-eager cost. That is the
        # crippled->graphs-only gap, so it must be a hard fail, not a warning.
        echo "  [FAIL] tier 'crippled' but a tuned MoE config is in use — this is"
        echo "         not the original baseline (autotune already run?)."
        echo "         -> unset VLLM_TUNED_CONFIG_FOLDER and relaunch."
        fail=1
    fi
    if [ "$TIER" = "graphs-only" ]; then
        # The failure mode this exists to catch: a graphs-only run performed
        # with the tuned config live silently becomes a second 'fair' run. That
        # collapses the graphs-only->fair gap to ~0 and makes "the autotuner
        # isn't worth running" look proven when it was never tested.
        echo "  [FAIL] tier 'graphs-only' but a tuned MoE config is in use — the"
        echo "         'no-autotuner' point is contaminated (autotune already run?)."
        echo "         -> unset VLLM_TUNED_CONFIG_FOLDER and relaunch."
        fail=1
    fi
fi

# --- prefix caching (required in every non-crippled tier) ---
if apc_off; then
    if [ "$TIER" = "crippled" ]; then
        echo "  [warn] Prefix caching OFF (the original baseline had it ON — keep it ON even when crippling)."
    else
        echo "  [FAIL] Prefix caching is OFF — not the intended APC baseline."
        fail=1
    fi
else
    echo "  [ ok ] Prefix caching not reported disabled."
fi

echo "  --- engine banner (for the writeup) ---"
grep -iE "vLLM API server version|Initializing.*engine|Automatic Prefix Caching|CUDA graph|MoE|attention backend|FLASH_ATTN|max_model_len|gpu_memory_utilization" "$LOG" \
    | sed 's/^/    /' | head -20 || true

if [ "$fail" -ne 0 ]; then
    echo "  RESULT: tier '$TIER' NOT satisfied — fix the flagged condition(s)."
    exit 1
fi
echo "  RESULT: banner satisfies the '$TIER' tier."
