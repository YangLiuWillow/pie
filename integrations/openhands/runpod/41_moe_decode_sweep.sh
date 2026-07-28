#!/usr/bin/env bash
# =============================================================================
# MoE decode kernel sweep.
#
# qwen3_5_moe_forward.cpp picks one of three decode paths off `routes = N*top_k`
# (top_k=8 for Qwen3-Coder-30B-A3B, so routes = 8 x batch):
#
#   aligned decode   routes >= PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES (64)
#   wmma decode      PIE_QWEN35_MOE_WMMA_DECODE=1                       (off)
#   cublas M=1       fallback                                           <-- ours
#
# At batch 1 routes=8, and at the concurrency the SWE-bench arms actually reach
# (mean R=2.63 -> routes~21) it is still 8x short of 64. So every decode measured
# in this study ran the cuBLAS batched-GEMV fallback, which is cuBLAS's worst
# shape. Both faster kernels are eligible on this model's dims (H=2048, Im=768,
# both %16==0). This sweep measures whether either is actually faster here.
#
# Metric is decode_tok_s from _metadata.pie_timings on ONE instance at
# temperature 0 -- a rate, so it survives trajectory divergence. Do NOT run this
# with PIE_QWEN35_MOE_PROFILE=1: that syncs on every stage and depresses the
# absolute rate. Profile for attribution, sweep for rate, never both at once.
#
# Usage:  bash 41_moe_decode_sweep.sh
#         MOE_VARIANTS="base wmma" bash 41_moe_decode_sweep.sh
# =============================================================================
set -uo pipefail   # not -e: one variant failing must not lose the rest

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /workspace/pie-bench-env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-/workspace/pie/integrations/openhands/logs}
SWEEP_LOG="$LOG_DIR/moe_sweep_${TS}"
mkdir -p "$SWEEP_LOG"
SUMMARY="$SWEEP_LOG/SUMMARY.tsv"

INSTANCE=${MOE_INSTANCE:-django__django-14373}
VARIANTS=${MOE_VARIANTS:-"base p16 wmma aligned8"}

printf 'variant\tcfg\trc\twall_s\tdecode_tok_s\teffective_tok_s\titers\ttokens\terror\n' > "$SUMMARY"
echo "=== MoE decode sweep $TS  instance=$INSTANCE"
echo "    variants: $VARIANTS"
echo "    logs:     $SWEEP_LOG"

wait_for_free_gpu() {
    local used
    for _ in $(seq 1 120); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-99999}" -lt 2000 ] && { echo "    GPU free (${used} MiB)"; return 0; }
        sleep 5
    done
    echo "    WARNING: GPU still holds ${used} MiB — continuing anyway"
}

for v in $VARIANTS; do
    unset PIE_QWEN35_MOE_WMMA_DECODE
    unset PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES
    unset PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK
    cfg=auto_p32
    case "$v" in
        base)          : ;;
        # Attention, not MoE: the phase profile puts full_attn at ~47% of decode
        # kernel time, reading KV at roughly 0.6 TB/s on a ~4.8 TB/s part. Page
        # size is the one attention knob reachable without a rebuild.
        p16)           cfg=auto_p16 ;;
        wmma)          export PIE_QWEN35_MOE_WMMA_DECODE=1 ;;
        aligned8)      export PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES=8 ;;
        aligned8_blk8) export PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES=8
                       export PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK=8 ;;
        *) echo "unknown variant '$v'" >&2; continue ;;
    esac

    out="predictions/moe_${v}_${TS}.jsonl"
    log="$SWEEP_LOG/${v}.log"
    echo ""
    echo "=== [$(date +%H:%M:%S)] variant=$v → $log"
    wait_for_free_gpu

    t0=$(date +%s)
    AB_INSTANCES="$INSTANCE" PIE_CFG_VARIANT="$cfg" ARM=pie OUTPUT="$out" \
        bash "$RUNPOD_DIR/30_ab_run.sh" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)

    # A variant can "succeed" at rc=0 and still be worthless: the profiled run
    # that motivated this sweep died on a websocket timeout after 11 calls and
    # recorded iters=0 with a decode rate that looked like a real measurement.
    # Carry the error field into the summary so that is visible at a glance.
    read -r dts ets it tk err < <(python3 - "$out" <<'PY'
import json, sys, os
p = os.path.join('/workspace/pie/integrations/openhands', sys.argv[1])
try:
    r = [json.loads(l) for l in open(p) if l.strip()][0]
    m = r.get('_metadata') or {}
    t = m.get('pie_timings') or {}
    e = str(m.get('error') or 'none').split(':')[0].replace(' ', '_')[:40]
    print(t.get('decode_tok_s', 'NA'), t.get('effective_tok_s', 'NA'),
          m.get('agent_iterations', 'NA'), t.get('tokens_generated', 'NA'), e)
except Exception:
    print('NA NA NA NA unreadable')
PY
)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$v" "$cfg" "$rc" "$((t1 - t0))" "$dts" "$ets" "$it" "$tk" "$err" >> "$SUMMARY"
    echo "=== variant=$v rc=$rc wall=$((t1 - t0))s decode_tok_s=$dts"
done

echo ""
echo "=== sweep done ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
