#!/usr/bin/env bash
# =============================================================================
# Concurrency sweep — the throughput question AGENT_HANDOVER_20260728.md §6a
# names as the recommended next phase.
#
# Single-stream we measured 1.34x (Pie slower). That number is a LATENCY result
# and it is also, by construction, the wrong lens for the overlap lever: with
# one conversation, turn N's prefill must precede turn N's decode, so no
# scheduling change can remove that serialization. Concurrency is where Pie's
# scheduling either closes the gap or widens it, and nobody has measured it.
#
# Four arms, strictly sequential — 60 GB of weights per engine and neither arm
# may see the other's GPU pressure:
#
#   1. pie      c8   the new question
#   2. litellm  c8   its control
#   3. pie      c1   scaling reference AND the 13-instance serial A/B that
#   4. litellm  c1   §6c gates publishing any ratio on
#
# c8 runs first so a mid-sweep failure still leaves the head-to-head that
# motivated the sweep. The headline metric is instances/GPU-hour (TEST_PLAN.md),
# NOT s/iter: at c8 the per-iteration number is contended by construction.
#
# Usage:  bash 40_concurrency_sweep.sh            # all four arms
#         SWEEP_ARMS="pie:8 litellm:8" bash ...   # a subset
# =============================================================================
set -uo pipefail   # deliberately NOT -e: one arm failing must not lose the rest

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /workspace/pie-bench-env.sh

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-/workspace/pie/integrations/openhands/logs}
SWEEP_LOG="$LOG_DIR/sweep_${SWEEP_TS}"
mkdir -p "$SWEEP_LOG"
SUMMARY="$SWEEP_LOG/SWEEP_SUMMARY.tsv"

ARMS=${SWEEP_ARMS:-"pie:8 litellm:8 pie:1 litellm:1"}

printf 'arm\tconcurrency\tstart\tend\twall_s\trc\toutput\n' > "$SUMMARY"
echo "=== concurrency sweep $SWEEP_TS ==="
echo "    arms:   $ARMS"
echo "    logs:   $SWEEP_LOG"
echo "    summary:$SUMMARY"

# An arm's server must be fully gone before the next boots — otherwise the
# incoming engine sizes its KV pool against memory the outgoing one still holds,
# which silently shrinks the pool rather than erroring.
wait_for_free_gpu() {
    local used
    for _ in $(seq 1 120); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${used:-99999}" -lt 2000 ]; then
            echo "    GPU free (${used} MiB)"
            return 0
        fi
        sleep 5
    done
    echo "    WARNING: GPU still holds ${used} MiB after 10 min — continuing anyway"
    return 0
}

for spec in $ARMS; do
    arm=${spec%%:*}
    conc=${spec##*:}
    tag="${arm}_c${conc}"
    log="$SWEEP_LOG/${tag}.log"

    echo ""
    echo "=== [$(date +%H:%M:%S)] arm=$arm concurrency=$conc → $log"
    wait_for_free_gpu

    t0=$(date +%s); t0h=$(date +%H:%M:%S)
    if [ "$arm" = "pie" ]; then
        CONCURRENCY=$conc PIE_CFG_VARIANT=auto_p32 ARM=pie \
            bash "$RUNPOD_DIR/30_ab_run.sh" > "$log" 2>&1
    else
        CONCURRENCY=$conc VLLM_TIER=fair ARM=litellm \
            bash "$RUNPOD_DIR/30_ab_run.sh" > "$log" 2>&1
    fi
    rc=$?
    t1=$(date +%s); t1h=$(date +%H:%M:%S)

    out=$(grep -oE 'predictions/[A-Za-z0-9_./-]+\.jsonl' "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$arm" "$conc" "$t0h" "$t1h" "$((t1 - t0))" "$rc" "${out:-?}" >> "$SUMMARY"
    echo "=== [$t1h] arm=$arm c=$conc rc=$rc wall=$((t1 - t0))s out=${out:-?}"
done

echo ""
echo "=== sweep done ==="
cat "$SUMMARY"
