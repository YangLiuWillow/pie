#!/usr/bin/env bash
# Re-trace a decode step's composition by ABLATION, on the current build.
#
# ## Why this exists
#
# The two decode breakdowns in `docs/` disagree about where the time is, and the
# disagreement decides what to work on: the dense projections are either 1.87x
# off their roofline and worth ~5.8% of a step, or already at it and worth
# nothing. One of the two came from the dispatch trace, which inflates small
# kernels by up to 38x, so it cannot settle this.
#
# ## The method, and its one precondition
#
# Run the decode step with a kernel kind NOT DISPATCHED and take the delta. That
# is sound only for a kind that emits VALUES. `moe_route_sort` emits INDICES --
# removing it sends every downstream kernel chasing garbage, and it once
# measured a 2.04 ms "saving" that was nothing of the kind. The kinds swept here
# all emit activations.
#
# The tokens are `Kernel` kind names, NOT pipeline host names: pasting
# `affine_qmv_routed_bfloat16_gs_64_b_4` in here matches no kind, ablates
# nothing, and reports the baseline while printing an armed-looking banner. The
# driver says so loudly; this script checks for it anyway.
#
# ## Reading it
#
# The delta is that kind's cost. Shares will not sum to 100% -- a step also
# contains norms, rope, the KV append, the sampler and the barriers between
# them -- and a kind whose delta is at or below the run-to-run spread is being
# reported as noise, not as free.
#
# ## ABLATE A CONCURRENCY GROUP WHOLE, NEVER ONE MEMBER
#
# `concurrency_group` in `model/llama/encode.cpp` runs some kinds with NO
# barrier between them, so their wall time is the OVERLAP and not the sum.
# Ablating one member leaves the other covering the same window and the removed
# one reads as free. Measured here, per kind at ctx 7424:
#
#     ll_expert_gate   4.03 ms   23.2%
#     ll_expert_up     0.03 ms    0.2%     <-- same shape, same weight bytes
#
# Two projections of identical shape cannot differ by 130x. They are group 5.
# The groups on a llama decode step are:
#
#     1  qmv_q, qmv_k, qmv_v          all read the attention norm's output
#     2  q_norm, k_norm               each rewrites its own tensor
#     3  rope_q, rope_k               disjoint
#     4  qmv_gate, qmv_up             both read the FFN norm's output
#     5  ll_expert_gate, ll_expert_up the routed pair
#
# So the per-kind sweep below is kept for the ungrouped kinds and to expose this
# effect, and the GROUP sweep after it is the one whose numbers mean anything
# for kinds 1, 4 and 5.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${RETRACE_OUT:-/tmp/decode-retrace-$(date +%m%d-%H%M)}"
CONF="${ROWS_PROBE_CONF:-/tmp/rows-probe/config.toml}"
mkdir -p "$OUT"
cd "$REPO"

cargo build --release -p pie-bin --features driver-metal || exit 1
if pgrep -f "$REPO/target/release/pie .*serve" >/dev/null; then
    echo "FATAL: a pie server is running; it will starve this probe of memory" >&2
    exit 1
fi

WASM=runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm
MANIFEST=runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml

one() {  # $1 label, $2 ablate spec ("" = baseline)
    sleep "${RETRACE_SETTLE:-25}"   # let the previous run's 16 GiB go back
    PIE_METAL_ABLATE="$2" ./target/release/pie -c "$CONF" run \
        --path "$WASM" --manifest "$MANIFEST" > "$OUT/$1.txt" 2>&1
    if [ -n "$2" ] && ! grep -q "\[ablate\] PIE_METAL_ABLATE=$2" "$OUT/$1.txt"; then
        echo "FATAL: $1 -- the ablate banner never printed, so nothing was skipped" >&2
        exit 1
    fi
    if grep -q "IS NOT A KERNEL KIND" "$OUT/$1.txt"; then
        echo "FATAL: $1 -- a token matched no kernel kind:" >&2
        grep "IS NOT A KERNEL KIND" "$OUT/$1.txt" >&2
        exit 1
    fi
    local ms
    ms=$(grep -o "\[rows\] ctx=[0-9]* rows=1 median_ms=[0-9.]*" "$OUT/$1.txt" \
         | head -1 | grep -o "[0-9.]*$")
    if [ -z "$ms" ]; then
        echo "FATAL: $1 produced no rows=1 timing:" >&2; tail -4 "$OUT/$1.txt" >&2; exit 1
    fi
    echo "$ms"
}

# Baseline FIRST and LAST. Same drift control `four_way.sh` carries: if the two
# disagree, the deltas between them are the machine and not the kernels.
BASE1=$(one baseline-first "")
echo "baseline            ${BASE1} ms"
# Whole GROUPS where a group exists, single kinds where it does not. A comma
# list is one ablation of everything in it, which is the only sound way to price
# a set of dispatches that run without barriers between them.
declare -a NAMES=(
    # The three big blocks, whole groups where pso_kind maps many kinds to one.
    sdpa
    ll_expert_gate,ll_expert_up
    ll_expert_down
    qmv_gate,qmv_up,qmv_down,qmv_q,qmv_k,qmv_v,qmv_o
    # THE UNATTRIBUTED 30.6%. Everything else a layer dispatches, which between
    # them cost more than any single block above and have never been priced.
    rms
    rope
    kv_append
    silu_mul
    residual
    ll_moe_gather
    ll_moe_combine
    # `ll_moe_sort` is deliberately ABSENT. It emits INDICES, and plain ablation
    # sends every downstream kernel chasing garbage -- that is how it once
    # measured a 2.04 ms "saving" that was nothing of the kind. It is already
    # priced at 1.4% by `PIE_METAL_MOE_SORT_SKIP_AFTER`, which leaves an earlier
    # layer's indices in place so the access patterns stay real.
)
for k in "${NAMES[@]}"; do
    t=$(one "ablate-$(echo "$k" | tr ',' '+')" "$k")
    awk -v b="$BASE1" -v t="$t" -v k="$k" \
        'BEGIN{printf "  -%-28s %7.2f ms   cost %6.2f ms  (%5.1f%% of the step)\n", k, t, b-t, (b-t)/b*100}'
done
BASE2=$(one baseline-last "")
echo "baseline repeated   ${BASE2} ms"
awk -v a="$BASE1" -v b="$BASE2" 'BEGIN{
  d=(b-a)/a*100; printf "\ndrift over the run: %+.2f%%%s\n", d,
    (d<-2||d>2) ? "   <-- EXCEEDS 2%: the deltas above are the machine, not the kernels" : ""}'
echo "raw output in $OUT"
