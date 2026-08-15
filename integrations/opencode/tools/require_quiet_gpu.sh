#!/usr/bin/env bash
# Refuse to start a benchmark arm on a machine that cannot hold it.
#
# ## Why this exists
#
# On 2026-08-14 a vLLM-metal SWE-bench arm booted cleanly, answered
# `/v1/models` in 22 s, and then died on the first real 7k-token prompt with
#
#     [METAL] Command buffer execution failed: Insufficient Memory
#     (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
#     -> vllm.v1.engine.exceptions.EngineDeadError -> HTTP 500
#
# opencode recorded `Cannot connect to API ... exit 1` on both instances, and
# the summary line read "0/2 produced a non-empty patch" -- which is
# indistinguishable from a model that genuinely solved nothing. It was nearly
# published as a model result.
#
# The machine was at 14% free memory and 89% swap consumption BEFORE the second
# server started. Nothing in the harness looked at that.
#
# ## The two rules this encodes
#
# 1. **Liveness at boot proves nothing.** A server that is up at t=0 and dead at
#    t=60 produces an identical summary. Check preconditions BEFORE, and check
#    for death AFTER (see `arm_is_valid` below).
# 2. **A run of identical fast failures is never a model result.** The first
#    failure is memory pressure; everything after it is a latched dead engine.
#    Confirmed on two independent stacks -- ggml-metal latches an unresettable
#    error flag, and vLLM raises `EngineDeadError` permanently -- so treat the
#    pattern as engine-independent.
#
# Usage:
#   source tools/require_quiet_gpu.sh
#   require_quiet_gpu 18        # need ~18 GB for this engine; exits 1 if not
#   ... run the arm ...
#   arm_is_valid <server.log> <client.log> <health-url> || echo "VOID"

# Free physical memory in GB, as macOS actually reports it. `vm_stat`'s
# free+inactive alone overstates headroom on a machine that is already
# swapping, so swap headroom is checked separately rather than folded in.
_free_gb() {
    vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3}
                   END{gsub(/\./,"",f); gsub(/\./,"",i);
                       printf "%.1f", (f+i)*16384/1073741824}'
}

_swap_free_mb() {
    sysctl -n vm.swapusage 2>/dev/null |
        sed -E 's/.*free = ([0-9.]+)M.*/\1/'
}

# require_quiet_gpu <needed_gb> [min_swap_free_mb]
#
# `min_swap_free_mb` defaults to 2048. That is not a tuned number -- it is one
# doubling above the 903 MB at which the vLLM arm died, chosen so the threshold
# sits clear of the one datapoint we have rather than on top of it.
require_quiet_gpu() {
    local need="${1:?usage: require_quiet_gpu <needed_gb> [min_swap_free_mb]}"
    local min_swap="${2:-2048}"
    local free swapfree
    free=$(_free_gb); swapfree=$(_swap_free_mb)

    echo "── preflight: free=${free} GB, swap_free=${swapfree} MB, need≈${need} GB"

    # Other model servers are the usual reason this fails, so name them rather
    # than just reporting a number the caller has to interpret.
    local others
    others=$(ps -Ao rss,comm -r 2>/dev/null |
             awk 'NR>1 && $1 > 2097152 {printf "    %.1f GB  %s\n", $1/1048576, $2}')
    [ -n "$others" ] && { echo "── processes over 2 GB:"; echo "$others"; }

    # FREE PHYSICAL MEMORY IS THE GATE. Swap is only a tie-breaker, and that
    # ordering is a correction: the first version of this function failed hard
    # on swap headroom alone and promptly refused a run on an idle machine with
    # 36.9 GB free, because macOS does not eagerly reclaim swap once pressure
    # subsides -- `used` there is historical, not current. Calibrating a
    # threshold on the single datapoint from one OOM (14% free AND 903 MB swap)
    # over-fitted to the swap half of a condition whose load-bearing half was
    # the free-memory half.
    if awk -v f="$free" -v n="$need" 'BEGIN{exit !(f < n)}'; then
        echo "FATAL: ${free} GB free, arm needs about ${need} GB." >&2
        echo "       Free memory before starting; an OOM here voids the arm" >&2
        echo "       silently rather than failing loudly." >&2
        return 1
    fi
    # Low swap on top of ample free memory is not dangerous, but low swap while
    # free memory is merely adequate is exactly the OOM condition. Only then.
    # The swap tie-breaker needs an ABSOLUTE headroom test as well as a ratio,
    # and this is the second calibration correction it has needed. The ratio
    # alone (`free < need * 1.5`) refused a 20 GB arm on a machine with 29.0 GB
    # free and 9 GB of headroom -- four times the free memory the 2026-08-14 OOM
    # actually happened at. A rule tuned on one datapoint generalises badly in
    # both directions: the first version over-fitted to swap, this one to the
    # ratio. Refuse only when headroom is thin in ratio AND in absolute terms.
    if awk -v f="$swapfree" -v m="$min_swap" 'BEGIN{exit !(f < m)}'; then
        if awk -v f="$free" -v n="$need" 'BEGIN{exit !(f < n * 1.5 && f - n < 8)}'; then
            echo "FATAL: ${free} GB free is only just above the ${need} GB needed" >&2
            echo "       AND swap headroom is ${swapfree} MB. That pairing is the" >&2
            echo "       state the 2026-08-14 vLLM arm OOM'd in." >&2
            return 1
        fi
        echo "── note: swap headroom ${swapfree} MB is low, but ${free} GB free is" 
        echo "──       ample; proceeding. Swap used is historical on macOS."
    fi
    # CPU contention, which is NOT the same question as free memory and was
    # missed once because of that: on 2026-08-15 an 8 GB python job at 17% CPU
    # inflated every kernel timing in `sdpa_paged_probe` by ~1.4x, and inflated
    # the memory-bound arm of an A/B more than the compute-bound one -- which
    # silently changed a 39/49 split into 65/45. Free memory was ample
    # throughout, so this function said OK. A benchmark that is contended is
    # not void, it is WORSE than void: it still produces plausible numbers.
    local busy
    busy=$(ps -Ao pcpu,comm -r 2>/dev/null |
           awk 'NR>1 && $1 > 10 && $2 !~ /WindowServer|ps$/ {n++} END{print n+0}')
    if [ "$busy" -gt 0 ]; then
        echo "── WARNING: ${busy} process(es) over 10% CPU. Timings will be inflated;"
        echo "──          A/B RATIOS will skew toward the memory-bound arm."
        ps -Ao pcpu,comm -r 2>/dev/null | awk 'NR>1 && $1 > 10 {printf "    %5.1f%%  %s\n", $1, $2}' | head -4
    fi
    # MEMORY PRESSURE, which is the one that actually corrupted a result. On
    # 2026-08-15 a recurring 14 GB python job cycled in the background: it sat
    # at 17% CPU (under the bar above) and left free memory looking fine, but
    # the compressor was doing millions of decompressions and the machine's
    # read-only streaming roof fell from ~200 GB/s to 69.8. Compute was
    # untouched -- `matrix_rate_probe` reproduced to 4% -- so every
    # register-bound measurement looked healthy while every memory-bound one
    # was inflated 1.3-2.2x. That asymmetry is what makes it dangerous: it does
    # not break an A/B, it TILTS one.
    #
    # `roofline_probe` is the ground truth here; this is the cheap proxy.
    local heavy
    heavy=$(ps -Ao rss,comm -r 2>/dev/null |
            awk 'NR>1 && $1 > 8388608 && $2 !~ /Claude/ {n++} END{print n+0}')
    if [ "$heavy" -gt 0 ]; then
        echo "FATAL: ${heavy} process(es) over 8 GB resident." >&2
        ps -Ao rss,comm -r 2>/dev/null |
            awk 'NR>1 && $1 > 8388608 {printf "       %.1f GB  %s\n", $1/1048576, $2}' >&2
        echo "       Memory-bound timings will be inflated and A/B ratios will" >&2
        echo "       tilt toward the memory-bound arm. Wait for it to finish." >&2
        return 1
    fi
    echo "── preflight OK"
    return 0
}

# arm_is_valid <server_log> <client_log> <health_url>
#
# Run AFTER the arm. Prints a verdict and returns non-zero if the numbers must
# not be quoted.
arm_is_valid() {
    local slog="$1" clog="$2" url="$3"
    local dead unreach alive
    # NOTE: no `|| echo 0` here, deliberately. `grep -c` prints "0" AND exits
    # 1 when nothing matches, so the fallback appends a SECOND zero and the
    # variable becomes "0\n0" -- which is != "0" and makes this gate declare a
    # perfectly healthy arm void. That false alarm cost a re-run once already.
    dead=$(grep -acE "Insufficient Memory|EngineDead|kIOGPUCommandBuffer|CUDA out of memory" "$slog" 2>/dev/null)
    unreach=$(grep -acE "Cannot connect to API|Unable to connect|Connection refused" "$clog" 2>/dev/null)
    # This one DOES need the fallback: `curl && echo 1` prints nothing on
    # failure, and an empty string is not "1" only by accident. Say 0 outright.
    alive=$(curl -s -m 5 "$url" >/dev/null 2>&1 && echo 1 || echo 0)
    # PER-REQUEST failures, which this gate did NOT check until an arm returned
    # 22 HTTP 500s out of 32 and was still stamped VALID (2026-08-15, pie
    # strategy B). The three checks above ask "did the server die?"; none asks
    # "did the requests work?". A server that stays up and refuses most of the
    # traffic is not a valid arm either, and its surviving throughput number is
    # a survivorship artifact -- exactly what this gate exists to prevent, in
    # the one shape it did not cover.
    local reqfail
    reqfail=$(grep -acE "HTTPError|HTTP Error [45][0-9][0-9]" "$clog" 2>/dev/null)
    echo "── validity: server_errors=${dead} client_unreachable=${unreach} alive_at_end=${alive} request_failures=${reqfail}"
    if [ "$reqfail" != "0" ]; then
        echo "!!! ARM VOID — requests failed. A throughput number computed from" >&2
        echo "    the survivors measures the survivors, not the engine." >&2
        return 1
    fi
    if [ "$dead" != "0" ] || [ "$unreach" != "0" ] || [ "$alive" != "1" ]; then
        echo "!!! ARM VOID — the server died or was unreachable. These numbers are" >&2
        echo "    not a model result and must not be reported as one." >&2
        return 1
    fi
    echo "── arm VALID"
    return 0
}
