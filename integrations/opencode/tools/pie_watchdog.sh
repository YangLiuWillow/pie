#!/usr/bin/env bash
# Fail-fast watchdog for the pie arm. Emits a line ONLY when a check trips, so
# silence means healthy and any output is worth stopping for.
#
# Every check answers a failure this repo has ACTUALLY hit, and each one exists
# because that failure was invisible in the timing columns, the patch count and
# the exit status at the time it happened. That is the selection rule: a failure
# that already announces itself does not need a check here.
#
# It lives in the repo now because the previous copy lived in /tmp, where it did
# not survive the machine and could not be reviewed alongside the code whose
# failures it watches.
#
#     tools/pie_watchdog.sh [--shim LOG] [--engine LOG] [--arm LOG] [--work DIR]
#
# Defaults match `tools/boot_pie.sh <tag>`: pass `--engine /tmp/pie_<tag>.log`.

set -uo pipefail

SHIM=/tmp/pie_opencode_shim.log
ENGINE=/tmp/pie_ramp.log
ARM=
WORK=
INTERVAL=${PIE_WATCHDOG_INTERVAL:-60}

while [ $# -gt 0 ]; do
    case "$1" in
        --shim)   SHIM="$2";   shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --arm)    ARM="$2";    shift 2 ;;
        --work)   WORK="$2";   shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# `grep -c` on a missing file, or a count with stray whitespace, silently makes
# every later arithmetic test false — which fails OPEN, the one direction a
# watchdog must never fail. Every count goes through here.
count() {  # count <pattern> <file>
    local f="$2"
    [ -f "$f" ] || { echo 0; return; }
    grep -ac -- "$1" "$f" 2>/dev/null | tr -dc '0-9' | sed 's/^$/0/'
}

fired_degraded=0
fired_reuse=0
fired_empty=0
fired_terminal=0
fired_refused=0
fired_cachefull=0
fired_cachewarn=0
fired_trim=0
fired_starved=0
fired_verify=0
fired_driver=0

# The Metal M1 program cache capacity, matched to the engine's own 256-entry
# program registry. A full cache no longer rejects -- it serves the program
# uncached and recompiles per fire -- so crossing this is a PERFORMANCE cliff
# now rather than the empty completions it used to cause. Still worth warning
# on: eviction is currently inert (every entry is pinned by the driver's
# ProgramRecord), so consumption remains one-way for the life of the process.
PROGRAM_CACHE_CAP=256
PROGRAM_CACHE_WARN=$((PROGRAM_CACHE_CAP * 7 / 8))

while true; do
    # ---- 1. Program cache exhausted. This was what the unexplained `status -5`
    #         failures were, back when a full cache REFUSED the fire and the
    #         turn came back empty. Kept after that was fixed: if it fires
    #         again, the fallback regressed.
    c=$(count "cache is full" "$ENGINE")
    if [ "$c" -gt "$fired_cachefull" ]; then
        echo "PROGRAM-CACHE-FULL=$c — register_program is rejecting; turns now degrade to empty completions"
        fired_cachefull=$c
    fi

    # ---- 2. ...and the same budget approaching its wall, while there is still
    #         time to act. Eviction is inert today, so this only ever rises.
    #         Counts compiles, which is the line `compile_program` now emits.
    used=$(count "register_program: compiling" "$ENGINE")
    if [ "$used" -ge "$PROGRAM_CACHE_WARN" ] && [ "$fired_cachewarn" -lt "$used" ]; then
        echo "PROGRAM-CACHE-HEADROOM: $used/$PROGRAM_CACHE_CAP entries used and never evicted"
        fired_cachewarn=$used
    fi

    # ---- 3. A turn the gateway refused. Survivable since the session is kept,
    #         but the turn still failed and the client saw a 503.
    #         Not necessarily admission: the same line carries "stream
    #         aborted" when a fire failed underneath, so the REASON is printed
    #         rather than assumed. Reading this as saturation once sent me
    #         looking at the pool for a driver poison epoch.
    r=$(count "gateway refused" "$SHIM")
    if [ "$r" -gt "$fired_refused" ]; then
        why=$(grep -a "gateway refused" "$SHIM" 2>/dev/null | tail -1 | sed 's/.*turn: //')
        echo "TURN-REFUSED=$r — the client got a 503; reason: ${why:-unknown}"
        fired_refused=$r
    fi

    # ---- 4. Retention thrashing. One flush at the top of the pool is by
    #         design; a stream of them means the pool cannot hold this
    #         workload and every flushed turn re-prefills at full price.
    t=$(count "dropped the TIP" "$SHIM")
    if [ "$t" -ge 3 ] && [ "$t" -gt "$fired_trim" ]; then
        echo "RETENTION-THRASH=$t tip drops — the pool cannot hold this conversation; turns are re-prefilling"
        fired_trim=$t
    fi

    # ---- 5. The trim ran out of things to drop, so the next turn is likely
    #         refused by a gate this guest can no longer influence.
    n=$(count "no retained branch left" "$SHIM")
    if [ "$n" -gt 0 ] && [ "$fired_starved" -eq 0 ]; then
        echo "POOL-UNRELIEVABLE: pages are held by something this guest does not own"
        fired_starved=1
    fi

    # ---- 6. A resume the engine refused on its own invariants. Loud by
    #         design, and it means a rebuild rather than a wrong prefix.
    v=$(count "kv_verify REFUSED" "$SHIM")
    if [ "$v" -gt "$fired_verify" ]; then
        echo "RESUME-REFUSED=$v — kv_verify rejected a branch; that turn rebuilt from cold"
        fired_verify=$v
    fi

    # ---- 7. Driver faults. A poison epoch is permanent: nothing clears it, so
    #         ONE of these converts the engine into a permanently refusing one
    #         that reads like a capacity problem.
    #
    #         Searched in BOTH logs, because it was missed in the engine one.
    #         The guest sees a poisoned channel at its own `take` and reports
    #         it on stderr, which the shim captures — so a poison epoch can be
    #         plainly visible to the inferlet and absent from the engine log
    #         this check was reading. Measured: `prefill take @7498: channel is
    #         poisoned: driver published poison epoch 1` in the shim log, zero
    #         hits in the engine log, and this check silent while the
    #         downstream TURN-REFUSED fired on the same event.
    d=0
    for pat in "poison" "Metal forward timed out" "readiness fault" "KV pool starved"; do
        d=$((d + $(count "$pat" "$ENGINE") + $(count "$pat" "$SHIM")))
    done
    if [ "$d" -gt "$fired_driver" ]; then
        echo "DRIVER-FAULT=$d (poison / fence timeout / readiness / pool starved) — see $ENGINE"
        fired_driver=$d
    fi

    # ---- 8. The socket died and took the process with it. Kept even though
    #         the gateway fix should stop it: if it fires again, that fix
    #         regressed, and this is the only place it would show.
    c=$(count "terminal event error" "$SHIM")
    if [ "$c" -gt "$fired_terminal" ]; then
        echo "INFERLET-CRASH=$c terminal event(s) — the process died and every retained branch with it"
        fired_terminal=$c
    fi

    # ---- 9. Reuse collapse, counted as CONSECUTIVE cold turns rather than a
    #         share of them.
    #
    #         A share cries wolf on exactly the run this exists to watch. Every
    #         new conversation legitimately starts cold, and a benchmark is
    #         nothing but conversation boundaries — three repeats of one
    #         SWE-bench instance tripped "6/12 recent turns had cached=0" while
    #         all three produced the correct patch and every health counter was
    #         zero. Over 30 instances a share-based check fires continuously
    #         and gets ignored, which is worse than no check.
    #
    #         Thrashing looks different: it is turn after turn re-prefilling
    #         the SAME growing history, so the cold turns are consecutive.
    #         Interspersed cold turns are just new conversations.
    if [ -f "$SHIM" ]; then
        streak=$(grep -aoE "cached=[0-9]+" "$SHIM" 2>/dev/null | tail -12 | awk '
            /cached=0$/ { n++; if (n > m) m = n; next }
            { n = 0 }
            END { print m + 0 }')
        streak=$(printf '%s' "${streak:-0}" | tr -dc '0-9')
        streak=${streak:-0}
        if [ "$streak" -ge 4 ] && [ "$fired_reuse" -eq 0 ]; then
            echo "REUSE-COLLAPSE: $streak consecutive turns had cached=0 — one conversation is re-prefilling every turn"
            fired_reuse=1
        fi
        [ "$streak" -lt 2 ] && fired_reuse=0
    fi

    # ---- 10. Degraded turns: a turn that died before producing anything, but
    #          still answered like a turn.
    if [ -n "$WORK" ] && [ -d "$WORK" ]; then
        g=$(cat "$WORK"/*.opencode.log 2>/dev/null | grep -ac "could not complete this turn" | tr -dc '0-9')
        g=${g:-0}
        if [ "$g" -gt "$fired_degraded" ]; then
            echo "DEGRADED-TURNS=$g — a turn died before producing anything"
            fired_degraded=$g
        fi
    fi

    # ---- 11. An all-empty arm is broken, not inaccurate.
    if [ -n "$ARM" ] && [ -f "$ARM" ]; then
        last=$(grep -aoE "patch [0-9]+ bytes" "$ARM" 2>/dev/null | tail -4 | grep -c "patch 0 bytes" | tr -dc '0-9')
        done_n=$(count "ok in" "$ARM")
        last=${last:-0}
        if [ "$last" -ge 4 ] && [ "$done_n" -ge 4 ] && [ "$fired_empty" -eq 0 ]; then
            echo "EMPTY-PATCH-RUN: last 4 instances all wrote 0 bytes (arm may be broken)"
            fired_empty=1
        fi
        [ "$last" -le 1 ] && fired_empty=0
    fi

    sleep "$INTERVAL"
done
