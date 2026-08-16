#!/usr/bin/env bash
# Hold the CoreSpotlight knowledge-graph indexer down while a benchmark runs.
#
# ## READ THIS BEFORE USING IT
#
# **This is probably not worth running.** It is here because it was asked for,
# and because the alternative -- discovering all of the below again -- costs more
# than the file. Three findings, in the order that matters:
#
# 1. **It cannot be disabled.** `com.apple.spotlightknowledged` is protected by
#    System Integrity Protection:
#
#        $ launchctl bootout gui/501/com.apple.spotlightknowledged
#        Boot-out failed: 150: Operation not permitted while System Integrity
#        Protection is engaged
#
#    Nor does `sudo mdutil -i off /` help: that governs the FILE indexer
#    (`mds` / `mds_stores`), which on this machine has sat at 0.0% CPU for days.
#    The busy process is the KNOWLEDGE-GRAPH indexer, a different service over
#    `~/Library/Metadata/CoreSpotlight/SpotlightKnowledge/index.V2/KG/`. Turning
#    off file indexing would degrade Spotlight and change nothing here.
#
#    So the only lever left is SIGTERM, which works (the process runs as the
#    logged-in user) and which launchd answers by restarting it.
#
# 2. **It is doing legitimate first-time work.** `/var/db/.AppleSetupDone` is
#    dated 2026-08-12 and every Spotlight daemon has been alive since then. This
#    is a four-day-old machine still building its initial semantic index over app
#    content -- Mail, Messages, Photos, Notes. Killing it in a loop does not
#    cancel that work, it defers it, and repeatedly interrupting a first-time
#    index may stop it ever finishing. **That is a real cost to the machine's
#    owner, paid to remove a contention that measurably almost does not matter.**
#
# 3. **It measurably almost does not matter.** The four-way benchmark was run
#    once with this indexer saturating a CPU core and once after killing it, on
#    an otherwise idle machine. Every cell agreed within ~3%, and several within
#    0.3% (`results-four-way.md`). This workload is GPU- and bandwidth-bound; the
#    streaming roof never left 288-296 GB/s. One busy CPU core does not reach it.
#
# A quick A/B of this script was attempted and is NOT evidence, which is worth
# recording so it is not repeated: run free-then-suppressed against one server,
# the suppressed arm reported 0.43 s TTFT at 5,840 tokens against 2.79 s. That is
# a PREFIX-CACHE HIT -- the free arm had just prefilled that exact prompt -- and
# the +5% / +10% at the two longer prompts is the known thermal drift, the
# suppressed arm having run second and warmer. Two confounds pointing opposite
# ways in one three-cell table. A valid A/B needs a fresh server per arm and
# alternating order, which is what `four_way.sh` does and why its
# contended-vs-clean comparison (<3% everywhere) is the number to trust.
#
# **Recommendation: let it finish.** Use `tools/when_quiet.sh` to wait it out if
# a run must be pristine, and rely on the repeated-arm drift control that
# `four_way.sh` already carries -- which, note, showed the ~11% drift at the long
# end to be THERMAL and present on an idle machine anyway.
#
# Usage:  bash tools/suppress_spotlight.sh -- <command...>
set -uo pipefail

PATTERN='CoreSpotlight.framework/spotlightknowledged'
THRESHOLD=25          # %CPU above which it is considered to be indexing
INTERVAL=5

while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; break ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: suppress_spotlight.sh -- <command...>" >&2; exit 2; }

kills=0
supervise() {
    while :; do
        # Only the BUSY instance, by CPU. There are idle spotlightknowledged
        # processes days old that are not doing anything and must be left alone.
        local pid
        pid=$(ps -Ao pid,pcpu,comm -r 2>/dev/null |
              awk -v p="$PATTERN" -v t="$THRESHOLD" \
                  '$2 > t && $0 ~ p {print $1; exit}')
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null && kills=$(( kills + 1 ))
        fi
        sleep "$INTERVAL"
    done
}

supervise & SUPERVISOR=$!
# Restore on ANY exit, including a failure or an interrupt. A supervisor left
# running would keep SIGTERMing a system service long after the benchmark that
# justified it, which is exactly the failure this trap exists to prevent.
cleanup() {
    kill "$SUPERVISOR" 2>/dev/null
    wait "$SUPERVISOR" 2>/dev/null
    echo "── supervisor stopped; the indexer will resume and finish its work" >&2
}
trap cleanup EXIT INT TERM

echo "── suppressing '$PATTERN' above ${THRESHOLD}% CPU while: $*" >&2
"$@"
