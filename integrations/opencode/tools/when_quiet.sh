#!/usr/bin/env bash
# Wait for the machine to actually go quiet, then run a command.
#
# `require_quiet_gpu` refuses an arm on a contended machine, which is right when
# a human is driving. When the contention is a system daemon that WILL finish on
# its own -- a Spotlight reindex, a Time Machine pass, a Photos analysis -- the
# useful behaviour is not to refuse but to wait, and then to run automatically.
#
# What "quiet" means here is deliberately two things, because on 2026-08-16 they
# disagreed: a CoreSpotlight indexer held a full CPU core for half an hour while
# the memory roof stayed at 292 GB/s, so the CPU check alone would have refused a
# run that the bandwidth check was happy with, and the CPU check alone is also
# what noticed the ~11% drift it caused on the longest prompt. Both are required:
#
#   * no non-trivial process over `MAX_PCPU`, for `STREAK` consecutive polls, so
#     a daemon that merely dips below the line does not count as finished;
#   * `roofline_probe` at or above `MIN_ROOF` GB/s, checked once at the end,
#     because it is the bandwidth authority and the cheap proxy can miss a
#     compressor storm entirely.
#
# Usage:  bash tools/when_quiet.sh [--timeout-min N] -- <command...>
set -uo pipefail

TIMEOUT_MIN=90
POLL_S=30
STREAK=3
MAX_PCPU=10
MIN_ROOF=280
ROOFLINE=/tmp/metaltools/bin/roofline_probe

while [ $# -gt 0 ]; do
    case "$1" in
        --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: when_quiet.sh [--timeout-min N] -- <command...>" >&2; exit 2; }

# WindowServer is always warm on a machine with a display; `ps` is this
# pipeline itself; and `claude` is the agent DRIVING the run, which spikes every
# time it issues a command. Counting the driver as contention makes this
# function unsatisfiable while it is being used, which is worse than useless --
# it is a wait that never ends and looks like a machine that never settles.
# Anything a caller wants ignored on top of these goes in IGNORE_RE.
IGNORE_RE="${IGNORE_RE:-WindowServer|ps$|claude|Claude}"

busy_count() {
    ps -Ao pcpu,comm -r 2>/dev/null |
        awk -v m="$MAX_PCPU" -v ig="$IGNORE_RE" \
            'NR>1 && $1 > m && $2 !~ ig {n++} END{print n+0}'
}

deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
quiet=0
echo "── waiting for a quiet machine (need ${STREAK} consecutive polls under ${MAX_PCPU}% CPU)"
while :; do
    n=$(busy_count)
    if [ "$n" -eq 0 ]; then
        quiet=$(( quiet + 1 ))
    else
        [ "$quiet" -gt 0 ] && echo "── ...busy again, streak reset"
        quiet=0
        ps -Ao pcpu,comm -r 2>/dev/null |
            awk -v m="$MAX_PCPU" -v ig="$IGNORE_RE" \
                'NR>1 && $1 > m && $2 !~ ig {printf "     %5.1f%%  %s\n", $1, $2}' | head -3
    fi
    [ "$quiet" -ge "$STREAK" ] && break
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "FATAL: still contended after ${TIMEOUT_MIN} min; NOT running the command." >&2
        echo "       Refusing rather than measuring dirty -- a contended run does" >&2
        echo "       not produce a void number, it produces a plausible wrong one." >&2
        exit 1
    fi
    sleep "$POLL_S"
done
echo "── CPU quiet for $(( STREAK * POLL_S ))s"

if [ -x "$ROOFLINE" ]; then
    roof=$("$ROOFLINE" 2>/dev/null | awk '/streaming roof/{print $(NF-1)}')
    if ! printf '%s' "$roof" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "FATAL: unparseable roof ('${roof}')" >&2; exit 1
    fi
    echo "── streaming roof ${roof} GB/s"
    awk -v r="$roof" -v m="$MIN_ROOF" 'BEGIN{exit !(r + 0 < m)}' && {
        echo "FATAL: roof ${roof} below ${MIN_ROOF}; something is still contending." >&2
        exit 1
    }
fi

echo "── machine is quiet; running: $*"
exec "$@"
