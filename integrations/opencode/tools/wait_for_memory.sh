#!/usr/bin/env bash
# Block until the machine can admit a model of $1 GiB (default 26), or give up.
#
# `sleep N` is the wrong instrument for this. Eight seconds died on instance 3
# of 30, thirty died on instance 14: the machine is not leaking -- idle it shows
# 38 GB reclaimable against 3 GB wired -- but macOS reclaims a 22.5 GiB model's
# pages on its own schedule and sometimes that exceeds any fixed sleep. A boot
# attempted into 11 GiB fails the whole arm, since the harness rightly refuses
# to drive a server it did not start.
#
# A FILE, not an inline string. The first attempt embedded this loop in the
# restart command and lost `avail` to three layers of quoting (bash script ->
# shell variable -> `zsh -c`), so the test never evaluated and the loop silently
# became a five-minute sleep per instance. Quoting bugs of that shape do not
# announce themselves; they just make the harness slow and the guard useless.
set -uo pipefail
NEED_GB="${1:-26}"
MAX_WAIT="${2:-300}"
avail_gb() {
    vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3}
                   END{gsub(/\./,"",f); gsub(/\./,"",i);
                       printf "%d", (f+i)*16384/1073741824}'
}
deadline=$(( $(date +%s) + MAX_WAIT ))
while :; do
    a=$(avail_gb)
    case "$a" in ''|*[!0-9]*) a=0 ;; esac      # a non-numeric read is not "plenty"
    if [ "$a" -ge "$NEED_GB" ]; then
        echo "[memwait] ${a} GB available (need ${NEED_GB})"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[memwait] TIMEOUT after ${MAX_WAIT}s: only ${a} GB available, need ${NEED_GB}" >&2
        exit 1
    fi
    sleep 5
done
