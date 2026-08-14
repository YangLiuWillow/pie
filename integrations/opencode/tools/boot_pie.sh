#!/usr/bin/env bash
# Boot ONE pie serve and refuse to return unless THIS boot is the one serving.
#
# The reason this exists: `pkill -f "pie serve"` does not match the process,
# whose argv is `pie -c <config> serve`. Every kill silently no-op'd, every
# later boot lost the bind to the first server, and four measurements were
# taken against a server booted with different flags than the one under test.
# Nothing in the output said so -- /health answered `ok` throughout, because
# something WAS listening.
#
# So: kill by the pattern that actually matches, wait for the port to be free,
# boot, and then prove the live process is the one this script started.
set -euo pipefail
REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TAG="$1"; shift

# Scoped to THIS worktree's binary, and that scope is not optional.
#
# Two properties are needed at once. The pattern must survive argument
# reordering — `release/pie -c` assumed the config flag sits immediately after
# the binary, and the first time a flag was added in front of it
# (`--metrics-addr`) the kill silently no-op'd and the next boot found :8080
# still held, the same trap as `pkill -f "pie serve"` above. And it must not
# reach outside this checkout: a bare `release/pie .*serve` matches
# `./target/release/pie serve` from ANY sibling worktree, and this script
# SIGTERMed a peer session's unrelated validation server three times in fifteen
# minutes, twice mid-run, before that was noticed.
#
# Anchoring on the absolute path this script actually boots gives both. A
# server started some other way no longer matches — which is correct: the
# :8080 check below then fails loudly instead of this script quietly killing
# something that is not its business.
PIE_PAT="$REPO/target/release/pie .*serve"
pkill -f "$PIE_PAT" 2>/dev/null || true
for _ in $(seq 1 30); do
    pgrep -f "$PIE_PAT" >/dev/null || break
    sleep 1
done
if pgrep -f "$PIE_PAT" >/dev/null; then
    echo "FATAL: a pie server survived the kill" >&2
    pgrep -fl "$PIE_PAT" >&2
    exit 1
fi
if lsof -ti :8080 >/dev/null 2>&1; then
    echo "FATAL: :8080 still held by something else" >&2
    exit 1
fi

# Strategy B puts the SESSION SHIM on the client port and moves pie to an
# internal one. The shim serves /v1/models and has no /health, so probing
# /health there waits forever for an endpoint that does not exist.
PROBE=/health
for a in "$@"; do case "$a" in PIE_STRATEGY=b|PIE_STRATEGY=B) PROBE=/v1/models ;; esac; done

cd "$REPO"
env PIE_BIN="$REPO/target/release/pie" LOG="/tmp/pie_$TAG.log" "$@" \
    nohup ./integrations/opencode/run_pie_opencode.sh --serve-only \
    > "/tmp/serve_$TAG.out" 2>&1 &

for _ in $(seq 1 120); do
    sleep 2
    if grep -q "Address already in use" "/tmp/pie_$TAG.log" 2>/dev/null; then
        echo "FATAL: this boot lost the bind" >&2
        exit 1
    fi
    if curl -sf -m 3 -o /dev/null "http://127.0.0.1:8080$PROBE" 2>/dev/null; then
        # /health answering is NOT proof it is OUR server: that was the whole
        # bug. The config path is unique per boot (mktemp), so match on it.
        cfg=$(grep -o "T/pie_opencode\.[A-Za-z0-9]*" "/tmp/serve_$TAG.out" | head -1)
        if [ -n "$cfg" ] && ps -o command= -p "$(pgrep -f "$PIE_PAT" | head -1)" 2>/dev/null | grep -q "$cfg"; then
            echo "up: $TAG serving from $cfg"
            exit 0
        fi
    fi
done
echo "FATAL: $TAG never came up" >&2
tail -20 "/tmp/pie_$TAG.log" >&2
exit 1
