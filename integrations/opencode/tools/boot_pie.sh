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

pkill -f "release/pie -c" 2>/dev/null || true
for _ in $(seq 1 30); do
    pgrep -f "release/pie -c" >/dev/null || break
    sleep 1
done
if pgrep -f "release/pie -c" >/dev/null; then
    echo "FATAL: a pie server survived the kill" >&2
    pgrep -fl "release/pie -c" >&2
    exit 1
fi
if lsof -ti :8080 >/dev/null 2>&1; then
    echo "FATAL: :8080 still held by something else" >&2
    exit 1
fi

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
    if curl -s -m 3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; then
        # /health answering is NOT proof it is OUR server: that was the whole
        # bug. The config path is unique per boot (mktemp), so match on it.
        cfg=$(grep -o "T/pie_opencode\.[A-Za-z0-9]*" "/tmp/serve_$TAG.out" | head -1)
        if [ -n "$cfg" ] && ps -o command= -p "$(pgrep -f "release/pie -c" | head -1)" 2>/dev/null | grep -q "$cfg"; then
            echo "up: $TAG serving from $cfg"
            exit 0
        fi
    fi
done
echo "FATAL: $TAG never came up" >&2
tail -20 "/tmp/pie_$TAG.log" >&2
exit 1
