#!/usr/bin/env bash
# Boot pie serve + the codex-responses HTTP daemon that Codex talks to.
#
#   bash run_pie_codex.sh            # foreground; Ctrl-C stops everything
#
# Env overrides: PIE (server binary), CFG (pie config), CONTROL_PORT,
# HTTP_PORT, WASM, MANIFEST.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

PIE="${PIE:-$REPO/target/release/pie}"
CFG="${CFG:-$HERE/pie_config.toml}"
CONTROL_PORT="${CONTROL_PORT:-18080}"
HTTP_PORT="${HTTP_PORT:-8123}"
WASM="${WASM:-$REPO/inferlets/codex-responses/target/wasm32-wasip2/release/codex_responses.wasm}"
MANIFEST="${MANIFEST:-$REPO/inferlets/codex-responses/Pie.toml}"
PIE_LOG="${PIE_LOG:-/tmp/pie_codex_serve.log}"
VENV="$HERE/.venv"

[[ -x "$PIE" ]] || { echo "pie binary not found at $PIE (cargo build --release -p pie-server)"; exit 1; }
[[ -f "$WASM" ]] || { echo "wasm not found at $WASM (cargo build --target wasm32-wasip2 --release)"; exit 1; }

echo "[1/3] pie serve on :$CONTROL_PORT (log: $PIE_LOG)"
# PIE_SHMEM_TIMEOUT_S: the shmem RPC hard timeout defaults to 60s; a
# 1024-token CPU prefill chunk takes ~90-110s, so every such forward would
# silently return an empty output (the driver keeps grinding). Generous on
# CPU; harmless on GPU where forwards take milliseconds.
PYTHONPATH="" PIE_SHMEM_TIMEOUT_S="${PIE_SHMEM_TIMEOUT_S:-600}" \
    "$PIE" serve --config "$CFG" --port "$CONTROL_PORT" --no-auth >"$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 900); do
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo "pie serve died — tail of $PIE_LOG:"; tail -20 "$PIE_LOG"; exit 1
    fi
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && break
    sleep 2
done
grep -q "pie-server serving on" "$PIE_LOG" || { echo "pie serve never became ready"; exit 1; }
echo "      ready."

echo "[2/3] install inferlet + launch HTTP daemon on :$HTTP_PORT"
"$VENV/bin/python" "$HERE/launch_daemon.py" \
    --uri "ws://127.0.0.1:$CONTROL_PORT" \
    --wasm "$WASM" --manifest "$MANIFEST" --port "$HTTP_PORT"

echo "[3/3] serving. Point Codex at http://127.0.0.1:$HTTP_PORT/v1 (see README.md)."
echo "      Ctrl-C to stop."
wait "$PIE_PID"
