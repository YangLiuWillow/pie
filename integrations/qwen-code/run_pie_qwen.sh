#!/usr/bin/env bash
# Boot the full qwen-code ↔ Pie stack:
#   1. `pie serve` on :18080 (control plane)
#   2. the chat-completions inferlet as an HTTP daemon on :8123
#   3. print the qwen-code launch command with the audited profile
#      (docs/qwen-code-rl-audit.md §6) pointed at the daemon.
#
# Usage:
#   ./run_pie_qwen.sh [config.toml]        # default: portable Qwen3-0.6B config
#   PIE_PORT=18080 HTTP_PORT=8123 ./run_pie_qwen.sh
#
# The script stays in the foreground; Ctrl-C tears the stack down.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PIE="${PIE:-$REPO/target/release/pie}"
CFG="${1:-$REPO/integrations/qwen-code/pie_config.toml}"
PIE_PORT="${PIE_PORT:-18080}"
HTTP_PORT="${HTTP_PORT:-8123}"
LOG="${LOG:-/tmp/pie_qwen_serve.log}"

# pie_client lives in the openhands integration venv on dev machines;
# override with PYTHON=… if it's installed elsewhere.
PYTHON="${PYTHON:-$REPO/integrations/openhands/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

WASM="$REPO/inferlets/chat-completions/target/wasm32-wasip2/release/chat_completions.wasm"
[ -f "$WASM" ] || {
    echo "building chat-completions.wasm…"
    (cd "$REPO/inferlets/chat-completions" && cargo build --target wasm32-wasip2 --release)
}

echo "── pie serve :$PIE_PORT (config: $CFG, log: $LOG)"
"$PIE" serve --config "$CFG" --port "$PIE_PORT" --no-auth >"$LOG" 2>&1 &
PIE_PID=$!
trap 'kill $PIE_PID 2>/dev/null || true' EXIT

# Wait for the control plane to accept connections.
for _ in $(seq 1 60); do
    if nc -z 127.0.0.1 "$PIE_PORT" 2>/dev/null; then break; fi
    kill -0 $PIE_PID 2>/dev/null || { echo "pie serve died — see $LOG"; exit 1; }
    sleep 1
done

echo "── launching chat-completions daemon on :$HTTP_PORT"
"$PYTHON" "$REPO/integrations/qwen-code/launch_daemon.py" \
    --uri "ws://127.0.0.1:$PIE_PORT" --port "$HTTP_PORT" --wasm "$WASM"

cat <<EOF

── stack up. Point qwen-code at it with the audited profile:

QWEN_HOME=/tmp/qwen-home QWEN_RUNTIME_DIR=/tmp/qwen-runtime \\
QWEN_USAGE_STATISTICS_ENABLED=false QWEN_CODE_SKIP_UPDATE_CHECK_ONCE=1 \\
QWEN_CODE_DISABLE_PRECONNECT=1 QWEN_DISABLE_AUTO_TITLE=1 \\
QWEN_CODE_TOOL_CALL_STYLE=general \\
OPENAI_BASE_URL=http://127.0.0.1:$HTTP_PORT/v1 OPENAI_API_KEY=pie-local \\
OPENAI_MODEL=qwen3 \\
qwen --yolo --bare --safe-mode --auth-type openai \\
     --max-session-turns 60 --max-wall-time 30m --max-tool-calls 400 \\
     --chat-recording false -p "<instruction>"

(workspace .qwen/settings.json per docs/qwen-code-rl-audit.md §6)

Acceptance: python3 $REPO/integrations/qwen-code/test_acceptance.py
Ctrl-C stops pie serve.
EOF

wait $PIE_PID
