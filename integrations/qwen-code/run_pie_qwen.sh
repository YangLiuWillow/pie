#!/usr/bin/env bash
# Boot the full qwen-code ↔ Pie stack on the rewritten engine:
#   1. `pie serve` (standalone: embedded controller + gateway + worker)
#   2. shim.py — the OpenAI /v1/chat/completions surface, which installs and
#      launches the chat-completions inferlet over the gateway WS
#   3. print the qwen-code launch command with the audited profile
#      (docs/qwen-code-rl-audit.md §6) pointed at the shim.
#
# Usage:
#   ./run_pie_qwen.sh [config.toml]      # default: Metal Qwen3-0.6B-4bit config
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

# shim.py needs pie_client (client/python, needs `websockets`); the openhands
# venv has it on dev machines. Override with PYTHON=… if installed elsewhere.
PYTHON="${PYTHON:-$REPO/integrations/openhands/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

WASM="$REPO/tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm"
[ -f "$WASM" ] || {
    echo "building chat_completions.wasm…"
    (cd "$REPO/tests/inferlets" && cargo build --target wasm32-wasip2 --release -p chat-completions)
}

# The model must be imported (converted to a .zt artifact) before serve.
MODEL="$(sed -n 's/^model = "\(.*\)"/\1/p' "$CFG" | head -1)"
if ! "$PIE" --config "$CFG" model list 2>/dev/null | grep -q "$(basename "$MODEL")"; then
    echo "── importing $MODEL (first run only)"
    "$PIE" --config "$CFG" model import "$MODEL"
fi

echo "── pie serve :$PIE_PORT (config: $CFG, log: $LOG)"
"$PIE" --config "$CFG" serve >"$LOG" 2>&1 &
PIE_PID=$!
trap 'kill $PIE_PID 2>/dev/null || true' EXIT

# Wait for the gateway to accept connections.
for _ in $(seq 1 120); do
    if nc -z 127.0.0.1 "$PIE_PORT" 2>/dev/null; then break; fi
    kill -0 $PIE_PID 2>/dev/null || { echo "pie serve died — see $LOG"; exit 1; }
    sleep 1
done

echo "── shim on :$HTTP_PORT (installs + launches the inferlet)"
"$PYTHON" "$REPO/integrations/qwen-code/shim.py" \
    --pie "ws://127.0.0.1:$PIE_PORT" --port "$HTTP_PORT" \
    --wasm "$WASM" --manifest "$REPO/tests/inferlets/chat-completions/Pie.toml" &
SHIM_PID=$!
trap 'kill $SHIM_PID $PIE_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    if nc -z 127.0.0.1 "$HTTP_PORT" 2>/dev/null; then break; fi
    kill -0 $SHIM_PID 2>/dev/null || { echo "shim died"; exit 1; }
    sleep 1
done

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

Acceptance: python3 $REPO/integrations/qwen-code/test_acceptance.py --base http://127.0.0.1:$HTTP_PORT
Ctrl-C stops the stack.
EOF

wait $PIE_PID
