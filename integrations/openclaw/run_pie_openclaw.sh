#!/usr/bin/env bash
# Boot `pie serve` with the OpenClaw-facing OpenAI surface, wait for /health,
# run the acceptance suite, report, and shut the server down.
#
# Usage:
#   ./run_pie_openclaw.sh [config.toml]      # default: trimmed profile
#                                            #   generated under mktemp
#   ./run_pie_openclaw.sh --serve-only       # boot + wait, skip the tests
#                                            #   (for the stock-OpenClaw e2e)
#
# Environment: same as ../opencode/run_pie_opencode.sh (PIE_BIN, PIE_HOME,
# PIE_PORT, LOG, PIE_METAL_ROW_BUDGET_MB — see its header for the RAM /
# Metal-admission notes; they apply here with MORE headroom needed).
#
# Differences from the opencode profile, both driven by oc-P0.3's parity
# measurement (openclaw fixtures render at ~24.2k tokens full-surface):
#   - max_model_len 32768 (= total_pages 1024 × kv_page_size 32), double
#     the opencode profile — a longer-than-max prompt is refused, not
#     chunked, and req-003 replays at ~24k tokens;
#   - the acceptance suite is ./test_acceptance.py (openclaw fixtures +
#     OpenClaw client policy: empty-delta keepalives, mapped finish_reason
#     set, max_completion_tokens, content-part tolerance).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PIE_BIN="${PIE_BIN:-$REPO/../pie/target/release/pie}"
PIE_HOME="${PIE_HOME:-$HOME/.pie}"
PIE_PORT="${PIE_PORT:-8080}"
BASE_URL="${PIE_BASE_URL:-http://127.0.0.1:$PIE_PORT}"
LOG="${LOG:-/tmp/pie_openclaw_serve.log}"

SERVE_ONLY=0
CFG=""
for arg in "$@"; do
    case "$arg" in
        --serve-only) SERVE_ONLY=1 ;;
        *) CFG="$arg" ;;
    esac
done

[ -x "$PIE_BIN" ] || {
    echo "no pie binary at $PIE_BIN (set PIE_BIN); NOT building — the release" >&2
    echo "build lives in the shared target dir <workspace>/pie/target" >&2
    exit 1
}

# ── config: argument, or a generated trimmed profile ─────────────────────────
if [ -z "$CFG" ]; then
    CFG="$(mktemp -d -t pie_openclaw)/config.toml"
    cat >"$CFG" <<EOF
[server]
host = "127.0.0.1"
port = $PIE_PORT

[model]
name = "default"
model = "Qwen--Qwen3-0.6B-optimized"

[driver]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
kv_page_size = 32
total_pages = 1024
max_forward_tokens = 2048
max_forward_requests = 32
max_model_len = 32768

[runtime]
request_timeout = "120s"

[sandbox]
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]
EOF
    echo "── generated config: $CFG"
fi

# ── refresh the chat-completions inferlet in \$PIE_HOME/programs ─────────────
WASM_SRC="$REPO/../pie/target/wasm32-wasip2/release/chat_completions.wasm"
MANIFEST_SRC="$REPO/inferlets/chat-completions/Pie.toml"
PROG_DIR="$PIE_HOME/programs/chat-completions"
if [ -f "$WASM_SRC" ]; then
    if [ ! -f "$PROG_DIR/0.1.0.wasm" ] || [ "$WASM_SRC" -nt "$PROG_DIR/0.1.0.wasm" ]; then
        mkdir -p "$PROG_DIR"
        cp "$WASM_SRC" "$PROG_DIR/0.1.0.wasm"
        cp "$MANIFEST_SRC" "$PROG_DIR/0.1.0.toml"
        echo "── refreshed $PROG_DIR/0.1.0.{wasm,toml} from the shared target dir"
    fi
else
    echo "── warning: no built wasm at $WASM_SRC — using whatever is installed" >&2
fi

# ── serverless preflight ─────────────────────────────────────────────────────
echo "── pie doctor (config parse + driver preflight)"
"$PIE_BIN" -c "$CFG" doctor || {
    echo "doctor says this machine/config cannot boot — see above" >&2
    exit 1
}

# ── boot ─────────────────────────────────────────────────────────────────────
echo "── pie serve on :$PIE_PORT (config: $CFG, log: $LOG)"
"$PIE_BIN" -c "$CFG" serve >"$LOG" 2>&1 &
PIE_PID=$!
cleanup() {
    if kill -0 "$PIE_PID" 2>/dev/null; then
        kill "$PIE_PID" 2>/dev/null || true
        wait "$PIE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo -n "── waiting for $BASE_URL/health "
UP=0
for _ in $(seq 1 180); do
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/health"; then
        UP=1
        break
    fi
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo
        echo "pie serve exited during startup — last log lines:" >&2
        tail -n 30 "$LOG" >&2
        echo "(RAM/Metal admission? Try PIE_METAL_ROW_BUDGET_MB=512, or free memory.)" >&2
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo
[ "$UP" = 1 ] || { echo "server never became healthy — see $LOG" >&2; exit 1; }
echo "── up."

if [ "$SERVE_ONLY" = 1 ]; then
    cat <<EOF

── serve-only mode. Point stock OpenClaw at it (see ./README.md):

     OPENCLAW_CONFIG_PATH=$HERE/openclaw.json \\
       openclaw agent exec "read the file $HERE/README.md and summarize it" \\
       --model pie/qwen3-0.6b --json

   Acceptance, separately: python3 $HERE/test_acceptance.py
   Ctrl-C stops pie serve.
EOF
    wait "$PIE_PID"
    exit 0
fi

# ── acceptance ───────────────────────────────────────────────────────────────
echo "── running acceptance suite"
RC=0
PIE_BASE_URL="$BASE_URL" python3 "$HERE/test_acceptance.py" || RC=$?

echo "── shutting down pie serve (pid $PIE_PID)"
cleanup
trap - EXIT INT TERM

if [ "$RC" = 0 ]; then
    echo "── GREEN: all hard acceptance checks passed."
else
    echo "── acceptance failed (exit $RC) — serve log: $LOG"
fi
exit "$RC"
