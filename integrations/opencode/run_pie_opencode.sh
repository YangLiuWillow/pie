#!/usr/bin/env bash
# Boot `pie serve` with the opencode-facing OpenAI surface, wait for /health,
# run the acceptance suite, report, and shut the server down.
#
# Usage:
#   ./run_pie_opencode.sh [config.toml]      # default: trimmed profile
#                                            #   generated under mktemp
#   ./run_pie_opencode.sh --serve-only       # boot + wait, skip the tests
#                                            #   (for the stock-opencode e2e)
#
#   PIE_MODEL=qwen3.6-35b-a3b ./run_pie_opencode.sh          # the 35B run
#
# Environment:
#   PIE_MODEL     which model to serve   (default: qwen3-0.6b). One of the
#                                        profile keys below, or a raw artifact
#                                        name from `pie model list` — a raw
#                                        name gets the shared driver settings
#                                        and whatever RAM it needs.
#   PIE_BIN       pie binary            (default: <repo>/../pie/target/release/pie
#                                        — the shared-target-dir release build,
#                                        see the 2026-08-11 environment note in
#                                        docs/opencode-integration-progress.md)
#   PIE_HOME      pie home              (default: ~/.pie; must hold the model
#                                        artifact and the chat-completions
#                                        inferlet — this script refreshes the
#                                        inferlet from the shared target dir
#                                        when it is newer)
#   PIE_PORT      gateway port          (default: 8080, matching [server].port
#                                        and the baseURL in ./opencode.json)
#   LOG           serve file            (default: /tmp/pie_opencode_serve.log)
#   PIE_METAL_ROW_BUDGET_MB
#                 Metal activation-row reservation in MB (driver default
#                 1024 = 1 GB, read in driver/metal/src/context.cpp
#                 row_budget_bytes()). The reservation is admission-relevant
#                 on a RAM-squeezed machine: lowering it (e.g. 512) shrinks
#                 what Metal must admit, at the cost of the longest prompt the
#                 driver will ACCEPT (a too-long prompt is refused, not
#                 chunked). opencode's build-agent prompt is ~7.5k tokens —
#                 don't go so low that it gets refused. Pass-through only;
#                 unset means the driver default.
#
# Memory, on a shared machine (2026-08-12): run ONE `pie serve` at a time and
# stop it with SIGTERM. The Metal driver's admission warning blames wired pages
# on abandoned GPU contexts "cleared only by reboot", but the identical warning
# appears when another `pie serve` simply holds its heap — that was the real
# cause here, and a clean SIGTERM to it took wired from 24.17 GiB to 2.85 GiB
# with no reboot. Check for a second `pie serve` before believing the leak
# reading. The hard-kill hazard is real, but it comes from `kill -9` mid-fire,
# not from running the thing.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# Same CARGO_TARGET_DIR reasoning as the wasm lookup below.
if [ -z "${PIE_BIN:-}" ]; then
    for CAND in "${CARGO_TARGET_DIR:-}/release/pie" "$REPO/target/release/pie" \
                "$REPO/../pie/target/release/pie"; do
        if [ -n "${CAND#/release*}" ] && [ -x "$CAND" ]; then PIE_BIN="$CAND"; break; fi
    done
fi
PIE_BIN="${PIE_BIN:-$REPO/../pie/target/release/pie}"
PIE_HOME="${PIE_HOME:-$HOME/.pie}"
PIE_PORT="${PIE_PORT:-8080}"
BASE_URL="${PIE_BASE_URL:-http://127.0.0.1:$PIE_PORT}"
LOG="${LOG:-/tmp/pie_opencode_serve.log}"

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
    echo "build lives in the shared target dir …/Liszt_ai/pie/target" >&2
    exit 1
}

# ── model profile ────────────────────────────────────────────────────────────
# Friendly name → stored artifact. The keys match the model ids in
# ./opencode.json, so `PIE_MODEL=x` and `opencode run -m pie/x` name the same
# thing. A value that matches no key is passed through as a raw artifact name
# (see `pie model list`).
PIE_MODEL="${PIE_MODEL:-qwen3-0.6b}"
case "$PIE_MODEL" in
    qwen3-0.6b)       ARTIFACT="Qwen--Qwen3-0.6B-optimized" ;;
    qwen3.6-35b-a3b)  ARTIFACT="mlx-community--Qwen3.6-35B-A3B-4bit" ;;
    *)                ARTIFACT="$PIE_MODEL" ;;
esac

# ── config: argument, or a generated trimmed profile ─────────────────────────
# One driver shape for every model here — these are the settings the 2026-08-12
# results were produced under (`integrations/opencode/results-*.md`), for both
# the 0.6B and the 35B, so a re-run reproduces them rather than approximating.
#
#   max_model_len 16384  opencode's build-agent prompt alone is ~7.5k tokens
#                        (req-005 replays at 7473) and a longer-than-max prompt
#                        is REFUSED by the Metal driver, not chunked. 16384 =
#                        total_pages(512) × kv_page_size(32).
#                        NOT a ceiling: 32768 (total_pages 1024) booted this
#                        model fine on a quiet machine. Admission is
#                        `want + min(transient,2GiB) + 2GiB > reclaimable`
#                        (driver/metal/src/batch/forward.cpp:892), so it is a
#                        function of what else is resident. Measured wants on
#                        this 35B: 24.77 GiB at 32768/1024pages/32reqs,
#                        22.55 GiB at 16384/512pages/8reqs. A client whose
#                        prompts exceed 16k (OpenClaw's full surface replays
#                        at ~24.2k) should raise both and check admission on a
#                        quiet machine rather than assume the lower value.
#   max_forward_*        1024/8 rather than the driver's roomier defaults: the
#                        35B needs ~22.6 GiB resident at these numbers, and a
#                        48 GB machine with anything else running cannot admit
#                        more. The 0.6B could afford far more and does not care,
#                        and holding both at one shape keeps the two runs
#                        comparable.
#   request_timeout      300s, not 120s: a 35B streaming turn with a large
#                        max_tokens outruns two minutes on Metal.
if [ -z "$CFG" ]; then
    CFG="$(mktemp -d -t pie_opencode)/config.toml"
    cat >"$CFG" <<EOF
[server]
host = "127.0.0.1"
port = $PIE_PORT

[model]
name = "default"
model = "$ARTIFACT"

[driver]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
kv_page_size = 32
total_pages = 512
max_forward_tokens = 1024
max_forward_requests = 8
max_model_len = 16384

[runtime]
request_timeout = "300s"

[sandbox]
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]
EOF
    echo "── generated config: $CFG (model $PIE_MODEL -> $ARTIFACT)"
fi

# ── refresh the chat-completions inferlet in \$PIE_HOME/programs ─────────────
# The gateway launches the serving inferlet by name; the engine resolves it
# from \$PIE_HOME/programs/<name>/<version>.wasm + <version>.toml.
# Honor CARGO_TARGET_DIR — sibling worktrees (pie-openclaw, …) carry crates
# with IDENTICAL package names but different content, so a target dir shared
# across them collides; a per-worktree dir is the safe default. Falls back to
# the crate-local target, then the old shared path.
for CAND in \
    "${CARGO_TARGET_DIR:-}/wasm32-wasip2/release/chat_completions.wasm" \
    "$REPO/inferlets/chat-completions/target/wasm32-wasip2/release/chat_completions.wasm" \
    "$REPO/../pie/target/wasm32-wasip2/release/chat_completions.wasm"; do
    if [ -n "${CAND#/wasm32*}" ] && [ -f "$CAND" ]; then WASM_SRC="$CAND"; break; fi
done
WASM_SRC="${WASM_SRC:-$REPO/inferlets/chat-completions/target/wasm32-wasip2/release/chat_completions.wasm}"
MANIFEST_SRC="$REPO/inferlets/chat-completions/Pie.toml"
PROG_DIR="$PIE_HOME/programs/chat-completions"
if [ -f "$WASM_SRC" ]; then
    if [ ! -f "$PROG_DIR/0.1.0.wasm" ] || [ "$WASM_SRC" -nt "$PROG_DIR/0.1.0.wasm" ]; then
        mkdir -p "$PROG_DIR"
        # Copy to a temp name and rename. `cp` straight onto the destination
        # is not atomic, and the source is a build output: a `cargo build` in
        # another terminal can be mid-write when this fires, and the
        # wasm32-wasip2 target's componentisation step leaves a LARGER
        # intermediate on disk before the final artifact. Copying that
        # intermediate installs a module the engine cannot run — and it does
        # not fail cleanly. It HANGS: the launch never acks, nothing is
        # logged, `/health` keeps answering 200, and every completion request
        # blocks forever. Cost an hour here, twice, before the byte size gave
        # it away (823132 installed vs 638350 for a real build).
        TMP_WASM="$PROG_DIR/.0.1.0.wasm.$$"
        cp "$WASM_SRC" "$TMP_WASM"
        # Sanity-check the magic before it can wedge an engine: every
        # wasm module/component starts "\0asm".
        if [ "$(head -c 4 "$TMP_WASM" | od -An -c | tr -d ' \n')" != "\0asm" ]; then
            rm -f "$TMP_WASM"
            echo "── refusing to install $WASM_SRC: not a wasm module (build in progress?)" >&2
            exit 1
        fi
        mv -f "$TMP_WASM" "$PROG_DIR/0.1.0.wasm"
        cp "$MANIFEST_SRC" "$PROG_DIR/0.1.0.toml"
        echo "── refreshed $PROG_DIR/0.1.0.{wasm,toml} from $(dirname "$WASM_SRC")" \
             "($(wc -c <"$PROG_DIR/0.1.0.wasm" | tr -d ' ') bytes)"
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

# Model load + Metal heap admission can take a while; 180s budget.
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

── serve-only mode. Point stock opencode at it:

     cd $HERE && opencode run -m pie/$PIE_MODEL "read the file $HERE/README.md and summarize it"

   (the ./opencode.json in this directory defines provider "pie" at
    $BASE_URL/v1; opencode picks it up from the cwd)

   Acceptance, separately: PIE_BASE_URL=$BASE_URL python3 $HERE/test_acceptance.py
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
