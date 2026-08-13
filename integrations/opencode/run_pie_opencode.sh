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
#   PIE_STRATEGY=b ./run_pie_opencode.sh                     # session inferlet
#
# ── The two strategies, and why one script serves both ──────────────────────
#
#   a (default)  stock opencode → gateway OpenAI ingress → one `chat-completions`
#                inferlet PER REQUEST. KV dies with the request. Frozen: this is
#                the control arm for the A/B, so nothing here may change its
#                behaviour.
#   b            stock opencode → session_shim.py → ONE long-lived
#                `opencode-session` inferlet over the sticky WebSocket, holding
#                the conversation's KV working set across turns.
#
# The arms are deliberately indistinguishable from the client's side: in BOTH,
# opencode and the acceptance suite talk to http://127.0.0.1:$PIE_PORT/v1. Under
# `b` the shim binds that port and `pie serve` moves to $PIE_ENGINE_PORT. So
# ./opencode.json and test_acceptance.py need no per-arm configuration, and a
# difference in the measurement cannot be a difference in how the client was
# pointed. (qwen-code's run-2 lost a benchmark exactly this way — the arms
# turned out to be measuring their renderers, not their servers.)
#
# Environment:
#   PIE_STRATEGY  a | b                  (default: a — see above)
#   PIE_ENGINE_PORT
#                 gateway port under strategy b (default: 18080). Under `a` the
#                 gateway is on PIE_PORT directly and this is unused.
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

# ── strategy ────────────────────────────────────────────────────────────────
# Under `b` the shim owns the client-facing port and the gateway moves aside, so
# both arms answer on $PIE_PORT. See the header.
PIE_STRATEGY="${PIE_STRATEGY:-a}"
case "$PIE_STRATEGY" in
    a|A) PIE_STRATEGY=a ;;
    b|B) PIE_STRATEGY=b ;;
    *) echo "PIE_STRATEGY must be 'a' or 'b' (got '$PIE_STRATEGY')" >&2; exit 1 ;;
esac
PIE_ENGINE_PORT="${PIE_ENGINE_PORT:-18080}"
# The shim imports the pie python client (websockets/msgpack/blake3/cryptography);
# point this at a venv that has them if the system python3 does not.
PIE_PYTHON="${PIE_PYTHON:-python3}"
if [ "$PIE_STRATEGY" = b ]; then
    SERVE_PORT="$PIE_ENGINE_PORT"
    HEALTH_URL="http://127.0.0.1:$SERVE_PORT"
    SHIM_LOG="${SHIM_LOG:-/tmp/pie_opencode_shim.log}"
else
    SERVE_PORT="$PIE_PORT"
    HEALTH_URL="$BASE_URL"
fi

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
# ── driver sizing ────────────────────────────────────────────────────────────
# These are matched to what vllm-metal actually boots with, because a ratio
# against a differently-configured baseline measures the configuration:
#
#   vLLM: max_num_batched_tokens=2048, KV pool 193,536 tokens, dtype bfloat16
#   pie:  max_forward_tokens=2048,     KV pool total_pages*kv_page_size
#
# `total_pages = 512` (16,384 tokens) was this repo's inherited default and it
# is a KV-STARVED setting: raising it to 2048 is worth **1.73x on prefill** at
# no memory pressure on this box (measured, `results-prefill-profile.md`). It
# was never memory-forced — the activation pool sat at 24 MB of a 1024 MB
# budget. Lower it again only if a model will not admit.
PIE_TOTAL_PAGES="${PIE_TOTAL_PAGES:-2048}"
# Context ceiling. A prompt longer than this is REFUSED by the Metal driver,
# not chunked, so an agent benchmark that explores a real repo needs headroom
# that a chat replay does not.
PIE_MAX_MODEL_LEN="${PIE_MAX_MODEL_LEN:-16384}"
PIE_KV_PAGE_SIZE="${PIE_KV_PAGE_SIZE:-32}"
PIE_MAX_FORWARD_TOKENS="${PIE_MAX_FORWARD_TOKENS:-2048}"

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
port = $SERVE_PORT

[model]
name = "default"
model = "$ARTIFACT"

[driver]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
kv_page_size = $PIE_KV_PAGE_SIZE
total_pages = $PIE_TOTAL_PAGES
max_forward_tokens = $PIE_MAX_FORWARD_TOKENS
max_forward_requests = 8
max_model_len = $PIE_MAX_MODEL_LEN

[runtime]
request_timeout = "300s"
# The engine kills an inferlet that has been silent for this long, and under
# strategy b that kill takes the gateway WebSocket down WITH it — the shim sees
# `terminal event error: 'WebSocket connection closed'`, every in-flight turn
# 500s, and the whole retained working set dies with the process. A 30 s default
# is far inside a single 7k-token prefill on a 30B (measured: 44 s to first
# content), so a long agentic turn trips it routinely.
silence_timeout = "300s"

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

# Strategy b resolves its own guest the same way, but does NOT install it here:
# the shim pushes it over the WebSocket with `install_program`, which is
# atomic server-side and needs no $PIE_HOME write from this script.
if [ "$PIE_STRATEGY" = b ]; then
    for CAND in \
        "${CARGO_TARGET_DIR:-}/wasm32-wasip2/release/opencode_session.wasm" \
        "$REPO/inferlets/opencode-session/target/wasm32-wasip2/release/opencode_session.wasm"; do
        if [ -n "${CAND#/wasm32*}" ] && [ -f "$CAND" ]; then SESSION_WASM="$CAND"; break; fi
    done
    SESSION_WASM="${SESSION_WASM:-}"
    if [ -z "$SESSION_WASM" ]; then
        echo "no built opencode-session wasm found. Build it:" >&2
        echo "  (cd $REPO/inferlets/opencode-session && cargo build --target wasm32-wasip2 --release)" >&2
        exit 1
    fi
    SESSION_MANIFEST="$REPO/inferlets/opencode-session/Pie.toml"
    # KV residency budget for the guest, derived from the config THIS script
    # generated: total_pages * kv_page_size is the whole pool, and the live
    # turn needs its own scratch inside it, so hand over about half. The guest
    # cannot work this out for itself — no pool capacity is reported on the
    # pie:inferlet surface — and over-committing kills the process rather than
    # evicting.
    PIE_RETAIN_TOKENS="${PIE_RETAIN_TOKENS:-$(( PIE_TOTAL_PAGES * PIE_KV_PAGE_SIZE / 2 ))}"
    echo "── strategy b: session inferlet $SESSION_WASM ($(wc -c <"$SESSION_WASM" | tr -d ' ') bytes), retain_tokens=$PIE_RETAIN_TOKENS"
fi

if [ "$PIE_STRATEGY" = a ] && [ -f "$WASM_SRC" ]; then
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
elif [ "$PIE_STRATEGY" = a ]; then
    echo "── warning: no built wasm at $WASM_SRC — using whatever is installed" >&2
fi

# ── serverless preflight ─────────────────────────────────────────────────────
echo "── pie doctor (config parse + driver preflight)"
"$PIE_BIN" -c "$CFG" doctor || {
    echo "doctor says this machine/config cannot boot — see above" >&2
    exit 1
}

# ── boot ─────────────────────────────────────────────────────────────────────
echo "── pie serve on :$SERVE_PORT (config: $CFG, log: $LOG)"
"$PIE_BIN" -c "$CFG" serve >"$LOG" 2>&1 &
PIE_PID=$!
SHIM_PID=""
cleanup() {
    # Shim first: it holds the WebSocket, and tearing the engine out from under
    # it just produces a relaunch storm in its log on the way down.
    if [ -n "$SHIM_PID" ] && kill -0 "$SHIM_PID" 2>/dev/null; then
        kill "$SHIM_PID" 2>/dev/null || true
        wait "$SHIM_PID" 2>/dev/null || true
    fi
    if kill -0 "$PIE_PID" 2>/dev/null; then
        # SIGTERM, never -9: a hard kill mid-fire is what actually leaves a
        # wedged Metal context behind.
        kill "$PIE_PID" 2>/dev/null || true
        wait "$PIE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Model load + Metal heap admission can take a while; 180s budget.
echo -n "── waiting for $HEALTH_URL/health "
UP=0
for _ in $(seq 1 180); do
    if curl -sf -o /dev/null --max-time 2 "$HEALTH_URL/health"; then
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

# ── strategy b: the session shim owns the client-facing port ────────────────
if [ "$PIE_STRATEGY" = b ]; then
    echo "── session shim on :$PIE_PORT -> ws://127.0.0.1:$SERVE_PORT (log: $SHIM_LOG)"
    "$PIE_PYTHON" "$HERE/session_shim.py" \
        --pie "ws://127.0.0.1:$SERVE_PORT" \
        --host 127.0.0.1 --port "$PIE_PORT" \
        --wasm "$SESSION_WASM" --manifest "$SESSION_MANIFEST" \
        --model-name "$PIE_MODEL" \
        --retain-tokens "$PIE_RETAIN_TOKENS" \
        >"$SHIM_LOG" 2>&1 &
    SHIM_PID=$!
    echo -n "── waiting for $BASE_URL/v1/models "
    SHIM_UP=0
    for _ in $(seq 1 60); do
        if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
            SHIM_UP=1
            break
        fi
        if ! kill -0 "$SHIM_PID" 2>/dev/null; then
            echo
            echo "session shim exited during startup — last log lines:" >&2
            tail -n 30 "$SHIM_LOG" >&2
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    echo
    [ "$SHIM_UP" = 1 ] || { echo "shim never came up — see $SHIM_LOG" >&2; exit 1; }
    echo "── shim up (session inferlet launched)."
fi

if [ "$SERVE_ONLY" = 1 ]; then
    cat <<EOF

── serve-only mode (strategy $PIE_STRATEGY). Point stock opencode at it:

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
