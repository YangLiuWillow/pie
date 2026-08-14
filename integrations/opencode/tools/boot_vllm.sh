#!/usr/bin/env bash
# Boot ONE vllm-metal server and refuse to return unless THIS boot is the one
# serving on :8000, under the name opencode actually asks for.
#
# The sibling of boot_pie.sh, and it exists for the same reason: the SWE-bench
# comparison is only fair if BOTH arms restart per instance (results-swebench.md
# §"the arms were not symmetric"), and a restart you cannot prove happened is
# worse than no restart at all -- a stale server keeps answering /health, and
# both arms of an A/B silently run against one process. That already happened
# once here, to pie.
#
# Three things are proven before this exits 0, not assumed:
#
#   1. Nothing from a previous boot survived.  Killing the launcher is not
#      enough: vLLM's V1 engine runs a separate EngineCore process holding the
#      ~17 GB of weights, and an orphaned one keeps the memory and can keep the
#      port. So the server is started in its own SESSION (via python's setsid --
#      macOS has no setsid(1)) and the whole process group is signalled.
#   2. The process listening on the port is the one this script started, by PID
#      and process group -- not merely "something answers".
#   3. The server answers to the model id opencode sends.  opencode reads
#      `qwen3-coder-30b` from opencode.json; a server that only knows
#      `coder30b` fails every request as opencode's generic
#      `{"name":"UnknownError","message":"Unexpected server error"}`, naming
#      nothing.  --served-model-name takes a list, so both names are served and
#      bench_ab.py / check_render_vllm.py keep working unchanged.
#
# Usage:
#   tools/boot_vllm.sh <tag> [extra vllm serve flags ...]
#
# Extra flags are appended, so they override the defaults below (argparse takes
# the last occurrence).  Env knobs: VLLM_BIN VLLM_MODEL VLLM_PORT
# VLLM_MAX_MODEL_LEN VLLM_SERVED_NAME VLLM_BOOT_TIMEOUT VLLM_KILL_PIE.
set -euo pipefail

TAG="${1:?usage: boot_vllm.sh <tag> [extra vllm serve flags ...]}"; shift

VLLM="${VLLM_BIN:-$HOME/.venv-vllm-metal/bin/vllm}"
MODEL="${VLLM_MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit}"
PORT="${VLLM_PORT:-8000}"
# 32768 is the AGENT arm's context, matched to pie's PIE_MAX_MODEL_LEN.  16384
# is the A/B bench's number and is not enough for opencode, which declares a
# 32768 context for this model and gets an over-long prompt REFUSED, not
# chunked.  If this will not fit, lower BOTH arms -- see results-swebench.md.
MAXLEN="${VLLM_MAX_MODEL_LEN:-32768}"
SERVED="${VLLM_SERVED_NAME:-qwen3-coder-30b}"
BOOT_TIMEOUT="${VLLM_BOOT_TIMEOUT:-900}"

LOG="/tmp/vllm_${TAG}.log"
PIDFILE="/tmp/vllm_${TAG}.pid"

# Matches the launcher (argv contains the venv path) AND the engine core, whose
# proctitle is `VLLM::EngineCore` and contains no path at all.  Deliberately
# scoped to this venv, and deliberately NOT the bare string `vllm`, which would
# match this script's own argv and make it kill itself.
PAT='(\.venv-vllm-metal/bin/vllm|VLLM::)'

die() { echo "FATAL: $*" >&2; exit 1; }

# pgrep, minus this script and its children, and never failing the `set -e`.
live_pids() {
    pgrep -f "$PAT" 2>/dev/null | grep -vx "$$" || true
}

[ -x "$VLLM" ] || die "no vllm binary at $VLLM (set VLLM_BIN)"

# ── One 30B at a time ────────────────────────────────────────────────────
# ~17 GB of weights plus KV, on a 48 GB box.  Booting vLLM on top of a live pie
# does not fail cleanly; it swaps, and every latency number taken afterwards is
# fiction.  Refuse loudly instead.
if pgrep -f "release/pie -c" >/dev/null 2>&1; then
    if [ "${VLLM_KILL_PIE:-0}" = "1" ]; then
        echo "killing a live pie server (VLLM_KILL_PIE=1)"
        pkill -f "release/pie -c" 2>/dev/null || true
        for _ in $(seq 1 30); do
            pgrep -f "release/pie -c" >/dev/null 2>&1 || break
            sleep 1
        done
    else
        echo "FATAL: a pie server is live; the box does not hold two 30B servers" >&2
        pgrep -fl "release/pie -c" >&2
        die "kill it, or re-run with VLLM_KILL_PIE=1"
    fi
fi

# ── 1. Nothing from a previous boot survives ─────────────────────────────
# The port holder's whole process group goes first: that is what catches an
# EngineCore whose parent launcher already died.
holder="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
if [ -n "$holder" ]; then
    hpgid="$(ps -o pgid= -p "$holder" 2>/dev/null | tr -d ' ' || true)"
    [ -n "$hpgid" ] && { kill -TERM -- "-$hpgid" 2>/dev/null || true; }
fi
pkill -f "$PAT" 2>/dev/null || true

for _ in $(seq 1 40); do
    [ -z "$(live_pids)" ] && ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1 && break
    sleep 1
done
if [ -n "$(live_pids)" ]; then
    echo "TERM was not enough; escalating to KILL" >&2
    # shellcheck disable=SC2046
    kill -KILL $(live_pids) 2>/dev/null || true
    sleep 3
fi
[ -z "$(live_pids)" ] || { pgrep -fl "$PAT" >&2; die "a vllm process survived the kill"; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
    die ":$PORT still held by something else"
fi

# ── speculative decoding parity ─────────────────────────────────────────
# vLLM 0.27 ships `ngram` speculation, which is the SAME algorithm the
# `opencode-session` inferlet implements in `draft.rs`: look up where the recent
# output occurred earlier in the context and copy what followed. So the two
# stacks can run like for like, and the parameters map one to one:
#
#   pie draft.rs                     vLLM speculative-config
#   DRAFT_K            = 4           num_speculative_tokens = 4
#   NGRAM_LONG         = 3           prompt_lookup_max      = 3
#   NGRAM_SHORT        = 2           prompt_lookup_min      = 2
#
# Set VLLM_SPEC_ARG to the JSON below to enable it. Leave it unset for the
# no-speculation arm. Enabling it on ONE side only would measure the feature
# rather than the engines -- the same mistake as the prefix-cache comparison,
# where vLLM had APC and pie's arm A did not.
#
#   VLLM_SPEC_ARG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'
#
# Worth knowing before quoting any result: on pie this technique removed 1.85x
# of the decode fires and bought 1.04x of wall clock, because a k-row verify
# fire there costs about k times a 1-row fire instead of sharing one KV read.
# If vLLM's verify amortizes where pie's does not, that difference -- not the
# drafting -- is what a spec-on comparison measures.

# ── 2. Boot, in its own session ──────────────────────────────────────────
# os.setsid() makes the child a session leader, so its PGID == its PID and the
# whole tree -- launcher, API workers, EngineCore -- can be signalled as one
# group by the next boot.  macOS ships no setsid(1); this is the portable
# equivalent.  execvp keeps the PID, so $! below is the group we recorded.
: > "$LOG"
nohup /usr/bin/python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "$VLLM" serve "$MODEL" \
    --port "$PORT" \
    --served-model-name "$SERVED" coder30b \
    --max-model-len "$MAXLEN" \
    --enable-prefix-caching \
    ${VLLM_SPEC_ARG:+--speculative-config "$VLLM_SPEC_ARG"} \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    "$@" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "booting $TAG: pid $PID, model $MODEL, max-model-len $MAXLEN, log $LOG"

t0=$(date +%s)
deadline=$(( t0 + BOOT_TIMEOUT ))
while :; do
    now=$(date +%s)
    [ "$now" -lt "$deadline" ] || { tail -30 "$LOG" >&2; die "$TAG never came up in ${BOOT_TIMEOUT}s"; }

    if ! kill -0 "$PID" 2>/dev/null; then
        tail -30 "$LOG" >&2
        die "the server exited during boot (see $LOG)"
    fi
    if grep -qiE "address already in use|EADDRINUSE" "$LOG" 2>/dev/null; then
        die "this boot lost the bind on :$PORT"
    fi

    models="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [ -n "$models" ] && printf '%s' "$models" | grep -q '"id"'; then
        # 2. Identity: the listener must BE us.  `something answers` is exactly
        #    the evidence that misled four measurements on the pie side.
        lp="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
        lpgid="$(ps -o pgid= -p "$lp" 2>/dev/null | tr -d ' ' || true)"
        if [ "$lp" != "$PID" ] && [ "$lpgid" != "$PID" ]; then
            echo "listener on :$PORT is pid $lp (pgid $lpgid), not this boot's $PID" >&2
            ps -o pid=,pgid=,command= -p "$lp" >&2 || true
            die "someone else owns the port"
        fi
        # 3. And it answers to the name opencode sends.
        printf '%s' "$models" | grep -q "\"$SERVED\"" \
            || { printf '%s\n' "$models" >&2; die "server does not serve '$SERVED'"; }
        echo "up: $TAG serving '$SERVED' as pid $PID in $(( now - t0 ))s"
        exit 0
    fi
    sleep 3
done
