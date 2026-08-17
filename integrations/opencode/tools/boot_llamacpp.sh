#!/usr/bin/env bash
# Boot ONE llama-server and refuse to return unless THIS boot is the one
# serving, under the name opencode actually asks for.
#
# The third sibling of boot_pie.sh and boot_vllm.sh, and it exists for their
# reason: an arm that cannot prove it restarted is worse than an arm that never
# restarts, because a stale server keeps answering /v1/models and both halves of
# a comparison silently run against one process. That has already happened once
# to pie in this repo.
#
# Three llama.cpp-specific things are set deliberately, and all three change the
# numbers if left at their defaults:
#
#   -np 1        llama.cpp DIVIDES --ctx-size among its slots, and the default
#                (-1, auto) picks more than one. A server booted `-c 65536`
#                with four slots gives each request 16384 and REFUSES
#                opencode's longer prompts -- which looks like an engine that
#                cannot hold context, not a flag.
#   -ngl 999     every layer on the GPU. The default leaves some on the CPU,
#                which measures the CPU.
#   --chat-template-kwargs {"enable_thinking":false}
#                Qwen3.6 is a THINKING model whose template prefills `<think>`
#                unless told otherwise. pie's renderer hard-codes the
#                no-think cue, so an unset flag here would have llama.cpp
#                answering a different prompt than pie -- a trajectory
#                difference reported as an engine difference. All four arms
#                are pinned to no-think; see results-qwen36-four-way.md.
#
# Usage:
#   tools/boot_llamacpp.sh <tag> [extra llama-server flags ...]
#
# Env knobs: LCPP_BIN LCPP_GGUF LCPP_PORT LCPP_CTX LCPP_SERVED LCPP_BOOT_TIMEOUT
set -euo pipefail

TAG="${1:?usage: boot_llamacpp.sh <tag> [extra llama-server flags ...]}"; shift

LCPP="${LCPP_BIN:-$HOME/src/llama.cpp/build/bin/llama-server}"
GGUF="${LCPP_GGUF:-$HOME/models/qwen36-gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
PORT="${LCPP_PORT:-8002}"
CTX="${LCPP_CTX:-65536}"
SERVED="${LCPP_SERVED:-qwen3.6-35b-a3b}"
BOOT_TIMEOUT="${LCPP_BOOT_TIMEOUT:-900}"

LOG="/tmp/llamacpp_${TAG}.log"
PIDFILE="/tmp/llamacpp_${TAG}.pid"

# Scoped to THIS binary, not the bare string `llama-server`: a bare pattern
# matches this script's own argv and makes it kill itself, the same trap
# documented in boot_vllm.sh.
PAT="$LCPP"

die() { echo "FATAL: $*" >&2; exit 1; }
live_pids() { pgrep -f "$PAT" 2>/dev/null | grep -vx "$$" || true; }

[ -x "$LCPP" ] || die "no llama-server at $LCPP (set LCPP_BIN)"
[ -f "$GGUF" ] || die "no GGUF at $GGUF (set LCPP_GGUF)"

# ── One 35B at a time ────────────────────────────────────────────────────
# ~19 GB of weights plus KV on a 48 GB box. Booting on top of a live pie or
# vLLM does not fail cleanly, it swaps, and every latency number taken
# afterwards is fiction. Refuse loudly.
for other in "release/pie -c" "VLLM::" "mlx_lm.server"; do
    if pgrep -f "$other" >/dev/null 2>&1; then
        pgrep -fl "$other" >&2
        die "another engine is live ($other); the box does not hold two 35B servers"
    fi
done

# ── 1. Nothing from a previous boot survives ─────────────────────────────
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
[ -z "$(live_pids)" ] || { pgrep -fl "$PAT" >&2; die "a llama-server survived the kill"; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
    die ":$PORT still held by something else"
fi

# ── 2. Boot, in its own session ──────────────────────────────────────────
# Same os.setsid() trick as boot_vllm.sh: the whole tree becomes one process
# group the next boot can signal. macOS ships no setsid(1).
: > "$LOG"
nohup /usr/bin/python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "$LCPP" \
    -m "$GGUF" \
    --host 127.0.0.1 --port "$PORT" \
    --alias "$SERVED" \
    -c "$CTX" \
    -np 1 \
    -ngl 999 \
    -fa on \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --no-webui \
    "$@" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "booting $TAG: pid $PID, gguf $(basename "$GGUF"), ctx $CTX, log $LOG"

t0=$(date +%s)
deadline=$(( t0 + BOOT_TIMEOUT ))
while :; do
    now=$(date +%s)
    [ "$now" -lt "$deadline" ] || { tail -30 "$LOG" >&2; die "$TAG never came up in ${BOOT_TIMEOUT}s"; }

    if ! kill -0 "$PID" 2>/dev/null; then
        tail -30 "$LOG" >&2
        die "the server exited during boot (see $LOG)"
    fi
    if grep -qiE "address already in use|bind.*failed" "$LOG" 2>/dev/null; then
        die "this boot lost the bind on :$PORT"
    fi

    models="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [ -n "$models" ] && printf '%s' "$models" | grep -q '"id"'; then
        # Identity: the listener must BE us. "something answers" is exactly the
        # evidence that misled four measurements on the pie side.
        lp="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
        lpgid="$(ps -o pgid= -p "$lp" 2>/dev/null | tr -d ' ' || true)"
        if [ "$lp" != "$PID" ] && [ "$lpgid" != "$PID" ]; then
            echo "listener on :$PORT is pid $lp (pgid $lpgid), not this boot's $PID" >&2
            ps -o pid=,pgid=,command= -p "$lp" >&2 || true
            die "someone else owns the port"
        fi
        printf '%s' "$models" | grep -q "\"$SERVED\"" \
            || { printf '%s\n' "$models" >&2; die "server does not serve '$SERVED'"; }
        echo "up: $TAG serving '$SERVED' as pid $PID in $(( now - t0 ))s"
        exit 0
    fi
    sleep 3
done
