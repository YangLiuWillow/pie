#!/usr/bin/env bash
# Four cells: {pie, vLLM} x {speculation on, off}, same canned transcript.
#
# The arms are INTERLEAVED (pie, vllm, pie, vllm) rather than grouped, so that
# thermal drift over the ~40 minutes does not line up with the engine axis. The
# same mistake in reverse is what made a 1.61x speculation "win" evaporate: the
# two runs differed in temperature as well as in drafting.
set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/spec-ab-decode
PIEPY=/Users/liuyang/.venvs/pie/bin/python
SPEC_JSON='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'
TURNS=4
MAXTOK=400
mkdir -p "$OUT"
cd "$REPO"

kill_vllm() {
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 5
}
kill_pie() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null || true
    pkill -f "session_shim.py" 2>/dev/null || true
    sleep 4
}

# `opencode-session` is its OWN cargo workspace (it is a wasm guest and is
# deliberately not a member of the host workspace), so `-p opencode-session`
# from the repo root does not resolve it -- it fails, and a script that ignores
# the failure then serves whatever wasm happens to be on disk. That is not
# hypothetical: the first launch of this file did exactly that and would have
# compared the spec-off build against itself.
SESSION_WASM="$REPO/inferlets/opencode-session/target/wasm32-wasip2/release/opencode_session.wasm"

build_session() {   # $1 = "on" | "off"
    # `option_env!` is read at COMPILE time, so the control is a different
    # build, not a different environment at run time. Touching the source
    # guarantees the recompile rather than trusting cargo to fingerprint an env
    # var it may not be tracking.
    touch inferlets/opencode-session/src/engine.rs
    ( cd inferlets/opencode-session
      if [ "$1" = "off" ]; then SPEC_OFF=1 cargo build --target wasm32-wasip2 --release
      else cargo build --target wasm32-wasip2 --release; fi ) 2>&1 | tail -3
    [ -f "$SESSION_WASM" ] || { echo "FATAL: no wasm at $SESSION_WASM"; return 1; }
    cp "$SESSION_WASM" "$OUT/session-spec-$1.wasm"
    echo "built session wasm spec=$1 sha=$(shasum -a 256 "$SESSION_WASM" | cut -c1-16)"
}

run_pie() {   # $1 = spec on|off, $2 = tag
    kill_vllm
    build_session "$1" || return 1
    local want; want=$(shasum -a 256 "$SESSION_WASM" | cut -d' ' -f1)
    PIE_PYTHON=$PIEPY ./integrations/opencode/tools/boot_pie.sh "$2" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_PYTHON=$PIEPY 2>&1 | tail -2 || return 1
    # The runner picks the NEWEST of three candidate paths. Prove the one it
    # picked is the one just built, by bytes -- a stale guest has silently
    # invalidated three measurements in this project already.
    local served; served=$(grep -a "strategy b: session inferlet" "/tmp/serve_$2.out" | sed -E 's/.*inferlet ([^ ]+) .*/\1/' | tail -1)
    local got; got=$(shasum -a 256 "$served" 2>/dev/null | cut -d' ' -f1)
    if [ "$want" != "$got" ]; then
        echo "FATAL[$2]: served $served does not match the build (want ${want:0:16}, got ${got:0:16})"
        kill_pie; return 1
    fi
    echo "verified[$2]: serving the spec=$1 build (${want:0:16})"
    PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/bench_ab.py \
        --arm "$2" --model qwen3-coder-30b --turns $TURNS --max-tokens $MAXTOK --decode-probe \
        --out "$OUT/$2.json" 2>&1 | tee "$OUT/$2.txt"
    grep -a "speculation:" "/tmp/pie_$2.log" 2>/dev/null | tail -3 | tee -a "$OUT/$2.txt"
    kill_pie
}

run_vllm() {   # $1 = spec on|off, $2 = tag
    kill_pie
    if [ "$1" = "on" ]; then export VLLM_SPEC_ARG="$SPEC_JSON"; else unset VLLM_SPEC_ARG; fi
    VLLM_MAX_MODEL_LEN=65536 ./integrations/opencode/tools/boot_vllm.sh "$2" 2>&1 | tail -3 || return 1
    PIE_BASE_URL=http://127.0.0.1:8000 python3 integrations/opencode/bench_ab.py \
        --arm "$2" --model qwen3-coder-30b --turns $TURNS --max-tokens $MAXTOK --decode-probe \
        --out "$OUT/$2.json" 2>&1 | tee "$OUT/$2.txt"
    grep -aiE "spec.*accept|acceptance|drafted" "/tmp/vllm_$2.log" 2>/dev/null | tail -3 | tee -a "$OUT/$2.txt"
    kill_vllm
}

echo "=== 1/4 pie, speculation ON ==="   ; run_pie  on  dpspec
echo "=== 2/4 vLLM, speculation ON ==="  ; run_vllm on  dvspec
echo "=== 3/4 pie, speculation OFF ===" ; run_pie  off dpctl
echo "=== 4/4 vLLM, speculation OFF ===" ; run_vllm off dvctl
echo "=== done; outputs in $OUT ==="
ls -la "$OUT"
