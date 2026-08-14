#!/usr/bin/env bash
# Speculation's BEST case, on both engines: an answer that is almost entirely a
# copy of text already in the context.
#
# Why this exists. On the agent-shaped turn, pie gained ~3% and vLLM ~0%, and
# vLLM's Prometheus spec counters read zero -- but `SpecDecodingStats` appears
# nowhere in the `vllm_metal` plugin, so those counters are never written on
# this backend and zero means "not reported", not "not drafted". Rather than
# instrument someone else's package, this asks the question functionally: give
# both engines output that prompt-lookup drafting cannot fail to predict. An
# engine whose speculation works shows a large gain here. One whose speculation
# is inert shows the same number twice.
set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/spec-ceiling
PIEPY=/Users/liuyang/.venvs/pie/bin/python
SPEC_JSON='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'
SESSION_WASM="$REPO/inferlets/opencode-session/target/wasm32-wasip2/release/opencode_session.wasm"
# Verbatim repetition: every token after the first pass is a copy of one three
# tokens back in the SAME output, which is exactly what a 2/3-gram lookup finds.
TAIL="Without calling any tools, repeat the following line exactly twenty times, one per line, changing nothing: The quick brown fox jumps over the lazy dog near pkg/mod_17.py."
mkdir -p "$OUT"; cd "$REPO"

stop_all() {
    pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null
    sleep 6
}

bench() {   # $1 = tag, $2 = base url
    PIE_BASE_URL="$2" python3 integrations/opencode/bench_ab.py \
        --arm "$1" --model qwen3-coder-30b --turns 2 --max-tokens 400 \
        --decode-probe --decode-tail "$TAIL" --out "$OUT/$1.json" 2>&1 | tee "$OUT/$1.txt"
}

pie_cell() {   # $1 = on|off, $2 = tag
    stop_all
    touch inferlets/opencode-session/src/engine.rs
    ( cd inferlets/opencode-session
      if [ "$1" = off ]; then SPEC_OFF=1 cargo build --target wasm32-wasip2 --release
      else cargo build --target wasm32-wasip2 --release; fi ) 2>&1 | tail -1
    local want; want=$(shasum -a 256 "$SESSION_WASM" | cut -d' ' -f1)
    PIE_PYTHON=$PIEPY ./integrations/opencode/tools/boot_pie.sh "$2" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 PIE_PYTHON=$PIEPY 2>&1 | tail -1
    local served; served=$(grep -a "strategy b: session inferlet" "/tmp/serve_$2.out" | sed -E 's/.*inferlet ([^ ]+) .*/\1/' | tail -1)
    local got; got=$(shasum -a 256 "$served" 2>/dev/null | cut -d' ' -f1)
    [ "$want" = "$got" ] || { echo "FATAL[$2]: served wasm is not the spec=$1 build"; return 1; }
    echo "verified[$2]: spec=$1 build (${want:0:16})"
    bench "$2" http://127.0.0.1:8080
    cp /tmp/pie_opencode_shim.log "$OUT/$2-shim.log" 2>/dev/null
    grep -ao "speculation [0-9][^)]*)" "$OUT/$2-shim.log" | tee -a "$OUT/$2.txt"
}

vllm_cell() {   # $1 = on|off, $2 = tag
    stop_all
    if [ "$1" = on ]; then export VLLM_SPEC_ARG="$SPEC_JSON"; else unset VLLM_SPEC_ARG; fi
    VLLM_MAX_MODEL_LEN=65536 ./integrations/opencode/tools/boot_vllm.sh "$2" 2>&1 | tail -1
    grep -ac "N-gram speculative decoding enabled" "/tmp/vllm_$2.log" | \
        sed "s/^/[$2] ngram-enabled log lines: /" | tee -a "$OUT/$2.txt"
    bench "$2" http://127.0.0.1:8000
}

pie_cell  on  cpspec
vllm_cell on  cvspec
pie_cell  off cpctl
vllm_cell off cvctl
stop_all
echo "=== done ==="
