#!/usr/bin/env bash
# The two speculation-ON cells again, this time capturing what each engine
# actually DRAFTED and ACCEPTED.
#
# Without those counts, "pie gained 3% and vLLM gained 0.3%" is unreadable: it
# is equally consistent with "drafting works and the verify is too expensive",
# "nothing was draftable in this text", and "the feature never engaged". The
# counters separate them, and they are the same quantity on both sides --
# proposed vs accepted draft tokens.
set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/spec-acct
PIEPY=/Users/liuyang/.venvs/pie/bin/python
SPEC_JSON='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'
SESSION_WASM="$REPO/inferlets/opencode-session/target/wasm32-wasip2/release/opencode_session.wasm"
mkdir -p "$OUT"; cd "$REPO"

pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null
sleep 6

# ── pie, speculation on ───────────────────────────────────────────────────
touch inferlets/opencode-session/src/engine.rs
( cd inferlets/opencode-session && cargo build --target wasm32-wasip2 --release ) 2>&1 | tail -2
WANT=$(shasum -a 256 "$SESSION_WASM" | cut -d' ' -f1)
PIE_PYTHON=$PIEPY ./integrations/opencode/tools/boot_pie.sh aspec \
    PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 PIE_PYTHON=$PIEPY 2>&1 | tail -1
SERVED=$(grep -a "strategy b: session inferlet" /tmp/serve_aspec.out | sed -E 's/.*inferlet ([^ ]+) .*/\1/' | tail -1)
GOT=$(shasum -a 256 "$SERVED" 2>/dev/null | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { echo "FATAL: served wasm is not the build"; exit 1; }
echo "verified: pie serving the spec=on build (${WANT:0:16})"
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/bench_ab.py \
    --arm aspec --model qwen3-coder-30b --turns 4 --max-tokens 400 --decode-probe \
    --out "$OUT/aspec.json" 2>&1 | tee "$OUT/aspec.txt"
# Copy the shim log BEFORE the next boot truncates it -- that truncation is why
# the first attempt at this measurement came back empty.
cp /tmp/pie_opencode_shim.log "$OUT/aspec-shim.log" 2>/dev/null
echo "── pie speculation accounting ──" | tee -a "$OUT/aspec.txt"
grep -ao "speculation [0-9][^)]*)" "$OUT/aspec-shim.log" | tee -a "$OUT/aspec.txt"
pkill -f "$REPO/target/release/pie .*serve"; pkill -f session_shim.py; sleep 5

# ── vLLM, speculation on ──────────────────────────────────────────────────
export VLLM_SPEC_ARG="$SPEC_JSON"
VLLM_MAX_MODEL_LEN=65536 ./integrations/opencode/tools/boot_vllm.sh avspec 2>&1 | tail -1
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:spec_decode" > "$OUT/vllm-metrics-before.txt"
PIE_BASE_URL=http://127.0.0.1:8000 python3 integrations/opencode/bench_ab.py \
    --arm avspec --model qwen3-coder-30b --turns 4 --max-tokens 400 --decode-probe \
    --out "$OUT/avspec.json" 2>&1 | tee "$OUT/avspec.txt"
echo "── vLLM speculation accounting ──" | tee -a "$OUT/avspec.txt"
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:spec_decode" | tee "$OUT/vllm-metrics-after.txt" | tee -a "$OUT/avspec.txt"
pkill -f "vllm serve"; pkill -f "VLLM::EngineCore"
echo "=== done ==="
