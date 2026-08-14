set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/phases
PIEPY=/Users/liuyang/.venvs/pie/bin/python
mkdir -p "$OUT"; cd "$REPO"
pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null; sleep 5
PIE_PYTHON=$PIEPY ./integrations/opencode/tools/boot_pie.sh phza \
    PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 PIE_PYTHON=$PIEPY 2>&1 | tail -1
# Tools INCLUDED and short generations: the prefill-shaped turn, which is where
# the unexplained ~1 s per turn lives.
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/bench_ab.py \
    --arm phza --model qwen3-coder-30b --turns 6 --max-tokens 96 \
    --out "$OUT/phza.json" 2>&1 | tee "$OUT/phza.txt"
cp /tmp/pie_opencode_shim.log "$OUT/phza-shim.log"
echo "── guest phases ──"
grep -ao "phases_ms[^\\\\]\{0,150\}" "$OUT/phza-shim.log" | tee -a "$OUT/phza.txt"
echo "── guest generate split ──"
grep -ao "gen_ms[^\\\\]\{0,150\}" "$OUT/phza-shim.log" | tee -a "$OUT/phza.txt"
pkill -f "$REPO/target/release/pie .*serve"; pkill -f session_shim.py
