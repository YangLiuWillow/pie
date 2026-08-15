#!/usr/bin/env bash
# The same instance, the same engine, N times per arm -- to tell a change from
# run-to-run variance before attributing anything to the change.
#
# Why this exists: the head-sharing decode kernel landed, the three-engine e2e
# was re-run, and pie came back with 3 turns and a 0-byte patch where it had had
# 5 turns and a 412-byte one. The kernel was the obvious suspect. It is also
# byte-identical to the shipped path on the same prompts at 250 tokens and at
# 28k context, which is not what a wrong kernel looks like.
#
# An agent loop is not a deterministic function of the engine. It calls tools,
# reads a filesystem, and decides when to stop; one flipped token near a tie
# changes which file it opens next. So "the arm changed" and "the engine changed
# the arm" are different claims, and only a repeated control separates them.
#
# Usage: bash tools/pie_ab.sh [instance] [reps]
set -uo pipefail
REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
INST="${1:-django__django-14373}"
REPS="${2:-3}"
OUT="${AB_OUT:-/tmp/pie-ab-$(date +%m%d-%H%M)}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
mkdir -p "$OUT"
cd "$REPO/integrations/opencode"

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    pkill -f "tools/turnlog.py" 2>/dev/null
    sleep 5
}

run_rep() {  # $1 arm tag, $2 rep, $3 HSHARE value
    local tag="$1-$2"
    stop_all
    PIE_PYTHON=$PIEPY tools/boot_pie.sh "ab$tag" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_MAX_FORWARD_TOKENS=4096 PIE_METAL_SDPA_HSHARE="$3" \
        PIE_PYTHON=$PIEPY > /dev/null 2>&1 || { echo "[$tag] BOOT FAILED"; return 1; }
    rm -f "$OUT/turns-$tag.jsonl"
    nohup python3 tools/turnlog.py --upstream http://127.0.0.1:8080 --port 8099 \
        --out "$OUT/turns-$tag.jsonl" --tag "$tag" > /dev/null 2>&1 &
    for _ in $(seq 1 20); do
        curl -s -m 2 -o /dev/null http://127.0.0.1:8099/v1/models && break
        sleep 1
    done
    local t0; t0=$(date +%s)
    $HARNESS_PY run_swebench.py --instances "$INST" --model probe/qwen3-coder-30b \
        --label "$tag" --out "$OUT/preds-$tag.jsonl" --timeout 2400 \
        > "$OUT/$tag.log" 2>&1
    local patch; patch=$(python3 -c "
import json;print(len(json.loads(open('$OUT/preds-$tag.jsonl').read().strip()).get('model_patch','')))
" 2>/dev/null || echo "?")
    local turns; turns=$(wc -l < "$OUT/turns-$tag.jsonl" 2>/dev/null | tr -d ' ')
    local server; server=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT/turns-$tag.jsonl') if l.strip()]
ok=[r for r in rows if r.get('ttft_s') is not None]
print('%.1f %.1f %.1f %d' % (sum(r['total_s'] for r in rows),
      sum(r['ttft_s'] for r in ok), sum(r['decode_s'] for r in ok),
      sum(r.get('completion_tokens') or 0 for r in ok)))
" 2>/dev/null || echo "- - - -")
    echo "[$tag] wall $(( $(date +%s) - t0 ))s  turns=$turns  patch=${patch}B  server/ttft/decode/out: $server"
}

echo "=== $INST, $REPS reps per arm ==="
for i in $(seq 1 "$REPS"); do run_rep hshare "$i" 1; done
for i in $(seq 1 "$REPS"); do run_rep base "$i" 0; done
stop_all
echo "=== artifacts in $OUT ==="
