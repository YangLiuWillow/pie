#!/usr/bin/env bash
# Greedy determinism at LENGTH, through the real server.
#
# The value-layer probe (`logprob-determinism`) already answers this more
# directly and more strongly: 32/32 logprobs bit-identical over six runs. This
# is the complementary test, and it covers what that one cannot -- the full
# serving path, with speculation, prefix caching, the session shim and the HTTP
# layer all in it, at a length where a small perturbation has many chances to
# reach a narrow top-2 gap and flip a token.
#
# Length is the amplifier: a value-layer perturbation only becomes a TOKEN
# difference where the top-2 gap is narrow, so 1000 tokens is roughly five times
# the exposure of the 200-token generations checked before this.
#
# (This was originally motivated by a CUDA nondeterminism report that its author
# has since retracted -- it was a first-request transient, not drift. The test
# stands on its own: it covers the serving path, which the inferlet probe does
# not.)
#
# Two boots, because a single boot shares one process, one set of compiled
# pipelines and one prefix cache; only a second boot recomputes from cold.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${LONGDET_OUT:-/tmp/longdet-$(date +%m%d-%H%M)}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
TOKENS="${LONGDET_TOKENS:-1000}"
PER_BOOT="${LONGDET_PER_BOOT:-3}"

if [ ! -x "$REPO/target/release/pie" ]; then
    echo "FATAL: PIE_REPO='$REPO' has no target/release/pie" >&2; exit 1
fi
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || { echo "FATAL: no integrations/opencode" >&2; exit 1; }

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 6
}
trap stop_all EXIT

gen() {  # $1 outfile
    curl -s -m 900 http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer local' \
        -d "{\"model\":\"qwen3-coder-30b\",\"temperature\":0,\"max_tokens\":$TOKENS,
             \"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed technical explanation of how a paged key-value cache works in a transformer inference engine, covering page tables, block allocation, eviction, and how attention reads across pages. Be specific and thorough.\"}]}" \
        > "$1" 2>&1
    python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
    print(d['choices'][0]['message']['content'])
except Exception as e:
    print('REQUEST-FAILED', e, file=sys.stderr); sys.exit(1)
" > "$1.txt" 2> "$1.err"
}

for boot in 1 2; do
    stop_all
    echo "── boot $boot"
    PIE_PYTHON=$PIEPY tools/boot_pie.sh "longdet$boot" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
        > "$OUT/boot-$boot.log" 2>&1 || { echo "FATAL: boot $boot failed"; exit 1; }
    for i in $(seq 1 "$PER_BOOT"); do
        gen "$OUT/b${boot}-g${i}.json"
        s=$(wc -c < "$OUT/b${boot}-g${i}.json.txt" | tr -d ' ')
        echo "   boot $boot gen $i: $s chars"
        # A failed request writes an empty file, and empty files compare equal
        # to each other -- which would report perfect determinism from a server
        # that answered nothing at all.
        if [ "$s" -lt 100 ]; then
            echo "FATAL: boot $boot gen $i produced $s chars; not a generation" >&2
            head -3 "$OUT/b${boot}-g${i}.json" >&2
            exit 1
        fi
    done
done

python3 - "$OUT" <<'PY'
import glob, sys, itertools
out = sys.argv[1]
files = sorted(glob.glob(f"{out}/b*-g*.json.txt"))
texts = {f.split('/')[-1].replace('.json.txt',''): open(f).read() for f in files}
print(f"\n{len(texts)} generations, {min(len(t) for t in texts.values())}-"
      f"{max(len(t) for t in texts.values())} chars each\n")
names = sorted(texts)
ref = texts[names[0]]
allsame = True
for a, b in itertools.combinations(names, 2):
    ta, tb = texts[a], texts[b]
    if ta == tb:
        verdict = "IDENTICAL"
    else:
        allsame = False
        k = next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), min(len(ta),len(tb)))
        verdict = f"DIVERGES @ char {k} of {min(len(ta),len(tb))}"
    same_boot = a[1] == b[1]
    print(f"  {a} vs {b}  {'(same boot)' if same_boot else '(cross-boot)':<13} {verdict}")
print()
if allsame:
    print(f"All {len(texts)} generations byte-identical at ~{len(ref)} chars,")
    print("within and across boots. Greedy is stable at this length.")
else:
    print("Divergence found — record WHERE, since a late flip and an early one")
    print("are different findings.")
PY
echo
echo "raw output in $OUT"
