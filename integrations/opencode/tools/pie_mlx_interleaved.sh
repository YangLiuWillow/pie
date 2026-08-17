#!/usr/bin/env bash
# pie vs mlx-lm, INTERLEAVED, because the four-way could not settle it.
#
# ## Why this exists
#
# `four_way.sh` runs four arms as blocks and repeats the first arm last to
# measure drift rather than assume it. On 2026-08-16 that control fired hard:
#
#     prompt   pie(1st)  pie(last)   drift
#       5840       67.0       67.2    0.3%
#      16090       40.7       49.4   21.4%   <-- same arm, same binary
#      28390       25.2       28.2   11.9%
#
# 40.7 is 0.86x of mlx's 47.4 and 49.4 is 1.04x of it, so the SAME arm supports
# "pie loses by 14%" and "pie wins by 4%" depending on which boot you read. The
# memory roofs were 296.0-296.8 GB/s across all five arms, so this is not
# bandwidth -- it is the GPU clock, and a block of one engine followed by a
# block of the other puts the ramp inside whichever ran second.
#
# The harness's own rule for that case is "redo it interleaved". This is that.
#
# ## The design
#
# Three ROUNDS. Each round boots both engines back to back and probes all three
# prompt sizes; the ORDER REVERSES every round:
#
#     round 1   pie, mlx
#     round 2   mlx, pie
#     round 3   pie, mlx
#
# Alternating is what interleaving buys: whatever the machine's thermal state
# does over ~25 minutes, each engine sees it from both sides. Three samples per
# engine per size also means the per-engine SPREAD is measured, and that spread
# is the instrument's own error bar -- a difference smaller than it is not a
# result, however tidy the medians look.
#
# Two 30B servers do not fit in 48 GB, so every arm boots, is measured, and is
# killed before the next starts. That is why this costs six boots.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${INTERLEAVE_OUT:-/tmp/pie-mlx-interleaved-$(date +%m%d-%H%M)}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
ROOFLINE=/tmp/metaltools/bin/roofline_probe
MLXMODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
ROUNDS="${INTERLEAVE_ROUNDS:-3}"

mkdir -p "$OUT"
cd "$REPO/integrations/opencode"
# shellcheck source=require_quiet_gpu.sh
source tools/require_quiet_gpu.sh

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    pkill -f mlx_lm.server 2>/dev/null
    sleep 6
}
trap stop_all EXIT

# The roof per arm, recorded rather than assumed. The four-way's roofs agreed to
# within 0.3% while its decode rates moved 21%, which is the evidence that the
# drift is the clock and not the memory system -- worth being able to say again.
check_roof() {  # $1 label
    local roof
    roof=$($ROOFLINE 2>/dev/null | awk '/streaming roof/{print $(NF-1)}')
    if ! printf '%s' "$roof" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "FATAL: unparseable roof for $1 (got '${roof}')" >&2; return 1
    fi
    echo "$1 $roof" >> "$OUT/roofs.txt"
    awk -v r="$roof" 'BEGIN{exit !(r + 0 < 250)}' && {
        echo "FATAL: roof ${roof} is far below this machine's ~296." >&2; return 1
    }
    return 0
}

probe() {  # $1 label -> json; FATAL if it produced nothing
    local label=$1 url=$2 model=$3
    python3 tools/rate_probe.py --base-url "$url" --model "$model" \
        --label "$label" --json "$OUT/rate-$label.json" > "$OUT/log-$label.txt" 2>&1
    if ! grep -q "tok/s" "$OUT/log-$label.txt"; then
        echo "FATAL: $label produced no rate. Tail:" >&2
        tail -5 "$OUT/log-$label.txt" >&2
        return 1
    fi
    grep -oE "prompt=[0-9]+ .*tok/s" "$OUT/log-$label.txt" | sed 's/^/      /'
}

run_pie() {  # $1 label
    stop_all; require_quiet_gpu 20 || return 1; check_roof "$1" || return 1
    echo "   ── $1 (pie)"
    PIE_PYTHON=$PIEPY tools/boot_pie.sh "$1" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
        PIE_METAL_SDPA_HSHARE=1 PIE_METAL_SDPA_NAX=1 PIE_METAL_QMM_NAX=1 \
        > "$OUT/boot-$1.log" 2>&1 || { echo "   [$1] BOOT FAILED"; return 1; }
    probe "$1" http://127.0.0.1:8080 qwen3-coder-30b
}

run_mlx() {  # $1 label
    stop_all; require_quiet_gpu 20 || return 1; check_roof "$1" || return 1
    echo "   ── $1 (mlx-lm)"
    nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model "$MLXMODEL" --port 8001 \
        --host 127.0.0.1 > "$OUT/boot-$1.log" 2>&1 &
    local ok=0
    for _ in $(seq 1 60); do
        curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && { ok=1; break; }
        sleep 3
    done
    # "Something answered" is not "the thing I meant is alive" -- but here the
    # server is freshly spawned and nothing else binds 8001, and a probe that
    # returns no rate is fatal below regardless.
    [ "$ok" = 1 ] || { echo "   [$1] mlx never came up"; return 1; }
    probe "$1" http://127.0.0.1:8001 "$MLXMODEL"
}

echo "=============================================================="
echo " pie vs mlx-lm, interleaved, $ROUNDS rounds, order alternating"
echo " out: $OUT"
echo "=============================================================="

for r in $(seq 1 "$ROUNDS"); do
    echo "── round $r"
    if [ $((r % 2)) -eq 1 ]; then
        run_pie "pie-r$r" || exit 1
        run_mlx "mlx-r$r" || exit 1
    else
        run_mlx "mlx-r$r" || exit 1
        run_pie "pie-r$r" || exit 1
    fi
done

python3 - "$OUT" "$ROUNDS" <<'PY'
import json, sys, statistics
out, rounds = sys.argv[1], int(sys.argv[2])
def load(engine):
    by = {}
    for r in range(1, rounds + 1):
        try: d = json.load(open(f"{out}/rate-{engine}-r{r}.json"))
        except Exception: continue
        for e in d:
            if "error" in e: continue
            by.setdefault(e["prompt_tokens"], []).append(e["decode_tok_s"])
    return by
pie, mlx = load("pie"), load("mlx")

print(f"\n{'prompt':>7}  {'pie samples':>22} {'median':>7} {'spread':>7}"
      f"   {'mlx samples':>22} {'median':>7} {'spread':>7}   {'pie/mlx':>8}")
verdicts = []
for p in sorted(set(pie) | set(mlx)):
    a, b = pie.get(p, []), mlx.get(p, [])
    if len(a) < 2 or len(b) < 2:
        print(f"{p:>7}  too few samples ({len(a)} pie, {len(b)} mlx)"); continue
    ma, mb = statistics.median(a), statistics.median(b)
    sa = (max(a)-min(a))/ma*100
    sb = (max(b)-min(b))/mb*100
    ratio = ma/mb
    # The honest test: is the gap bigger than the noise that produced it? The
    # worst case for pie against the best case for mlx, and vice versa.
    lo, hi = min(a)/max(b), max(a)/min(b)
    if lo > 1:   v = "pie faster"
    elif hi < 1: v = "mlx faster"
    else:        v = "UNDECIDED"
    verdicts.append((p, v))
    print(f"{p:>7}  {str([round(x,1) for x in a]):>22} {ma:>7.1f} {sa:>6.1f}%"
          f"   {str([round(x,1) for x in b]):>22} {mb:>7.1f} {sb:>6.1f}%"
          f"   {ratio:>7.2f}x  {v}")
print("\nbracket = min(pie)/max(mlx) .. max(pie)/min(mlx); a bracket straddling")
print("1.00 means this session cannot tell the two apart at that size.")
PY
echo
echo "raw output in $OUT"
