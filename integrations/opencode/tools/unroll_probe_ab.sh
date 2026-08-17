#!/usr/bin/env bash
# The unroll, on the fire probe, at a shape that can actually reach it.
#
# A 2x2 with TWO built-in controls, which is the point of running it here rather
# than only against the server:
#
#            ctx 7424 (below the 8192 gate)   ctx 28160 (above it)
#   rows=1   split-K kernel   -> CONTROL      split-K kernel   -> CONTROL
#   rows=5   hshare, gate off -> CONTROL      hshare, unrolled -> THE CELL
#
# Only the bottom-right cell can differ. Three of the four are the same code in
# both arms, so if any of them moves, the run is measuring the machine and the
# fourth number means nothing either. That is the check the original A/B could
# not make: it had only rows=1, where BOTH arms are split-K, and reported the
# resulting 1.003x as evidence about a kernel it never ran.
#
# rows=5 is `DRAFT_K + 1` -- the width a verify fire actually carries -- and not
# a round number chosen for the table.
set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
CONF=/tmp/rows-probe/config.toml
WASM=$REPO/runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm
MANIFEST=$REPO/runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml
OUT=${UNROLL_PROBE_OUT:-/tmp/unroll-probe-ab}
REPS=${UNROLL_PROBE_REPS:-3}
ARG="short=7424,long=28160,rows=1:5"
mkdir -p "$OUT"
cd "$REPO"

if pgrep -f "$REPO/target/release/pie .*serve" >/dev/null; then
    echo "FATAL: a pie server is running; it will starve this probe of memory" >&2
    exit 1
fi

one() {  # $1 label, $2 unroll value, $3 rep
    sleep "${UNROLL_PROBE_SETTLE:-25}"
    PIE_METAL_SDPA_UNROLL="$2" RUST_LOG=error "$REPO/target/release/pie" -c "$CONF" run \
        --path "$WASM" --manifest "$MANIFEST" -- "$ARG" > "$OUT/$1-$3.txt" 2>&1
    # The probe prints one median line per (ctx, rows). A run that produced none
    # is fatal, never blank: an arm that died on model load otherwise leaves the
    # other arm's numbers sitting under both headings.
    if ! grep -q "median_ms=" "$OUT/$1-$3.txt"; then
        echo "FATAL: $1 rep $3 produced no medians. Tail:" >&2
        tail -4 "$OUT/$1-$3.txt" >&2
        exit 1
    fi
    # Guard the argument itself: if the probe fell back to its defaults, long
    # would read 7424 and every cell would be below the gate.
    if ! grep -q "short=7424 long=28160" "$OUT/$1-$3.txt"; then
        echo "FATAL: $1 rep $3 did not take the argument:" >&2
        grep -m1 "\[rows\] page=" "$OUT/$1-$3.txt" >&2
        exit 1
    fi
}

for r in $(seq 1 "$REPS"); do
    echo "── rep $r"
    one on  1 "$r"; echo "   unroll=1 done"
    one off 0 "$r"; echo "   unroll=0 done"
done

python3 - "$OUT" "$REPS" <<'PY'
import re, sys, statistics
out, reps = sys.argv[1], int(sys.argv[2])
def read(label, rep):
    d = {}
    for line in open(f"{out}/{label}-{rep}.txt"):
        m = re.search(r"\[rows\] ctx=(\d+) rows=(\d+) median_ms=([\d.]+)", line)
        if m: d[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return d
on  = [read("on", r)  for r in range(1, reps+1)]
off = [read("off", r) for r in range(1, reps+1)]
keys = sorted(set().union(*[set(d) for d in on+off]))
print(f"\n{'ctx':>7} {'rows':>5}  {'unroll=1':>22} {'med':>7}  {'unroll=0':>22} {'med':>7}  {'off/on':>7}  what")
for k in keys:
    a = [d[k] for d in on if k in d]; b = [d[k] for d in off if k in d]
    if not a or not b: continue
    ma, mb = statistics.median(a), statistics.median(b)
    ctx, rows = k
    gate_on = ctx >= 8192
    hshare  = rows != 1
    cell = "THE CELL" if (gate_on and hshare) else "control"
    ratio = mb/ma
    bad = cell == "control" and abs(ratio-1) > 0.01
    print(f"{ctx:>7} {rows:>5}  {str([round(x,2) for x in a]):>22} {ma:>7.2f}"
          f"  {str([round(x,2) for x in b]):>22} {mb:>7.2f}  {ratio:>6.3f}x  {cell}"
          + ("   <-- CONTROL MOVED, run is void" if bad else ""))
print("\ncontrols must read 1.000x: same kernel, same code, both arms.")
PY
echo
echo "raw output in $OUT"
