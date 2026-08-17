#!/usr/bin/env bash
# Is the unroll's end-to-end regression a SLOWER KERNEL or a WORSE DRAFTER?
#
# ## The contradiction this settles
#
# Per fire, measured on `decode-rows-probe` at the shape a verify fire actually
# has (ctx 28160, rows = DRAFT_K + 1 = 5), the unroll is 1.066x FASTER:
#
#     unroll=1  85.89 ms      unroll=0  91.53 ms
#
# End to end, through the server at ctx 28390, it is 0.85x SLOWER:
#
#     unroll=1  28.3 tok/s    unroll=0  33.4 tok/s
#
# Both were measured with clean controls, so both are believed. The candidate
# reconciliation is that they are not measuring the same thing: `_u4` perturbs
# numerics (proved -- the generated TEXT differs at every prompt size, including
# BELOW the kernel's own 8192 gate), different tokens mean different draft
# acceptance, and acceptance decides how many fires 200 tokens costs. A faster
# fire that drafts worse can be slower per token.
#
# ## What is captured that the last run did not
#
# `opencode-session` prints per request:
#
#     [opencode-session] speculation 24% accepted (26/110 drafted, 173 verify
#                        fires, 26 fires saved)
#
# to the SHARED /tmp/pie_opencode_shim.log. Reading it after a three-arm session
# gives one arm's numbers under every heading -- the same "half an A/B" that has
# already cost this project a full measurement. So each arm records the log's
# BYTE OFFSET before probing and extracts only what was appended after, which
# survives both truncation-per-boot and append-across-boots without assuming
# either.
#
# ## Reading the result
#
# If fires-per-token tracks tok/s across the arms, the regression is acceptance
# and the kernel is innocent -- and the upper gate this was heading toward would
# have been fitted to an artifact. If fires-per-token is flat and tok/s still
# moves, the kernel really is slower in situ and the per-fire probe is missing
# something the server sees.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${ACCEPT_OUT:-/tmp/unroll-accept-$(date +%m%d-%H%M)}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
ROOFLINE=/tmp/metaltools/bin/roofline_probe
SHIM_LOG=/tmp/pie_opencode_shim.log
ROUNDS="${ACCEPT_ROUNDS:-2}"

if [ ! -x "$REPO/target/release/pie" ]; then
    echo "FATAL: PIE_REPO='$REPO' has no target/release/pie; stop_all would kill nothing." >&2
    exit 1
fi
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || { echo "FATAL: no integrations/opencode" >&2; exit 1; }
source tools/require_quiet_gpu.sh

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 6
}
trap stop_all EXIT

check_roof() {
    local roof
    roof=$($ROOFLINE 2>/dev/null | awk '/streaming roof/{print $(NF-1)}')
    printf '%s' "$roof" | grep -qE '^[0-9]+(\.[0-9]+)?$' || {
        echo "FATAL: unparseable roof for $1" >&2; return 1; }
    echo "$1 $roof" >> "$OUT/roofs.txt"
    awk -v r="$roof" 'BEGIN{exit !(r + 0 < 250)}' && {
        echo "FATAL: roof $roof far below ~296" >&2; return 1; }
    # NOT optional, and its absence is why the first run of this script did
    # nothing at all: on a HEALTHY roof the awk above exits non-zero, `&&`
    # short-circuits, and the function returns awk's status -- so every arm
    # read as failed and the session ended after the preflight. `four_way.sh`
    # carries this same line for this same reason.
    return 0
}

arm() {  # $1 label, $2 unroll value
    stop_all; require_quiet_gpu 20 || return 1; check_roof "$1" || return 1
    echo "   ── $1 (UNROLL=$2)"
    PIE_PYTHON=$PIEPY tools/boot_pie.sh "$1" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
        PIE_METAL_SDPA_HSHARE=1 PIE_METAL_SDPA_NAX=1 PIE_METAL_QMM_NAX=1 \
        PIE_METAL_SDPA_UNROLL="$2" \
        > "$OUT/boot-$1.log" 2>&1 || { echo "   [$1] BOOT FAILED"; return 1; }

    # Offset AFTER the boot (the boot itself writes to this log) and before the
    # first request, so the slice is exactly this arm's generations.
    local before=0
    [ -f "$SHIM_LOG" ] && before=$(wc -c < "$SHIM_LOG" | tr -d ' ')

    python3 tools/rate_probe.py --base-url http://127.0.0.1:8080 \
        --model qwen3-coder-30b --label "$1" --json "$OUT/rate-$1.json" \
        > "$OUT/log-$1.txt" 2>&1
    grep -q "tok/s" "$OUT/log-$1.txt" || {
        echo "FATAL: $1 produced no rate"; tail -5 "$OUT/log-$1.txt"; return 1; }

    local after=0
    [ -f "$SHIM_LOG" ] && after=$(wc -c < "$SHIM_LOG" | tr -d ' ')
    # A log that SHRANK was truncated by this boot, so everything in it is ours.
    if [ "$after" -lt "$before" ]; then
        cp "$SHIM_LOG" "$OUT/shim-$1.log"
    else
        tail -c "+$((before + 1))" "$SHIM_LOG" > "$OUT/shim-$1.log" 2>/dev/null || :
    fi
    local n
    n=$(grep -c "speculation" "$OUT/shim-$1.log" 2>/dev/null || echo 0)
    echo "      $(grep -oE 'prompt=[0-9]+ .*tok/s' "$OUT/log-$1.txt" | tr '\n' ' ')"
    echo "      speculation lines captured: $n"
    [ "$n" -gt 0 ] || echo "      WARNING: no speculation lines -- acceptance cannot be read for this arm"
}

echo "=============================================================="
echo " unroll: kernel cost or draft acceptance?   out: $OUT"
echo "=============================================================="
for r in $(seq 1 "$ROUNDS"); do
    echo "── round $r"
    if [ $((r % 2)) -eq 1 ]; then
        arm "on-r$r" 1 || exit 1; arm "off-r$r" 0 || exit 1
    else
        arm "off-r$r" 0 || exit 1; arm "on-r$r" 1 || exit 1
    fi
done

python3 - "$OUT" "$ROUNDS" <<'PY'
import json, re, sys, glob, statistics
out, rounds = sys.argv[1], int(sys.argv[2])
pat = re.compile(r"speculation (\d+)% accepted \((\d+)/(\d+) drafted, (\d+) verify fires, (\d+) fires saved\)")
def arm(label):
    rows = []
    try: rates = json.load(open(f"{out}/rate-{label}.json"))
    except Exception: return []
    specs = []
    try:
        for line in open(f"{out}/shim-{label}.log"):
            m = pat.search(line)
            if m: specs.append(tuple(int(x) for x in m.groups()))
    except Exception: pass
    for i, r in enumerate(rates):
        if "error" in r: continue
        s = specs[i] if i < len(specs) else None
        rows.append((r["prompt_tokens"], r["decode_tok_s"], r["output_tokens"], s))
    return rows

print(f"\n{'arm':>8} {'prompt':>7} {'tok/s':>7} {'out':>5} {'acc%':>5} {'drafted':>9} {'fires':>6} {'saved':>6} {'fires/token':>12}")
data = {}
for rnd in range(1, rounds+1):
    for label in (f"on-r{rnd}", f"off-r{rnd}"):
        for p, tps, outk, s in arm(label):
            if s: acc, ok, dr, fires, saved = s
            else: acc = ok = dr = fires = saved = None
            fpt = fires/outk if fires else None
            key = ("on" if label.startswith("on") else "off", p)
            data.setdefault(key, []).append((tps, fpt))
            print(f"{label:>8} {p:>7} {tps:>7} {outk:>5} "
                  f"{(str(acc)+'%') if acc is not None else '   -':>5} "
                  f"{(f'{ok}/{dr}') if dr else '-':>9} "
                  f"{fires if fires else '-':>6} {saved if saved is not None else '-':>6} "
                  f"{(f'{fpt:.3f}') if fpt else '-':>12}")

print(f"\n{'prompt':>7}  {'tok/s on':>9} {'tok/s off':>10} {'ratio':>7}   "
      f"{'f/tok on':>9} {'f/tok off':>10} {'ratio':>7}   verdict")
for p in sorted({k[1] for k in data}):
    on  = data.get(("on", p), []); off = data.get(("off", p), [])
    if not on or not off: continue
    t_on = statistics.median([x[0] for x in on]); t_off = statistics.median([x[0] for x in off])
    f_on = [x[1] for x in on if x[1]]; f_off = [x[1] for x in off if x[1]]
    if f_on and f_off:
        m_on, m_off = statistics.median(f_on), statistics.median(f_off)
        # If fires/token moved the same way tok/s did, acceptance explains it.
        tr, fr = t_off/t_on, m_on/m_off
        v = "ACCEPTANCE explains it" if abs(fr - tr) < 0.05 else "acceptance does NOT explain it"
        print(f"{p:>7}  {t_on:>9} {t_off:>10} {tr:>6.3f}x   "
              f"{m_on:>9.3f} {m_off:>10.3f} {fr:>6.3f}x   {v}")
    else:
        print(f"{p:>7}  {t_on:>9} {t_off:>10} {t_off/t_on:>6.3f}x   "
              f"{'-':>9} {'-':>10} {'-':>7}   no speculation data")
print("\nfires/token ratio is on/off (more fires = slower), tok/s ratio is off/on.")
print("They match => the tok/s difference is the fire COUNT, not the fire COST.")
PY
echo
echo "raw output in $OUT"
