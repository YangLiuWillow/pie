#!/usr/bin/env bash
# How OFTEN does an agentic run degenerate into a loop?
#
# ## Why a rate, and not another bisect
#
# The bisect died on its own control. `current-all-on` -- byte-for-byte the
# configuration that produced 0 bytes and maxrep 45 on django-12276 -- produced
# a 3216-byte patch at maxrep 2 on the next run. And `spec-off` looped at 104 on
# 11099 where the shipped build looped at 132. The loop appears and disappears
# across repetitions of ONE configuration, so no single-rep A/B of any switch can
# say anything about it, and running the remaining arms would only have produced
# more uninterpretable points.
#
# What is worth knowing instead is the RATE. It is the dominant failure mode for
# every engine measured tonight -- pie and mlx-lm both -- and nobody has a number
# for it. A rate is also the only thing that could later show a mitigation
# working, since a mitigation has to move a distribution, not an instance.
#
# ## Design
#
# One configuration (the shipped one), several instances, several repetitions
# each. Three instances that looped tonight, and one that never has, because a
# metric that fires everywhere measures nothing -- 14373 is the control and
# should stay clean.
#
# Every run restarts the server, exactly as the graded run did, so this measures
# the same thing the accuracy number was measured on.
#
# ## Reading it
#
# `maxrep >= 10` is degenerate; every such run tonight produced a 0-byte patch.
# The rate per instance matters more than the pooled rate: if 12276 loops half
# the time and 14373 never does, the exposure is instance-shaped, which is
# actionable in a way "pie loops sometimes" is not.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${LOOPRATE_OUT:-/tmp/loop-rate}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
INSTANCES="${LOOPRATE_INSTANCES:-django__django-12276 django__django-11099 django__django-13158 django__django-14373}"
REPS="${LOOPRATE_REPS:-5}"
DEADLINE_S="${LOOPRATE_BUDGET:-14400}"   # stop starting runs after 4h

[ -x "$REPO/target/release/pie" ] || { echo "FATAL: no pie binary" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || exit 1
START=$(date +%s)

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 8
}
trap stop_all EXIT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/loop-rate.log"; }

echo "instance,rep,patch_bytes,maxrep,looped" > "$OUT/results.csv"
log "=== loop rate: $REPS reps x $(echo $INSTANCES | wc -w | tr -d ' ') instances, shipped config ==="

for inst in $INSTANCES; do
  for r in $(seq 1 "$REPS"); do
    if [ $(( $(date +%s) - START )) -gt "$DEADLINE_S" ]; then
        log "budget spent; stopping before $inst rep $r"
        break 2
    fi
    stop_all
    wd="$OUT/wd-$inst-$r"; mkdir -p "$wd"
    restart="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 8; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh lr \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY >/dev/null 2>&1"
    $HARNESS_PY run_swebench.py --instances "$inst" \
        --model pie/qwen3-coder-30b --label "lr" \
        --out "$OUT/preds-$inst-$r.jsonl" --timeout 1800 \
        --workdir "$wd" --restart-cmd "$restart" > "$OUT/$inst-$r.log" 2>&1
    bytes=0
    [ -s "$OUT/preds-$inst-$r.jsonl" ] && bytes=$(python3 -c "
import json;print(len(json.loads(open('$OUT/preds-$inst-$r.jsonl').readline()).get('model_patch') or ''))" 2>/dev/null || echo 0)
    maxrep=0
    tr=$(ls "$wd"/*.opencode.log 2>/dev/null | head -1)
    [ -n "$tr" ] && maxrep=$(awk 'length($0)>60' "$tr" | sort | uniq -c | sort -rn | head -1 | awk '{print $1+0}')
    looped=0; [ "${maxrep:-0}" -ge 10 ] && looped=1
    log "$(printf '%-26s rep %s  patch=%-6s maxrep=%-5s %s' "$inst" "$r" "$bytes" "${maxrep:-0}" "$([ $looped = 1 ] && echo LOOP || echo ok)")"
    echo "$inst,$r,$bytes,${maxrep:-0},$looped" >> "$OUT/results.csv"
  done
done

log "=== loop rate summary ==="
python3 - "$OUT/results.csv" <<'PY' 2>&1 | tee -a "$OUT/loop-rate.log"
import csv, sys, collections
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows: print("no runs recorded"); raise SystemExit
by=collections.defaultdict(list)
for r in rows: by[r["instance"]].append(r)
print(f"\n{'instance':<26} {'runs':>5} {'looped':>7} {'rate':>7}  {'patches':>8}")
tot=lp=0
for inst,rs in sorted(by.items()):
    n=len(rs); l=sum(int(r["looped"]) for r in rs)
    p=sum(1 for r in rs if int(r["patch_bytes"])>0)
    tot+=n; lp+=l
    print(f"{inst:<26} {n:>5} {l:>7} {l/n*100:>6.0f}% {p:>8}")
print(f"\n{'POOLED':<26} {tot:>5} {lp:>7} {lp/tot*100:>6.0f}%")
print("""
A loop is maxrep>=10; every such run produced a 0-byte patch. Per-instance rates
matter more than the pooled one -- an instance-shaped exposure is actionable,
"pie loops sometimes" is not. django-14373 is the control and should read 0%.""")
PY
