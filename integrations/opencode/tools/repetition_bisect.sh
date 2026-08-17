#!/usr/bin/env bash
# What makes pie loop? Speculation, or the kernels?
#
# ## The regression being diagnosed
#
# Tonight, 4 of pie's 5 failures were degenerate repetition and every looping
# instance produced a 0-byte patch:
#
#     11099 maxrep 132 | 13028 maxrep 127 | 13158 maxrep 74 | 12276 maxrep 45
#
# django-12276 was RESOLVED on 2026-08-13 with a 411-byte patch. A control run
# from 2026-08-14 09:29 -- before the drafter landed at 15:37 that day -- has it
# at maxrep 2. So looping is not new (13028 was already at 38 in that control)
# but it has roughly tripled, and it has spread to an instance that used to pass.
#
# ## The two suspects, tested together
#
# SPECULATION. The drafter is PROMPT LOOKUP: it finds where recent output recurs
# and proposes the continuation, which is a repetition amplifier by construction.
# It is also not output-neutral on this driver, whatever its accept test does,
# because a verify fire carries DRAFT_K+1 = 5 rows and `rows > 1` selects the
# HEAD-SHARING kernel while a plain rows=1 decode selects SPLIT-K. Different
# kernel, different rounding, different argmax near ties.
#
# KERNELS. Five runtime switches, all landed in the same window, all altering
# numerics. `_u4` was measured this session to change generated text outright.
#
# ## Arms
#
#     spec-off            drafting compiled out (staged wasm), kernels all on
#     current-all-on      tonight's configuration, the thing to reproduce
#     unroll-off          only PIE_METAL_SDPA_UNROLL=0
#     decode-kernels-off  split-K and head-sharing off too
#     all-off-like-orig   every switch off
#
# `spec-off` runs FIRST: it is the strongest hypothesis and the cheapest to
# interpret. If it is clean and every kernel arm loops, speculation is the cause
# and no kernel bisect is needed.
#
# ## What is recorded
#
# maxrep (highest repeat count of any substantial transcript line; healthy is
# 1-2), patch bytes, and the drafter's own acceptance counters sliced per run
# from the SHARED shim log by byte offset -- reading that log after a multi-arm
# session hands one arm's numbers to every heading.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${BISECT_OUT:-/tmp/repetition-bisect}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
CANON="$REPO/target/wasm32-wasip2/release/opencode_session.wasm"
SPECON=/tmp/wasm-specON-local.wasm
SPECOFF=/tmp/wasm-specOFF-local.wasm
SHIM_LOG=/tmp/pie_opencode_shim.log
# Two of the worst loopers, one from each subset. 12276 is the regression
# (resolved -> 0 bytes); 11099 is the highest maxrep of the night.
INSTANCES="${BISECT_INSTANCES:-django__django-12276 django__django-11099}"
REPS="${BISECT_REPS:-1}"

for f in "$SPECON" "$SPECOFF"; do
    [ -s "$f" ] || { echo "FATAL: missing staged wasm $f" >&2; exit 1; }
done
cmp -s "$SPECON" "$SPECOFF" && { echo "FATAL: staged wasms are identical" >&2; exit 1; }
[ -x "$REPO/target/release/pie" ] || { echo "FATAL: no pie binary" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || exit 1

ORIG_BACKUP="$OUT/canonical-wasm-at-start.wasm"
cp "$CANON" "$ORIG_BACKUP"
stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 8
}
# Always put the canonical wasm back, however this exits: leaving a spec-off
# build in place would silently change every later pie measurement.
trap 'stop_all; cp "$ORIG_BACKUP" "$CANON"; echo "canonical wasm restored"' EXIT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bisect.log"; }

arm() {  # $1 tag, $2 wasm, $3..$7 switches
    local tag=$1 wasm=$2 split=$3 hshare=$4 nax=$5 qmm=$6 unroll=$7
    cp "$wasm" "$CANON"
    for inst in $INSTANCES; do
      for r in $(seq 1 "$REPS"); do
        stop_all
        local wd="$OUT/wd-$tag-$inst-$r"; mkdir -p "$wd"
        local before=0
        [ -f "$SHIM_LOG" ] && before=$(wc -c < "$SHIM_LOG" | tr -d ' ')
        local restart="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 8; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh bis \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
  PIE_METAL_SDPA_SPLIT=$split PIE_METAL_SDPA_HSHARE=$hshare \
  PIE_METAL_SDPA_NAX=$nax PIE_METAL_QMM_NAX=$qmm \
  PIE_METAL_SDPA_UNROLL=$unroll >/dev/null 2>&1"
        $HARNESS_PY run_swebench.py --instances "$inst" \
            --model pie/qwen3-coder-30b --label "$tag" \
            --out "$OUT/preds-$tag-$inst-$r.jsonl" --timeout 1800 \
            --workdir "$wd" --restart-cmd "$restart" > "$OUT/$tag-$inst-$r.log" 2>&1
        local bytes=0 maxrep=0 acc="-"
        [ -s "$OUT/preds-$tag-$inst-$r.jsonl" ] && bytes=$(python3 -c "
import json;print(len(json.loads(open('$OUT/preds-$tag-$inst-$r.jsonl').readline()).get('model_patch') or ''))" 2>/dev/null || echo 0)
        local tr; tr=$(ls "$wd"/*.opencode.log 2>/dev/null | head -1)
        if [ -n "$tr" ]; then
            maxrep=$(awk 'length($0)>60' "$tr" | sort | uniq -c | sort -rn | head -1 | awk '{print $1+0}')
        fi
        if [ -f "$SHIM_LOG" ]; then
            local after; after=$(wc -c < "$SHIM_LOG" | tr -d ' ')
            if [ "$after" -ge "$before" ]; then
                acc=$(tail -c "+$((before+1))" "$SHIM_LOG" 2>/dev/null \
                      | grep -o "speculation [0-9]*%" | head -1 | grep -o "[0-9]*%" || echo "-")
            fi
            [ -n "$acc" ] || acc="-"
        fi
        log "$(printf '%-20s %-24s patch=%-6s maxrep=%-5s accept=%s' "$tag" "$inst" "$bytes" "${maxrep:-0}" "$acc")"
        echo "$tag,$inst,$r,$bytes,${maxrep:-0},$acc" >> "$OUT/results.csv"
      done
    done
}

log "=== repetition bisect ==="
log "instances: $INSTANCES   reps: $REPS"
log "target to reproduce: 12276 maxrep 45 patch 0 | 11099 maxrep 132 patch 0"
log "pre-change control (08-14 09:29, no speculation): 12276 maxrep 2, RESOLVED"
echo "arm,instance,rep,patch_bytes,maxrep,accept" > "$OUT/results.csv"

#     tag                  wasm       split hshare nax qmm unroll
arm "spec-off"           "$SPECOFF"     1     1     1   1    1
arm "current-all-on"     "$SPECON"      1     1     1   1    1
arm "unroll-off"         "$SPECON"      1     1     1   1    0
arm "decode-kernels-off" "$SPECON"      0     0     1   1    0
arm "all-off-like-orig"  "$SPECON"      0     0     0   0    0

log "=== summary ==="
column -s, -t < "$OUT/results.csv" | tee -a "$OUT/bisect.log"
log ""
log "maxrep 1-2 healthy; tonight's failures were 45-132."
log "If spec-off is clean and every kernel arm loops -> speculation, done."
log "If every arm loops including spec-off -> neither; the model itself"
log "  degenerates here and the graded run was a luckier trajectory."
