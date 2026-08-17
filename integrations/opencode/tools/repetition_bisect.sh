#!/usr/bin/env bash
# Why does pie loop on django-12276, and which switch is responsible?
#
# ## The regression
#
# 2026-08-13, officially graded: django__django-12276 RESOLVED, 545 s, 411-byte
# patch. Tonight, same instance, same agent, same checkpoint: 99 s, **0 bytes**,
# and a transcript that is one sentence emitted over and over until cutoff. The
# other 0-byte instance (13028) repeats a single line 127 times. The instances
# that produced patches show no repetition at all, so this is not "the agent
# gave up" -- it is degenerate decoding, and it correlates exactly with failure.
#
# ## What changed between the graded run and tonight
#
# Kernels, all of which alter numerics and all of which are runtime switches:
#
#     PIE_METAL_SDPA_SPLIT    split-K decode attention
#     PIE_METAL_SDPA_HSHARE   GQA head sharing in decode
#     PIE_METAL_SDPA_NAX      fused prefill attention on the accelerators
#     PIE_METAL_QMM_NAX       both quantized GEMMs on the accelerators
#     PIE_METAL_SDPA_UNROLL   the key-loop unroll (proved to change TEXT)
#
# `_u4` alone was measured this session to change generated text -- including
# below the context gate that was supposed to make it inert -- so "the kernels
# cannot affect output" is already known to be false.
#
# ## The other suspect, which this script does NOT test
#
# Speculation. `opencode-session` drafts by PROMPT LOOKUP -- it finds where
# recent output recurs and proposes the continuation. That is a repetition
# amplifier by construction: once the model emits a line twice, the drafter
# offers the whole line again and greedy verification accepts it in bulk. It
# cannot invent a loop the model would not have entered, but it makes one cheap
# and fast, which fits the 5-15x shorter wall times. Disabling it needs a
# REBUILD (`SPEC_OFF=1 cargo build`), so it is phase two and only if the kernel
# switches come back clean.
#
# ## Reading it
#
# `maxrep` is the highest repeat count of any substantial line in the agent
# transcript. A healthy run is 1-2. The failing runs tonight are 127.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${BISECT_OUT:-/tmp/repetition-bisect}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
INST="${BISECT_INSTANCE:-django__django-12276}"
REPS="${BISECT_REPS:-2}"

[ -x "$REPO/target/release/pie" ] || { echo "FATAL: no pie binary" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || exit 1

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 8
}
trap stop_all EXIT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bisect.log"; }

# $1 tag, $2..$6 the five switches
arm() {
    local tag=$1 split=$2 hshare=$3 nax=$4 qmm=$5 unroll=$6
    for r in $(seq 1 "$REPS"); do
        stop_all
        local wd="$OUT/wd-$tag-$r"
        mkdir -p "$wd"
        local restart="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 8; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh bis \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
  PIE_METAL_SDPA_SPLIT=$split PIE_METAL_SDPA_HSHARE=$hshare \
  PIE_METAL_SDPA_NAX=$nax PIE_METAL_QMM_NAX=$qmm \
  PIE_METAL_SDPA_UNROLL=$unroll >/dev/null 2>&1"
        $HARNESS_PY run_swebench.py --instances "$INST" \
            --model pie/qwen3-coder-30b --label "$tag-$r" \
            --out "$OUT/preds-$tag-$r.jsonl" --timeout 1800 \
            --workdir "$wd" --restart-cmd "$restart" \
            > "$OUT/$tag-$r.log" 2>&1
        local bytes=0 maxrep=0
        if [ -s "$OUT/preds-$tag-$r.jsonl" ]; then
            bytes=$(python3 -c "
import json,sys
d=json.loads(open('$OUT/preds-$tag-$r.jsonl').readline())
print(len(d.get('model_patch') or ''))" 2>/dev/null || echo 0)
        fi
        local tr; tr=$(ls "$wd"/*.opencode.log 2>/dev/null | head -1)
        if [ -n "$tr" ]; then
            maxrep=$(awk 'length($0)>60' "$tr" | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
            [ -n "$maxrep" ] || maxrep=0
        fi
        log "$(printf '%-22s rep=%s  patch=%-6s maxrep=%-5s' "$tag" "$r" "$bytes" "$maxrep")"
        echo "$tag,$r,$bytes,$maxrep" >> "$OUT/results.csv"
    done
}

log "=== repetition bisect on $INST ($REPS reps per arm) ==="
log "baseline to beat: 2026-08-13 graded RESOLVED, 411-byte patch"
echo "arm,rep,patch_bytes,maxrep" > "$OUT/results.csv"

#      tag              split hshare nax qmm unroll
arm "current-all-on"      1     1     1   1    1
arm "unroll-off"          1     1     1   1    0
arm "decode-kernels-off"  0     0     1   1    0
arm "all-off-like-orig"   0     0     0   0    0

log "=== summary ==="
column -s, -t < "$OUT/results.csv" | tee -a "$OUT/bisect.log"
log "maxrep 1-2 is healthy; the failing runs tonight were 127."
log "If every arm loops, the kernels are exonerated and speculation is next:"
log "  SPEC_OFF=1 rebuild of inferlets/opencode-session, then re-run current-all-on."
