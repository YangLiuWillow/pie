#!/usr/bin/env bash
# Three engines, 30 SWE-bench Verified instances: accuracy, throughput, latency.
#
# ## What is different from the 10-instance run
#
# **Every call goes through `tools/turnlog.py`.** That run reported one number
# per instance -- agent wall clock -- which confounds three independent things:
#
#     turns x (prefill + decode)
#
# An engine loses on wall clock by being slow at prefill, slow at decode, or by
# provoking more turns, and those have different fixes. The proxy sits between
# opencode and whichever server is under test and records per call: TTFT,
# decode wall, and prompt/completion tokens FROM THE SERVER'S OWN USAGE RECORD.
# Same proxy, same clock, same fields, all three engines.
#
# **Thirty instances across twelve repos**, not ten django ones. The previous
# set was django-only and half of it was the August run's known wins; an
# all-django set measures one codebase's idioms. The first ten are kept so the
# two runs are comparable, and reported as their own subset.
#
# ## Symmetry, which is not optional here
#
# Every arm restarts its server per instance. The August comparison gave pie
# that mitigation and vLLM none, scored 4/5 against 1/5, and flagged the
# asymmetry in its own caveats. Re-run symmetrically the same five went 3/5 to
# vLLM's 4/5. Asymmetry produced the entire gap.
#
# ## Degeneration is measured, not assumed
#
# Wall clock scores a degenerate loop as a WIN, because looping hits the cap
# sooner: last night pie ran ten instances in 27 minutes with four of them
# spinning. `transcript_health.py` runs over every arm's transcripts afterward,
# and `maxrep` is reported beside the timings.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${SWE30_OUT:-/tmp/swe30}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
MLXMODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
BUDGET="${SWE30_BUDGET:-39600}"          # 11h; stop starting arms past this
TIMEOUT="${SWE30_TIMEOUT:-1800}"
INSTANCES="$(cat ${SWE30_INSTANCES:-/tmp/swe30-instances.txt})"

[ -x "$REPO/target/release/pie" ] || { echo "FATAL: no pie binary" >&2; exit 1; }
[ -n "$INSTANCES" ] || { echo "FATAL: no instance list" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || exit 1
START=$(date +%s)
left() { echo $(( BUDGET - ( $(date +%s) - START ) )); }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "VLLM::EngineCore" 2>/dev/null
    pkill -f mlx_lm.server 2>/dev/null
    pkill -f "turnlog.py" 2>/dev/null
    sleep 8
}
trap 'stop_all; log "stopped"' EXIT

# The proxy is started ONCE PER ARM, not per instance: it must survive the
# per-instance server restarts, since opencode's baseURL points at it and not at
# the engine. It forwards to whatever is listening upstream at the time.
start_proxy() {  # $1 arm, $2 upstream
    pkill -f "turnlog.py" 2>/dev/null; sleep 2
    nohup $HARNESS_PY tools/turnlog.py --upstream "$2" --port 8099 \
        --out "$OUT/calls-$1.jsonl" --tag "$1" > "$OUT/proxy-$1.log" 2>&1 &
    for _ in $(seq 1 30); do
        curl -s -m 2 -o /dev/null "http://127.0.0.1:8099/v1/models" && return 0
        sleep 1
    done
    # A dead proxy means opencode cannot reach anything and every instance fails
    # identically -- which looks like an engine result. Refuse instead.
    log "[$1] FATAL: proxy never came up"; return 1
}

arm() {  # $1 tag, $2 upstream, $3 model string, $4 restart cmd
    local tag=$1 upstream=$2 model=$3 restart=$4
    if [ "$(left)" -lt $(( TIMEOUT + 1200 )) ]; then
        log "SKIP $tag: only $(left)s left"; return 0; fi
    log "===== $tag ($(left)s budget left) ====="
    stop_all
    # Boot the engine first so the proxy has an upstream to probe.
    eval "$restart" >/dev/null 2>&1
    start_proxy "$tag" "$upstream" || return 1
    local t0; t0=$(date +%s)
    $HARNESS_PY run_swebench.py --instances $INSTANCES --model "$model" \
        --label "$tag" --out "$OUT/preds-$tag.jsonl" --timeout "$TIMEOUT" \
        --workdir "$OUT/wd-$tag" --restart-cmd "$restart" \
        > "$OUT/$tag.log" 2>&1
    local rc=$? n=0 calls=0
    [ -f "$OUT/preds-$tag.jsonl" ] && n=$(wc -l < "$OUT/preds-$tag.jsonl" | tr -d ' ')
    [ -f "$OUT/calls-$tag.jsonl" ] && calls=$(wc -l < "$OUT/calls-$tag.jsonl" | tr -d ' ')
    log "$tag done rc=$rc wall=$(( $(date +%s) - t0 ))s predictions=$n calls=$calls"
    [ "$n" -gt 0 ] || log "WARNING: $tag produced NO predictions"
    [ "$calls" -gt 0 ] || log "WARNING: $tag recorded NO calls -- latency unavailable"
}

# 30s, not 8. `pie` needs ~22.5 GiB resident and macOS RELEASES its pages well
# before it RECLAIMS them: a back-to-back boot reports "only 11.35 GiB is
# reclaimable" and refuses, while `vm_stat` shows 35 GB free. That is exactly how
# the first attempt at this run died on instance 3 of 30 -- and the failure is
# per-arm fatal, since the harness rightly refuses to drive a server it did not
# start. The other two engines get the same settle for symmetry.
# Wait for the CONDITION, not a duration -- see tools/wait_for_memory.sh for why
# `sleep 8` died on instance 3 and `sleep 30` on instance 14. If the machine
# cannot admit the model within five minutes the restart fails loudly, which is
# better than booting into 11 GiB and losing the arm.
PIE_RESTART="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 10; \
$REPO/integrations/opencode/tools/wait_for_memory.sh 26 300 || exit 1; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh s30 \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY"
VLLM_RESTART="pkill -f 'vllm serve'; pkill -f 'VLLM::EngineCore'; sleep 10; \
$REPO/integrations/opencode/tools/wait_for_memory.sh 26 300 || exit 1; \
VLLM_MAX_MODEL_LEN=65536 $REPO/integrations/opencode/tools/boot_vllm.sh s30"
MLX_RESTART="pkill -f mlx_lm.server; sleep 10; \
$REPO/integrations/opencode/tools/wait_for_memory.sh 26 300 || exit 1; \
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model $MLXMODEL --port 8001 --host 127.0.0.1 \
  >/tmp/mlx_s30.log 2>&1 & \
for i in \$(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && break; sleep 3; done"

log "=== 30-instance three-way: accuracy + throughput + latency ==="
log "instances: $(echo $INSTANCES | wc -w | tr -d ' ')  budget: ${BUDGET}s"

# `probe/...` routes opencode through turnlog on 8099.
arm pie  http://127.0.0.1:8080 "probe/qwen3-coder-30b" "$PIE_RESTART"
arm mlx  http://127.0.0.1:8001 "probe/$MLXMODEL"       "$MLX_RESTART"
arm vllm http://127.0.0.1:8000 "probe/qwen3-coder-30b" "$VLLM_RESTART"

for a in pie mlx vllm; do
    n=0; [ -f "$OUT/preds-$a.jsonl" ] && n=$(wc -l < "$OUT/preds-$a.jsonl" | tr -d ' ')
    [ "$n" -ge 25 ] || log "INCOMPLETE ARM: $a has only $n/30 predictions -- do not compare it"
done
log "=== transcript health per arm ==="
for a in pie mlx vllm; do
    [ -d "$OUT/wd-$a" ] && python3 tools/transcript_health.py "$OUT/wd-$a" 2>&1 | tail -6 | sed "s/^/  [$a] /" | tee -a "$OUT/run.log"
done
log "=== DONE (grade with tools/swe_grade_overnight.sh, SWE_OUT=$OUT) ==="
