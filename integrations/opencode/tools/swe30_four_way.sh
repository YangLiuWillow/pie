#!/usr/bin/env bash
# Four engines, one model, the same 30 SWE-bench Verified instances.
#
#     Qwen3.6-35B-A3B  ·  pie / mlx-lm / vLLM-metal / llama.cpp
#
# ## What is different from swe30_three_way.sh
#
# **A fourth arm.** llama.cpp joins, and it is the one arm that CANNOT serve
# the same bytes as the others: pie, mlx-lm and vLLM all read
# `mlx-community/Qwen3.6-35B-A3B-4bit` (4-bit affine, group 64, 19.00 GiB),
# and no GGUF quantization is bit-equivalent to that. The nearest match by
# bits-per-weight is `UD-Q4_K_S` at 19.46 GiB -- 2.4% larger, which keeps the
# memory-bandwidth side of the throughput comparison honest, but it is
# imatrix-calibrated where the MLX quantization is not. That asymmetry favours
# llama.cpp on accuracy and cannot be removed; it is reported, not hidden.
#
# **A different model, and it is a THINKING model.** Qwen3.6's chat template
# prefills `<think>` unless `enable_thinking=false`. pie's renderer hard-codes
# the no-think cue, mlx-lm and vLLM follow the template default, and llama.cpp
# follows whichever flag it was booted with. Left alone, pie would answer a
# different prompt than the other three and the difference would land in the
# accuracy column attributed to the engine.
#
# So the proxy pins it: `turnlog.py --template-kwargs '{"enable_thinking":false}'`
# merges the field into EVERY chat request on every arm. One control point, and
# a request that loses the field is stamped `template_dropped` so the report can
# refuse to compare that arm rather than quietly averaging it in. Verified
# before the run: all four render this prompt to the same token count.
#
# Everything else is deliberately unchanged from the three-way run so the two
# are comparable: same 30 instances, same per-instance restarts, same proxy,
# same token-weighted throughput, same degeneration check.
#
# ## Speculation, which is asymmetric and was asymmetric before
#
# pie drafts with prompt-lookup (`DRAFT_K=4`); vLLM's equivalent ngram
# speculation is left off, as it was in the three-way run; mlx-lm and llama.cpp
# draft not at all here. This is inherited, not introduced, and it is the one
# knob that would most change the throughput column. It is in the caveats.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${SWE30_OUT:-/tmp/swe36}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
MLXMODEL=mlx-community/Qwen3.6-35B-A3B-4bit
PIEMODEL=qwen3.6-35b-a3b
GGUF="${LCPP_GGUF:-$HOME/models/qwen36-gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
NOTHINK='{"enable_thinking":false}'
BUDGET="${SWE30_BUDGET:-72000}"          # 20h; four arms on a 35B
TIMEOUT="${SWE30_TIMEOUT:-1800}"
INSTANCES="$(cat "${SWE30_INSTANCES:-$REPO/integrations/opencode/swe30-instances.txt}")"

[ -x "$REPO/target/release/pie" ] || { echo "FATAL: no pie binary" >&2; exit 1; }
[ -f "$GGUF" ] || { echo "FATAL: no GGUF at $GGUF" >&2; exit 1; }
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
    pkill -f "llama.cpp/build/bin/llama-server" 2>/dev/null
    pkill -f "turnlog.py" 2>/dev/null
    sleep 8
}
trap 'stop_all; log "stopped"' EXIT

# The proxy is started ONCE PER ARM, not per instance: it must survive the
# per-instance server restarts, since opencode's baseURL points at it and not at
# the engine.
start_proxy() {  # $1 arm, $2 upstream
    pkill -f "turnlog.py" 2>/dev/null; sleep 2
    nohup $HARNESS_PY tools/turnlog.py --upstream "$2" --port 8099 \
        --out "$OUT/calls-$1.jsonl" --tag "$1" --template-kwargs "$NOTHINK" \
        > "$OUT/proxy-$1.log" 2>&1 &
    for _ in $(seq 1 30); do
        curl -s -m 2 -o /dev/null "http://127.0.0.1:8099/v1/models" && return 0
        sleep 1
    done
    log "[$1] FATAL: proxy never came up"; return 1
}

arm() {  # $1 tag, $2 upstream, $3 model string, $4 restart cmd
    local tag=$1 upstream=$2 model=$3 restart=$4
    if [ "$(left)" -lt $(( TIMEOUT + 1200 )) ]; then
        log "SKIP $tag: only $(left)s left"; return 0; fi
    log "===== $tag ($(left)s budget left) ====="
    stop_all
    eval "$restart" >/dev/null 2>&1
    start_proxy "$tag" "$upstream" || return 1
    local t0; t0=$(date +%s)
    $HARNESS_PY run_swebench.py --instances $INSTANCES --model "$model" \
        --label "$tag" --out "$OUT/preds-$tag.jsonl" --timeout "$TIMEOUT" \
        --workdir "$OUT/wd-$tag" --restart-cmd "$restart" \
        > "$OUT/$tag.log" 2>&1
    local rc=$? n=0 calls=0 dropped=0
    [ -f "$OUT/preds-$tag.jsonl" ] && n=$(wc -l < "$OUT/preds-$tag.jsonl" | tr -d ' ')
    [ -f "$OUT/calls-$tag.jsonl" ] && calls=$(wc -l < "$OUT/calls-$tag.jsonl" | tr -d ' ')
    [ -f "$OUT/calls-$tag.jsonl" ] && dropped=$(grep -c '"template_dropped": true' "$OUT/calls-$tag.jsonl" 2>/dev/null || echo 0)
    log "$tag done rc=$rc wall=$(( $(date +%s) - t0 ))s predictions=$n calls=$calls"
    [ "$n" -gt 0 ] || log "WARNING: $tag produced NO predictions"
    [ "$calls" -gt 0 ] || log "WARNING: $tag recorded NO calls -- latency unavailable"
    # The prompt-shape guarantee, checked rather than assumed.
    [ "$dropped" = "0" ] || log "WARNING: $tag dropped enable_thinking on $dropped calls -- NOT COMPARABLE"
}

# Wait for the CONDITION, not a duration: `sleep 8` killed an earlier run at
# instance 3 and `sleep 30` at instance 14, because macOS RELEASES a 19 GiB
# model's pages well before it RECLAIMS them.
MEMWAIT="$REPO/integrations/opencode/tools/wait_for_memory.sh 26 300 || exit 1"

PIE_RESTART="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 10; \
$MEMWAIT; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh s36 \
  PIE_STRATEGY=b PIE_MODEL=$PIEMODEL PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY"
VLLM_RESTART="pkill -f 'vllm serve'; pkill -f 'VLLM::EngineCore'; sleep 10; \
$MEMWAIT; \
VLLM_MODEL=$MLXMODEL VLLM_SERVED_NAME=$PIEMODEL VLLM_MAX_MODEL_LEN=65536 \
  $REPO/integrations/opencode/tools/boot_vllm.sh s36"
MLX_RESTART="pkill -f mlx_lm.server; sleep 10; \
$MEMWAIT; \
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model $MLXMODEL --port 8001 --host 127.0.0.1 \
  >/tmp/mlx_s36.log 2>&1 & \
for i in \$(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && break; sleep 3; done"
LCPP_RESTART="pkill -f 'llama.cpp/build/bin/llama-server'; sleep 10; \
$MEMWAIT; \
LCPP_GGUF=$GGUF LCPP_CTX=65536 LCPP_SERVED=$PIEMODEL \
  $REPO/integrations/opencode/tools/boot_llamacpp.sh s36"

log "=== 30-instance four-way on Qwen3.6-35B-A3B: accuracy + throughput + latency ==="
log "instances: $(echo $INSTANCES | wc -w | tr -d ' ')  budget: ${BUDGET}s  no-think: $NOTHINK"

arm pie      http://127.0.0.1:8080 "probe/$PIEMODEL"  "$PIE_RESTART"
arm mlx      http://127.0.0.1:8001 "probe/$MLXMODEL"  "$MLX_RESTART"
arm vllm     http://127.0.0.1:8000 "probe/$PIEMODEL"  "$VLLM_RESTART"
arm llamacpp http://127.0.0.1:8002 "probe/$PIEMODEL"  "$LCPP_RESTART"

for a in pie mlx vllm llamacpp; do
    n=0; [ -f "$OUT/preds-$a.jsonl" ] && n=$(wc -l < "$OUT/preds-$a.jsonl" | tr -d ' ')
    [ "$n" -ge 25 ] || log "INCOMPLETE ARM: $a has only $n/30 predictions -- do not compare it"
done
log "=== transcript health per arm ==="
for a in pie mlx vllm llamacpp; do
    [ -d "$OUT/wd-$a" ] && python3 tools/transcript_health.py "$OUT/wd-$a" 2>&1 | tail -6 | sed "s/^/  [$a] /" | tee -a "$OUT/run.log"
done
log "=== DONE (grade with tools/swe_grade_overnight.sh, SWE_OUT=$OUT) ==="
