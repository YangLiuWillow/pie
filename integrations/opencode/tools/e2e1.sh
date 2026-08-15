#!/usr/bin/env bash
# One SWE-bench instance, end to end, on all three engines, decomposed.
#
# ## What this answers that `swe3.sh` does not
#
# `swe3.sh` reports agent wall clock per arm. That number has already misled
# once on this benchmark: the vLLM arm's 10-second runs read as "faster" when
# they were an agent giving up after two calls. Wall clock is
#
#     turns x (prefill + decode)
#
# and an engine can lose it three unrelated ways. Every arm here runs through
# `tools/turnlog.py`, so each one comes back with its turn count, its TTFT per
# turn, its decode rate per turn, and the prompt/completion tokens the server
# itself reported -- the same fields, from the same clock, for pie, vLLM-metal
# and mlx-lm.
#
# ## Why ONE instance and why this one
#
# django__django-14373 is the only member of the known-5 that **both** pie and
# vLLM-metal resolved (results-swebench.md: 309 s and 39 s). On every other
# instance at least one arm produced no patch, so its wall clock is the time an
# agent took to give up and comparing it to a working arm's is meaningless.
# Here both arms did the job, so the times are comparable and the gap is real.
#
# Single instance is deliberate: n=1 cannot support a correctness claim and is
# not asked to. Correctness is `swe3.sh` plus `grade.sh` over the full known-5.
# This measures SPEED on a case known to work, which is the thing being
# optimised, and it runs in minutes rather than an hour.
#
# ## The standing rule
#
# Two 30B servers do not fit in 48 GB (measured -- a vLLM arm OOM'd and
# dead-latched on 2026-08-14 while a peer held 12.5 GB). So: boot, measure,
# kill, next. `roofline_probe` gates each arm, because a contended machine does
# not produce a void measurement, it produces a TILTED one.
#
# Usage:  bash tools/e2e1.sh [instance-id]
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${E2E_OUT:-/tmp/e2e1-$(date +%m%d-%H%M)}"
INST="${1:-django__django-14373}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
ROOFLINE=/tmp/metaltools/bin/roofline_probe
PROXY_PORT=8099

mkdir -p "$OUT"
cd "$REPO/integrations/opencode"
# shellcheck source=require_quiet_gpu.sh
source tools/require_quiet_gpu.sh

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    pkill -f "\.venv-vllm-metal/bin/vllm" 2>/dev/null
    pkill -f "VLLM::" 2>/dev/null
    pkill -f mlx_lm.server 2>/dev/null
    pkill -f "tools/turnlog.py" 2>/dev/null
    sleep 6
}

# The roof, not the process list. On 2026-08-15 a background job dropped it to
# 69.8 GB/s while leaving free memory ample and CPU near idle, and every
# memory-bound number inflated 1.3-2.2x while every register-bound one did not.
# The line is "streaming roof (read-only, 512 MB): 1.802 ms -> 297.9 GB/s", so
# the number is $(NF-1) and $NF is the unit. The first version of this took $NF
# and compared the string "GB/s" against 250 -- which awk resolves
# LEXICALLY ("G" > "2"), so the gate returned OK for every machine state
# including a contended one. It ran three arms while stamping them checked.
#
# That is trap §9.1 -- "an instrument's silence is not a measurement" --
# reproduced inside a gate written to prevent it. So the parse is now asserted
# to be a number before it is compared to one, and an unparseable roof is fatal
# rather than permissive.
check_roof() {
    local roof
    roof=$($ROOFLINE 2>/dev/null | awk '/streaming roof/{print $(NF-1)}')
    if ! printf '%s' "$roof" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "FATAL: could not parse a roof from roofline_probe (got '${roof}')." >&2
        echo "       Refusing to run: an unchecked machine is worse than none," >&2
        echo "       because contention TILTS an A/B rather than breaking it." >&2
        return 1
    fi
    echo "── streaming roof: ${roof} GB/s"
    if awk -v r="$roof" 'BEGIN{exit !(r + 0 < 250)}'; then
        echo "FATAL: roof ${roof} GB/s is far below this machine's ~298." >&2
        echo "       Something is contending for memory. Every number taken" >&2
        echo "       now would be inflated, and asymmetrically." >&2
        return 1
    fi
    return 0
}

start_proxy() {  # $1 tag, $2 upstream
    rm -f "$OUT/turns-$1.jsonl"
    nohup python3 tools/turnlog.py --upstream "$2" --port "$PROXY_PORT" \
        --out "$OUT/turns-$1.jsonl" --tag "$1" > "$OUT/turnlog-$1.log" 2>&1 &
    for _ in $(seq 1 20); do
        curl -s -m 2 -o /dev/null "http://127.0.0.1:$PROXY_PORT/v1/models" && return 0
        sleep 1
    done
    echo "FATAL: turnlog never answered on :$PROXY_PORT" >&2
    cat "$OUT/turnlog-$1.log" >&2
    return 1
}

run_arm() {  # $1 tag, $2 model id as the UPSTREAM expects it
    echo "----- running $1 on $INST -----"
    local t0; t0=$(date +%s)
    $HARNESS_PY run_swebench.py --instances "$INST" --model "probe/$2" \
        --label "$1" --out "$OUT/preds-$1.jsonl" --timeout 2400 \
        > "$OUT/$1.log" 2>&1
    echo "[$1] agent wall $(( $(date +%s) - t0 ))s"
    grep -aE "resolved|patch|empty|FAIL|error" "$OUT/$1.log" | tail -5
    python3 tools/turn_summary.py "$OUT/turns-$1.jsonl"
}

echo "=============================================================="
echo " instance : $INST"
echo " out      : $OUT"
echo "=============================================================="

# ── pie, strategy B ──
stop_all
require_quiet_gpu 20 || exit 1
check_roof || exit 1
PIE_PYTHON=$PIEPY tools/boot_pie.sh e2eB \
    PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
    PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY 2>&1 | tail -1 || exit 1
start_proxy pie http://127.0.0.1:8080 || exit 1
run_arm pie qwen3-coder-30b

# ── vLLM-metal ──
stop_all
require_quiet_gpu 20 || exit 1
check_roof || exit 1
VLLM_MAX_MODEL_LEN=65536 tools/boot_vllm.sh e2ev 2>&1 | tail -1 || exit 1
start_proxy vllm http://127.0.0.1:8000 || exit 1
run_arm vllm qwen3-coder-30b

# ── mlx-lm ──
stop_all
require_quiet_gpu 20 || exit 1
check_roof || exit 1
nohup /tmp/venv-mlxlm/bin/mlx_lm.server \
    --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
    --port 8001 --host 127.0.0.1 > "$OUT/mlxlm.log" 2>&1 &
for _ in $(seq 1 60); do
    curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && break
    sleep 3
done
echo "mlx-lm up"
start_proxy mlx http://127.0.0.1:8001 || exit 1
run_arm mlx mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

stop_all
echo "=============================================================="
echo "DONE. Artifacts in $OUT"
ls -la "$OUT"
