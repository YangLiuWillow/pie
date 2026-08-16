#!/usr/bin/env bash
# pie (now) vs pie (original) vs mlx-lm vs vLLM-metal, in one session.
#
# ## The standing rule, and why "original pie" is an arm here
#
# Two 30B servers do not fit in 48 GB -- measured, not assumed -- so every arm
# boots, is measured, and is killed before the next one starts. And no number
# for another engine is ever quoted from memory: if mlx-lm appears in a table,
# mlx-lm ran today, on this machine, next to the others.
#
# "Original pie" is the same binary with the three kernels this work added
# switched off:
#
#     PIE_METAL_SDPA_HSHARE=0   GQA head sharing in decode attention
#     PIE_METAL_SDPA_NAX=0      fused prefill attention on the accelerators
#     PIE_METAL_QMM_NAX=0       both quantized GEMMs on the accelerators
#
# That is a stronger control than checking out an old commit: same binary, same
# build flags, same config, same server, so the ONLY difference is the kernels.
# It is also the honest test of those switches -- if any of them half-applied,
# this arm is where it would show.
#
# ## Two measurements, both deterministic
#
#   * `rate_probe.py`  -- TTFT and decode rate at three fixed prompt sizes.
#   * `bench_ab.py`    -- the canned 6-turn agentic replay.
#
# Neither drives a live agent, deliberately: `tools/pie_ab.sh` established that
# one agentic run cannot price anything, because opencode's own prompt varies
# and the agent then takes 3 turns or 19.
#
# ## Drift
#
# Four arms take ~20 minutes and a thermal drift over that window would line up
# with the engine axis. `matched_spec.sh` solves this by interleaving; with four
# arms that doubles the run. Instead the FIRST arm is repeated LAST, so the
# drift over the whole session is measured rather than assumed. If the repeat
# disagrees with the original by more than a few percent, the ordering matters
# and the run should be redone interleaved.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${FOURWAY_OUT:-/tmp/fourway-$(date +%m%d-%H%M)}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
ROOFLINE=/tmp/metaltools/bin/roofline_probe
MLXMODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

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
    sleep 6
}

check_roof() {  # $1 label -- records the roof per arm so drift is visible
    local roof
    roof=$($ROOFLINE 2>/dev/null | awk '/streaming roof/{print $(NF-1)}')
    if ! printf '%s' "$roof" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "FATAL: unparseable roof for $1 (got '${roof}')" >&2; return 1
    fi
    echo "── [$1] streaming roof ${roof} GB/s"
    echo "$1 $roof" >> "$OUT/roofs.txt"
    awk -v r="$roof" 'BEGIN{exit !(r + 0 < 250)}' && {
        echo "FATAL: roof ${roof} is far below this machine's ~296." >&2; return 1
    }
    return 0
}

measure() {  # $1 label, $2 base-url, $3 model
    python3 tools/rate_probe.py --base-url "$2" --model "$3" --label "$1" \
        --json "$OUT/rate-$1.json"
    $HARNESS_PY bench_ab.py --arm "$1" --base-url "$2" --model "$3" \
        --turns 6 --max-tokens 256 --out "$OUT/ab-$1.json" 2>&1 \
        | grep -E "TOTAL|turn 1:" | sed 's/^/    /'
}

boot_pie_arm() {  # $1 label, $2..$4 the three switches
    PIE_PYTHON=$PIEPY tools/boot_pie.sh "fw$1" \
        PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
        PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY \
        PIE_METAL_SDPA_HSHARE="$2" PIE_METAL_SDPA_NAX="$3" \
        PIE_METAL_QMM_NAX="$4" > /dev/null 2>&1
}

arm_pie() {  # $1 label, $2..$4 switches
    stop_all; require_quiet_gpu 20 || return 1; check_roof "$1" || return 1
    echo "===== $1 ====="
    boot_pie_arm "$1" "$2" "$3" "$4" || { echo "[$1] BOOT FAILED"; return 1; }
    measure "$1" http://127.0.0.1:8080 qwen3-coder-30b
}

echo "=============================================================="
echo " four-way: pie(now) / pie(original) / mlx-lm / vLLM-metal"
echo " out: $OUT"
echo "=============================================================="

arm_pie pie_now 1 1 1
arm_pie pie_orig 0 0 0

# ── mlx-lm ──
stop_all; require_quiet_gpu 20 || exit 1; check_roof mlx || exit 1
echo "===== mlx ====="
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model "$MLXMODEL" --port 8001 \
    --host 127.0.0.1 > "$OUT/mlx.log" 2>&1 &
for _ in $(seq 1 60); do
    curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && break; sleep 3
done
measure mlx http://127.0.0.1:8001 "$MLXMODEL"

# ── vLLM-metal ──
stop_all; require_quiet_gpu 20 || exit 1; check_roof vllm || exit 1
echo "===== vllm ====="
VLLM_MAX_MODEL_LEN=65536 tools/boot_vllm.sh fwv > "$OUT/vllm-boot.log" 2>&1 \
    || { echo "[vllm] BOOT FAILED"; tail -5 "$OUT/vllm-boot.log"; }
if curl -s -m 5 -o /dev/null http://127.0.0.1:8000/v1/models; then
    measure vllm http://127.0.0.1:8000 qwen3-coder-30b
else
    echo "  [vllm] not serving -- arm VOID, and reported as void rather than omitted"
fi

# ── the drift control: arm 1 again, last ──
arm_pie pie_now_repeat 1 1 1

stop_all
echo "=============================================================="
echo "DONE. $OUT"
cat "$OUT/roofs.txt" 2>/dev/null
