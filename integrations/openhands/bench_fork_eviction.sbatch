#!/usr/bin/env bash
#SBATCH --job-name=fork-eviction
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/fork_eviction_%j.out

# Fork-eviction microbench: does vLLM+APC evict the SHARED TRUNK under KV
# pressure (the cost Pie's refcounted fork avoids)? Boots vllm TWICE — a
# PRESSURE config (small KV) and a ROOMY control (large KV) — same K/trunk, and
# measures per-call cached_tokens => trunk re-prefill count. Design/logic:
# bench_fork_eviction.py header.

set -uo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

PIE_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
mkdir -p logs

VLLM_PORT=${VLLM_PORT:-18000}
MODEL_HF=${MODEL_HF:-Qwen/Qwen3-Coder-30B-A3B-Instruct}

export MODEL="$MODEL_HF"
export BASE_URL="http://localhost:${VLLM_PORT}"
export TRUNK_TOKENS=${TRUNK_TOKENS:-16000}
export K=${K:-24}
export ROUNDS=${ROUNDS:-4}
export DECODE_TOKENS=${DECODE_TOKENS:-64}
export SUFFIX_TOKENS=${SUFFIX_TOKENS:-600}
export CONCURRENCY=${CONCURRENCY:-1}

boot_vllm() {  # $1 = gpu_mem_util, $2 = tag
    local util="$1" tag="$2"
    local log="logs/vllm_evict_${tag}_${SLURM_JOB_ID}.log"
    echo "[boot:$tag] vllm serve gpu_mem_util=$util ..."
    PYTHONPATH="" HF_HOME=$HF_HOME \
      $PIE_VENV/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_HF" --port $VLLM_PORT \
        --enable-prefix-caching \
        --gpu-memory-utilization "$util" \
        --max-model-len 40960 \
        --enforce-eager \
        --disable-log-requests \
        > "$log" 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 1800); do
        curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q "200" && { echo "  up ($((i*2))s)"; return 0; }
        kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vllm($tag) died"; tail -30 "$log"; return 1; }
        sleep 2
    done
    echo "ERROR: vllm($tag) not ready"; tail -30 "$log"; return 1
}
kill_vllm() { kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; sleep 5; }
trap 'kill_vllm' EXIT

# --- PRESSURE: small KV (weights ~60GB on 96GB Blackwell; 0.72 => ~9GB KV) ---
if boot_vllm "${PRESSURE_UTIL:-0.72}" pressure; then
    echo "[run:pressure] fork eviction K=$K trunk=$TRUNK_TOKENS ..."
    OUT_JSON=logs/fork_eviction_pressure_${SLURM_JOB_ID}.json \
      PYTHONPATH="" $PIE_VENV/bin/python bench_fork_eviction.py
fi
kill_vllm

# --- ROOMY control: large KV (0.92 => ~28GB KV) ---
if boot_vllm "${ROOMY_UTIL:-0.92}" roomy; then
    echo "[run:roomy] fork eviction K=$K trunk=$TRUNK_TOKENS ..."
    OUT_JSON=logs/fork_eviction_roomy_${SLURM_JOB_ID}.json \
      PYTHONPATH="" $PIE_VENV/bin/python bench_fork_eviction.py
fi
kill_vllm

echo ""
echo "=== SUMMARY (pressure vs roomy) ==="
for tag in pressure roomy; do
    f=logs/fork_eviction_${tag}_${SLURM_JOB_ID}.json
    [ -f "$f" ] && PYTHONPATH="" $PIE_VENV/bin/python - "$f" "$tag" <<'PY'
import json, sys
r=json.load(open(sys.argv[1])); tag=sys.argv[2]
print(f"[{tag}] K={r['config']['K']} trunk={r['config']['trunk_tokens']} "
      f"trunk_reprefills={r['trunk_reprefills']}/{r['calls']} "
      f"(first-call {r['first_call_trunk_reprefills']}/{r['first_calls']}), "
      f"hit_rate={r['prefix_cache_hit_rate_delta']} preempt={r['preemptions_delta']} "
      f"vLLM_prefill={r['vllm_real_prefill_tokens']} vs Pie_fork={r['pie_fork_real_prefill_tokens']} "
      f"({r['prefill_ratio_vllm_over_pie']}x)")
PY
done
echo "=== done ==="
