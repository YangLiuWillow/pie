#!/usr/bin/env bash
#SBATCH --job-name=vllm-fork-baseline
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/vllm_fork_baseline_%j.out

# The FAIR vLLM baseline for fork test-time scaling. Boots `vllm serve` with
# PREFIX CACHING ON (the incumbent's best trick — never disabled), then runs the
# SAME mid-trajectory K-way branch structure the Pie fork run uses, via
# bench_vllm_fork_baseline.py. Head-to-head against the Pie fork gen job on the
# identical instances / branch point. Design: docs/VLLM_FORK_BASELINE_DESIGN.md.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
PIE_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_vllm_fork_ws_${SLURM_JOB_ID}
mkdir -p "$SWEB_WS_ROOT" logs predictions

VLLM_PORT=${VLLM_PORT:-18000}
MODEL_HF=${MODEL_HF:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
VLLM_LOG=logs/vllm_serve_fork_baseline_${SLURM_JOB_ID}.log

# Branch structure — MUST match the Pie fork gen run for a fair head-to-head.
export NUM_BRANCHES=${NUM_BRANCHES:-3}
export BRANCH_AT_STEP=${BRANCH_AT_STEP:-5}
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-0.95}
export MAX_ITERS=${MAX_ITERS:-60}
export INSTANCE_IDS=${INSTANCE_IDS:-django__django-14373,scikit-learn__scikit-learn-12973,django__django-15569,matplotlib__matplotlib-22719}
export BACKEND=litellm
export MODEL="openai/${MODEL_HF}"
export BASE_URL="http://localhost:${VLLM_PORT}/v1"
export LITELLM_API_KEY=dummy
export OUTPUT=predictions/vllm_fork_baseline_${NUM_BRANCHES}_${SLURM_JOB_ID}.jsonl

echo "[boot] vllm serve (prefix caching ON) model=$MODEL_HF ..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE_VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    --port $VLLM_PORT \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; rm -rf "$SWEB_WS_ROOT" 2>/dev/null || true; }
trap cleanup EXIT

echo "[wait] vllm readiness..."
READY=0
for i in $(seq 1 1800); do
    if curl -s "http://localhost:$VLLM_PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q "200"; then
        echo "  vllm up ($((i*2))s)"; READY=1; break
    fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vllm died"; tail -30 "$VLLM_LOG"; exit 1; }
    sleep 2
done
[ "$READY" -eq 1 ] || { echo "ERROR: vllm not ready in time"; tail -30 "$VLLM_LOG"; exit 1; }

echo "[run] fork baseline (K=$NUM_BRANCHES, branch_at_step=$BRANCH_AT_STEP, temp=$TEMPERATURE)..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python bench_vllm_fork_baseline.py
RC=$?

echo ""
echo "=== splitting candidates into K scoreable files ==="
PYTHONPATH="" $VENV/bin/python bestofk.py split "$OUTPUT" || true

echo ""
echo "=== candidate summary ==="
PYTHONPATH="" $VENV/bin/python - "$OUTPUT" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    r=json.loads(line); cs=(r.get("_metadata") or {}).get("candidates") or []
    sizes=[len(c.get("model_patch","")) for c in cs]
    pt=[c.get("prompt_tokens",0) for c in cs]
    dist=len({c.get("model_patch","") for c in cs})
    print(f"{r['instance_id']}: {len(cs)} cands, patch_sizes={sizes}, {dist} distinct, "
          f"branch_prefill_tok={pt} (sum={sum(pt)})")
PY
echo ""
echo "Predictions: $OUTPUT"
echo "=== exit code: $RC ==="
exit $RC
