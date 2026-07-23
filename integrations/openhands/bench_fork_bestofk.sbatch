#!/usr/bin/env bash
#SBATCH --job-name=fork-bestofk-gen
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/fork_bestofk_gen_%j.out

# Step-3 GENERATION: fork test-time scaling on SWE-bench via the pie-agent
# (Pattern A) inferlet. Runs num_branches candidates per instance (trunk to
# branch_at_step, then fork KV+workspace); writes ONE predictions.jsonl whose
# _metadata.candidates holds the K per-branch patches, then splits it into K
# scoreable files (bestofk.py split). Score with score_fork_bestofk.sbatch.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_fork_bestofk_ws_${SLURM_JOB_ID}
mkdir -p "$SWEB_WS_ROOT" logs predictions

NUM_BRANCHES=${NUM_BRANCHES:-3}
BRANCH_AT_STEP=${BRANCH_AT_STEP:-5}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_ITERS=${MAX_ITERS:-60}
OUT=predictions/fork_bestof${NUM_BRANCHES}_${SLURM_JOB_ID}.jsonl

echo "[build] openhands-agent wasm..."
( cd /nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent \
    && cargo build --release --target wasm32-wasip2 )

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent-fork+qwen3-coder-30b \
OUTPUT="$OUT" \
REQUEST_TIMEOUT_S=3600 \
PIE_PORT=$PIE_PORT \
  bash run_pie_backend.sh \
    --instance-id django__django-14373 \
    --instance-id scikit-learn__scikit-learn-12973 \
    --instance-id django__django-15569 \
    --instance-id matplotlib__matplotlib-22719 \
    --num-branches "$NUM_BRANCHES" \
    --branch-at-step "$BRANCH_AT_STEP" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --instance-timeout-s 2400 \
    --idle-timeout-s 600 \
    --max-iterations "$MAX_ITERS"

echo ""
echo "=== splitting candidates into K scoreable files ==="
PYTHONPATH="" .venv/bin/python bestofk.py split "$OUT"

echo ""
echo "=== candidate patch summary ==="
.venv/bin/python - "$OUT" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    r=json.loads(line); cs=(r.get("_metadata") or {}).get("candidates") or []
    sizes=[len(c.get("model_patch","")) for c in cs]
    dist=len({c.get("model_patch","") for c in cs})
    print(f"{r['instance_id']}: {len(cs)} cands, sizes={sizes}, {dist} distinct")
PY
echo ""
echo "Predictions: $OUT  (score cand files with score_fork_bestofk.sbatch)"
