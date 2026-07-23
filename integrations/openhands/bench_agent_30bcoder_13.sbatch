#!/usr/bin/env bash
#SBATCH --job-name=agent-30bcoder-13
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/agent_30bcoder_13_%j.out

# Pattern A (openhands-agent = full loop INSIDE the inferlet, live ctx across
# turns, tools via HTTP to a host tool server) on the SAME 13 instances as the
# coder-session parity run (job 18951267) and the litellm baseline (18962143).
#
# WHY: the coder-session (Pattern B) is stateless-per-turn — it re-opens/
# refreshes the session snapshot every LLM call. Our decode-tps diagnostic
# showed Pie's prefill (saved 95-97%) and decode (~parity) are NOT the source
# of the 3.65x OpenHands slowdown, leaving per-call session-management overhead
# as the suspect. Pattern A keeps the ctx LIVE across turns (no per-call
# open/refresh/save), so if that overhead was the culprit, per-iteration
# latency should drop sharply here. This isolates it.
#
# NOTE: Pattern A uses a simpler hand-rolled loop (not the full OpenHands SDK
# agent the coder-session drives), so RESOLVE RATE may differ — this run is
# about per-iteration LATENCY, not accuracy parity. Compare s/iter, not just
# pass count.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18081
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_agent_ws_30bcoder_13

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-t0-13 \
OUTPUT=predictions/agent_30bcoder_13_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=agent_30bcoder_13 \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-12276 \
    --instance-id django__django-13028 \
    --instance-id django__django-13089 \
    --instance-id django__django-14373 \
    --instance-id django__django-15569 \
    --instance-id django__django-16485 \
    --instance-id matplotlib__matplotlib-22719 \
    --instance-id pydata__xarray-4075 \
    --instance-id pydata__xarray-4966 \
    --instance-id scikit-learn__scikit-learn-10908 \
    --instance-id scikit-learn__scikit-learn-12973 \
    --instance-id scikit-learn__scikit-learn-13496 \
    --instance-id sympy__sympy-19346 \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== non-empty patch summary ==="
.venv/bin/python - "predictions/agent_30bcoder_13_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
n=ne=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    n+=1; ne+= 1 if p else 0
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
print(f"\nTOTAL {n}  non-empty {ne}")
PY
