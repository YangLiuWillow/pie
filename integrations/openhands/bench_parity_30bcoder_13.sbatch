#!/usr/bin/env bash
#SBATCH --job-name=parity-30bcoder-13
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/parity_30bcoder_13_%j.out

# Parity run: openhands-coder-session + Qwen3-Coder-30B-A3B (native
# <function=…> tool format, ToolFormat::Coder in the pie runtime) across
# the 13 instances that OpenHands+litellm resolved with the same model
# (baseline_t0_full_50, litellm+qwen3-coder-30b-a3b-t0). Goal: measure
# accuracy parity of the accelerated session inferlet vs. the litellm
# baseline. Validated end-to-end on django-14373 (v8, resolved=True).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_parity_ws_30bcoder_13

COMPLETIONS_DIR=logs/completions_parity_30bcoder_13_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-session+qwen3-coder-30b-t0-parity \
OUTPUT=predictions/parity_30bcoder_13_${SLURM_JOB_ID}.jsonl \
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
.venv/bin/python - "predictions/parity_30bcoder_13_${SLURM_JOB_ID}.jsonl" <<'PY'
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
