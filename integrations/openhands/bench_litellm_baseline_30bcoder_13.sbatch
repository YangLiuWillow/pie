#!/usr/bin/env bash
#SBATCH --job-name=litellm-base-30bcoder-13
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/litellm_base_30bcoder_13_%j.out
# TRUE BASELINE RERUN: OpenHands + litellm (real harness, native tool
# calling) with Qwen3-Coder-30B-A3B served by vLLM (qwen3_coder tool-call
# parser). Reruns the exact 13 instances that the original t0 baseline
# (baseline_t0_full_50, litellm+qwen3-coder-30b-a3b-t0, 2026-07-15)
# resolved, at temperature 0 — the accuracy reference the accelerated
# coder-session inferlet (bench_parity_30bcoder_13, job 18951267) must
# match. Same model, same temperature (0), same native tool calling.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
export MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export LABEL=litellm+qwen3-coder-30b-a3b-t0-rerun
export HARNESS=benchmarks.run_swe_bench
export OUTPUT=predictions/litellm_base_30bcoder_13_${SLURM_JOB_ID}.jsonl
COMPLETIONS_DIR=logs/completions_litellm_base_30bcoder_13_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR" logs predictions
bash run_litellm_baseline.sh \
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
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"
echo ""
echo "=== predictions ==="
ls -l "$OUTPUT" 2>/dev/null || echo "(no output written)"
echo ""
echo "=== non-empty patch summary ==="
.venv/bin/python - "$OUTPUT" <<'PY'
import json,sys
n=ne=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); n+=1
    p=d.get("model_patch") or d.get("prediction") or ""
    if p.strip(): ne+=1
    print(d.get("instance_id"), "NONEMPTY(%d)"%len(p) if p.strip() else "empty")
print("total=%d nonempty=%d"%(n,ne))
PY
