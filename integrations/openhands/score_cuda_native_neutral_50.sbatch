#!/usr/bin/env bash
#SBATCH --job-name=score-neutral-50
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_neutral_50_%j.out

# Score the neutral-50 Pie arm (bench_cuda_native_neutral_50.sbatch), then
# print the head-to-head against the already-scored litellm baseline on the
# same 50 instances.
#
# 24h, not the 8h used for the 13-instance scorer: each instance needs an
# Apptainer sandbox built and a test suite run (the 13-instance score took
# ~1h35m), so 50 needs roughly 4x that, with headroom for slow repos.
# score_instance skips instances whose report.json already exists and
# ensure_sandbox reuses built sifs, so a requeue resumes cheaply.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_neutral_50
mkdir -p "$SANDBOX_ROOT" logs
PRED=predictions/neutral_50_pie.jsonl
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_neutral_50 \
    --timeout 3600
echo "Report: ${PRED%.jsonl}.report.json"

echo ""
echo "=== HEAD-TO-HEAD on the neutral 50 ==="
$VENV/bin/python - <<'PY'
import json
pie  = json.load(open("predictions/neutral_50_pie.report.json"))
base = json.load(open("predictions/baseline_t0_full_50.report.json"))
p, b = set(pie["resolved_ids"]), set(base["resolved_ids"])
print(f"pie (cuda_native+APC): {len(p)}/{pie['total']}")
print(f"litellm+vLLM baseline: {len(b)}/{base['total']}")
print(f"both:        {len(p & b)}")
print(f"only pie:    {sorted(p - b)}")
print(f"only litellm:{sorted(b - p)}")
PY
