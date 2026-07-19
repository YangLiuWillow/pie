#!/usr/bin/env bash
#SBATCH --job-name=score-32b-base-200
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_32b_baseline_200_%j.out

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/baseline_32b_200.jsonl

echo "=== Scoring 32B baseline (200 instances) ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_32b_baseline_200 \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: ${PRED%.jsonl}.report.json"
