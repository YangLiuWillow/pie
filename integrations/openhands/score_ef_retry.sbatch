#!/usr/bin/env bash
#SBATCH --job-name=score-ef-retry
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_ef_retry_%j.out

# Score empty-finish retry prediction (sympy__sympy-23413 with 48K context).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/pie_agent_ef_retry_20260715_061148.jsonl

echo "=== Scoring empty-finish retry (sympy) ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_ef_retry \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: ${PRED%.jsonl}.report.json"
