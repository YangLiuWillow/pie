#!/usr/bin/env bash
#SBATCH --job-name=score-t0-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_t0_smoke_%j.out

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

echo "=== Scoring baseline temp0 smoke ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/baseline_t0_smoke_20260715_004144.jsonl \
    --report-file predictions/baseline_t0_smoke_20260715_004144.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_baseline_t0_smoke \
    --timeout 3600

echo ""
echo "=== Scoring agent temp0 smoke ==="
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/pie_agent_t0_smoke_20260715_005947.jsonl \
    --report-file predictions/pie_agent_t0_smoke_20260715_005947.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_agent_t0_smoke \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
