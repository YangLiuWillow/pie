#!/usr/bin/env bash
#SBATCH --job-name=score-t0-agent-v2
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_t0_agent_full_v2_%j.out

# Score the v2 full 50-instance temperature-0 pie-agent run (job 18526004).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/pie_agent_t0_full_v2.jsonl

echo "=== Scoring t0 agent full v2 (50 instances) ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_t0_agent_full_v2 \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: ${PRED%.jsonl}.report.json"
