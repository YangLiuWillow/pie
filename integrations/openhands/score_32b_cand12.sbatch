#!/usr/bin/env bash
#SBATCH --job-name=score-32b-cand12
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_32b_cand12_%j.out

# Score the 12 pie_agent_32b_200 predictions whose instances were resolved
# by Qwen3-Coder-30B in earlier evals AND got a non-empty patch from
# Qwen2.5-Coder-32B (Pattern A). Goal: find instances Qwen2.5-Coder-32B
# genuinely solves, to use as machinery-validation targets for the
# OpenHands pie-session trajectory debugging.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_cand12

mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/pie_agent_32b_cand12.jsonl

echo "=== Scoring 32B agent candidate-12 subset ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_32b_cand12 \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: ${PRED%.jsonl}.report.json"
