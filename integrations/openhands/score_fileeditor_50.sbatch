#!/usr/bin/env bash
#SBATCH --job-name=score-fe-50
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_fileeditor_50_%j.out

# Score SWE-bench predictions for PIE agent with FileEditor (30B-MoE, 50 instances).
# Uses Apptainer (no Docker needed).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

echo "=== Scoring PIE Agent FileEditor 50 predictions ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/pie_agent_30b_moe_fileeditor_50_20260714_090457.jsonl \
    --report-file predictions/pie_agent_30b_moe_fileeditor_50_20260714_090457.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_fileeditor_50 \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: predictions/pie_agent_30b_moe_fileeditor_50_20260714_090457.report.json"
