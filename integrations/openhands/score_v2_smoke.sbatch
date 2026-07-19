#!/usr/bin/env bash
#SBATCH --job-name=score-v2-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_v2_smoke_%j.out

# Score SWE-bench predictions for PIE agent v2 smoke test (30B-MoE, 3 instances).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

echo "=== Scoring PIE Agent v2 smoke predictions ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/pie_agent_v2_smoke_20260714_232346.jsonl \
    --report-file predictions/pie_agent_v2_smoke_20260714_232346.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_v2_smoke \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Report: predictions/pie_agent_v2_smoke_20260714_232346.report.json"
