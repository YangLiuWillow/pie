#!/usr/bin/env bash
#SBATCH --job-name=score-30b-moe
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_30b_moe_%j.out

# Score SWE-bench predictions for both PIE agent and baseline (30B-MoE).
# Uses Apptainer (no Docker needed). Results written to .report.json files
# next to each predictions JSONL.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes

mkdir -p "$SANDBOX_ROOT" logs

echo "=== Scoring PIE Agent predictions ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/pie_agent_qwen3_coder_30b_moe_50.jsonl \
    --report-file predictions/pie_agent_qwen3_coder_30b_moe_50.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_pie_agent \
    --timeout 3600

echo ""
echo "=== Scoring Baseline predictions ==="
echo "Start: $(date)"
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    predictions/baseline_qwen3_coder_30b_moe_v2_50.jsonl \
    --report-file predictions/baseline_qwen3_coder_30b_moe_v2_50.report.json \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_baseline \
    --timeout 3600

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "PIE Agent report:  predictions/pie_agent_qwen3_coder_30b_moe_50.report.json"
echo "Baseline report:   predictions/baseline_qwen3_coder_30b_moe_v2_50.report.json"
