#!/usr/bin/env bash
#SBATCH --job-name=score-litellm-base-30bcoder
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_litellm_base_30bcoder_13_%j.out
# Score the litellm-baseline rerun of the 13 instances. The predictions
# file path is filled in at submit time (sed) with the rerun job id.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_litellm_base_30bcoder
mkdir -p "$SANDBOX_ROOT" logs
PRED=predictions/litellm_base_30bcoder_13_18962143.jsonl
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_litellm_base_30bcoder \
    --timeout 3600
echo "Report: ${PRED%.jsonl}.report.json"
