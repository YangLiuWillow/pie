#!/usr/bin/env bash
#SBATCH --job-name=score-parity-30bcoder
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_parity_30bcoder_13_%j.out

# Score the 13-instance coder-session + Qwen3-Coder-30B parity run against
# the litellm+OH baseline (which resolved all 13 with the same model).
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_parity_30bcoder
mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/parity_30bcoder_13_18951267.jsonl
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_parity_30bcoder \
    --timeout 3600

echo "Report: ${PRED%.jsonl}.report.json"
