#!/usr/bin/env bash
#SBATCH --job-name=score-30bcoder-v8
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_30bcoder_v8_%j.out

# Score the v8 coder-session + Qwen3-Coder-30B patch on django-14373 to
# confirm it resolves (litellm+OH resolves this instance; goal is parity).
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_30bcoder_v8
mkdir -p "$SANDBOX_ROOT" logs

PRED=predictions/traj_debug_30bcoder_v8_18949114.jsonl
PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_30bcoder_v8 \
    --timeout 3600

echo "Report: ${PRED%.jsonl}.report.json"
