#!/usr/bin/env bash
#SBATCH --job-name=score-v9-pyparser-13028
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_v9_pyparser_13028_%j.out

# Score the Option-B (python-tool-parser) v9 debug patch for django-13028.
# Reuses the litellm scoring run's already-built django-13028 Apptainer
# sandbox (ensure_sandbox returns immediately when sandbox_root/<id> exists),
# so this only runs the eval, ~a few min. Fresh score-dir so THIS patch is
# scored (not a cached report).
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_litellm_base_30bcoder
PRED=predictions/traj_debug_30bcoder_v9_pyparser_19019373.jsonl

PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python score_swebench_apptainer.py \
    "$PRED" \
    --report-file "${PRED%.jsonl}.report.json" \
    --sandbox-root "$SANDBOX_ROOT" \
    --score-dir predictions/apptainer_eval_v9_pyparser_13028 \
    --timeout 3600

echo "Report: ${PRED%.jsonl}.report.json"
cat "${PRED%.jsonl}.report.json"
