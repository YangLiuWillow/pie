#!/usr/bin/env bash
#SBATCH --job-name=score-fork-bestofk
#SBATCH --account=pi_ql324
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/score_fork_bestofk_%j.out

# Step-3 SCORING: score each fork candidate file, then compute best-of-K.
# Pass the generation OUT path via PRED=predictions/fork_bestofN_<jid>.jsonl
# (bench_fork_bestofk.sbatch already ran `bestofk.py split`, producing
# PRED.cand0.jsonl .. PRED.cand{K-1}.jsonl).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

VENV=.venv
HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
SANDBOX_ROOT=/nfs/roberts/scratch/pi_ql324/ly337/swebench_sandboxes_fork_bestofk
mkdir -p "$SANDBOX_ROOT" logs

: "${PRED:?set PRED=predictions/fork_bestofN_<jobid>.jsonl}"
STEM="${PRED%.jsonl}"

REPORTS=()
for CAND in "$STEM".cand*.jsonl; do
    [ -e "$CAND" ] || { echo "no candidate files for $STEM"; exit 1; }
    REPORT="${CAND%.jsonl}.report.json"
    echo "=== scoring $CAND ==="
    PYTHONPATH=vendor/benchmarks OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
      $VENV/bin/python score_swebench_apptainer.py \
        "$CAND" \
        --report-file "$REPORT" \
        --sandbox-root "$SANDBOX_ROOT" \
        --score-dir "predictions/apptainer_eval_fork_$(basename "${CAND%.jsonl}")" \
        --timeout 3600
    REPORTS+=("$REPORT")
done

echo ""
echo "=== best-of-K combine ==="
PYTHONPATH="" $VENV/bin/python bestofk.py combine "${REPORTS[@]}"
