#!/usr/bin/env bash
#SBATCH --job-name=pie-neutral-50
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/neutral_50_%j.out

# PIE ARM ON A NEUTRAL INSTANCE SET.
#
# Why this run exists: every accuracy comparison so far used the 13 instances
# that ARE the litellm baseline's own resolved_ids, so the baseline scores
# 13/13 by construction and the ceiling for Pie is parity — that set can never
# show Pie more accurate, only less. This run uses the full 50-instance subset
# those 13 were drawn from, which was sampled before any Pie result existed.
#
# The litellm half of this comparison is ALREADY DONE and needs no rerun:
#   predictions/baseline_t0_full_50.report.json  (job 18397164, 9h03m)
#   = 13/50 resolved, Qwen3-Coder-30B-A3B-Instruct, temperature 0,
#     max-iterations 100, native tool calling w/ the qwen3_coder parser.
# Everything below matches those settings so only the backend differs.
#
# Instance ids are read from that baseline report rather than regenerated via
# --subset-size 50. Both routes should agree (deterministic_subset_indices is
# a fixed-seed sample), but reading the ids makes the two arms provably the
# same set regardless of dataset version or seed drift.
#
# Config is the validated path from the 13-instance A/B (job 19251912): native
# CUDA driver, self-keyed content-addressed APC, host qwen3_coder tool parser,
# the snapshot-leak fix (f1aac6eb) and the WebSocket Close-frame fix (e4626e5b).
#
# Walltime 16h against the baseline's 9h03m: Pie ran ~29% faster on the easy
# 13, but 37 of these 50 are instances the baseline never solved, which tend to
# run the full 100 iterations, so the easy-set speedup may not carry. --requeue
# plus the harness's --resume (run_pie_backend.sh always passes it) means a
# preemption or a walltime kill resumes from the completed instances instead of
# starting over — which is why OUTPUT below is a FIXED path with no job id in
# it. Do not add ${SLURM_JOB_ID} to OUTPUT or resume breaks.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_neutral_50
export PIE_DEBUG_LOG=logs/pie_debug_neutral_50_${SLURM_JOB_ID}.jsonl

BASELINE_REPORT=predictions/baseline_t0_full_50.report.json

# Expand the baseline's 50 instance ids into repeated --instance-id flags.
mapfile -t INSTANCE_ARGS < <(.venv/bin/python - "$BASELINE_REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ids = sorted(set(d["resolved_ids"]) | set(d["unresolved_ids"]))
assert len(ids) == d["total"] == 50, f"expected 50 ids, got {len(ids)}"
for i in ids:
    print("--instance-id")
    print(i)
PY
)
echo "Instance set: $(( ${#INSTANCE_ARGS[@]} / 2 )) ids from $BASELINE_REPORT"

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-cuda-native-apc+qwen3-coder-30b-t0-neutral50 \
OUTPUT=predictions/neutral_50_pie.jsonl \
REQUEST_TIMEOUT_S=900 \
  bash run_pie_backend.sh \
    "${INSTANCE_ARGS[@]}" \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

PRED=predictions/neutral_50_pie.jsonl
LOG=logs/neutral_50_${SLURM_JOB_ID}.out
echo ""
echo "=== session KV-reuse (hit-rate) across the run ==="
grep -oE "modes=\{[^}]*\}" "$LOG" | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" "$LOG" || echo 0
echo ""
echo "=== non-empty patch summary ==="
.venv/bin/python - "$PRED" <<'PY'
import json,sys
n=ne=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    n+=1; ne+= 1 if p else 0
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
print(f"\nTOTAL {n}  non-empty {ne}")
PY
