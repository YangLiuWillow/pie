#!/usr/bin/env bash
#SBATCH --job-name=conc-ab-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/conc_ab_smoke_%j.out

# Throughput A/B smoke test for the new --concurrency driver (swe_bench.run
# ThreadPoolExecutor fan-out). Same 6 instances, same fresh-booted Pie server
# per arm, temperature 0. We measure wall-clock per arm to get instances/hour
# and the concurrency speedup.
#
# Fresh server PER ARM is deliberate: with temperature 0 both arms run the
# SAME trajectories, so sharing one server would let the second arm hit warm
# KV/prefix state and look artificially fast — confounding the concurrency
# signal. Booting fresh per arm isolates the effect we want to measure.
set -uo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
# Leave SWEB_WS_ROOT UNSET so each instance gets a unique tempdir workspace
# (fully concurrency-safe isolation; see checked_out_repo).
unset SWEB_WS_ROOT 2>/dev/null || true

INSTANCES=(
  --instance-id django__django-14373
  --instance-id django__django-13028
  --instance-id sympy__sympy-19346
  --instance-id scikit-learn__scikit-learn-13496
  --instance-id pydata__xarray-4075
  --instance-id matplotlib__matplotlib-22719
)
N_INSTANCES=6

COMMON_ENV=(
  BACKEND=pie
  CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
  MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
  REQUEST_TIMEOUT_S=3600
)
COMMON_ARGS=( --temperature 0 --max-iterations 50 )

run_arm() {
  # $1 = arm label, $2 = concurrency, $3 = output jsonl, $4 = pie port
  local label="$1" conc="$2" out="$3" port="$4"
  echo ""
  echo "########################################################################"
  echo "# ARM $label  (concurrency=$conc)  port=$port"
  echo "########################################################################"
  local t0 t1
  t0=$(date +%s)
  env "${COMMON_ENV[@]}" \
      PIE_PORT="$port" \
      LABEL="conc-ab-$label" \
      OUTPUT="predictions/concab_${label}_${SLURM_JOB_ID}.jsonl" \
    bash run_pie_backend.sh \
      "${INSTANCES[@]}" \
      "${COMMON_ARGS[@]}" \
      --concurrency "$conc"
  t1=$(date +%s)
  echo "ARM_${label}_TOTAL_SECONDS=$((t1 - t0))"
}

# Arm B (concurrency=8) FIRST, Arm A (serial) SECOND: if anything cross-arm
# leaked despite fresh boots, it would help the serial arm — biasing the
# measured speedup DOWN (a conservative lower bound).
run_arm conc8  8 "predictions/concab_conc8_${SLURM_JOB_ID}.jsonl"  18101
run_arm serial 1 "predictions/concab_serial_${SLURM_JOB_ID}.jsonl" 18102

echo ""
echo "########################################################################"
echo "# RESULTS"
echo "########################################################################"
.venv/bin/python - "$SLURM_JOB_ID" "$N_INSTANCES" <<'PY'
import json, os, re, sys, glob

job, n = sys.argv[1], int(sys.argv[2])

# Total per-arm wall-clock is printed above as ARM_<label>_TOTAL_SECONDS; the
# server-boot portion is logged by run_pie_backend.sh as "pie serve is up (Xs)".
# We recover both from this job's stdout to report harness-only throughput.
logpath = os.path.join("logs", f"conc_ab_smoke_{job}.out")
text = open(logpath).read() if os.path.exists(logpath) else ""

def arm_seconds(label):
    m = re.search(rf"ARM_{label}_TOTAL_SECONDS=(\d+)", text)
    return int(m.group(1)) if m else None

# Boot times appear in order arm-B then arm-A; pair them up.
boots = [int(x) for x in re.findall(r"pie serve is up \((\d+)s\)", text)]

def load(label):
    path = f"predictions/concab_{label}_{job}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    return {r["instance_id"]: r for r in rows}

def summarize(label, total_s, boot_s):
    rows = load(label)
    harness_s = (total_s - boot_s) if (total_s and boot_s is not None) else total_s
    ne = sum(1 for r in rows.values() if (r.get("model_patch") or "").strip())
    sum_wall = sum(r.get("wall_clock_s", 0) for r in rows.values())
    print(f"\n[{label}] instances={len(rows)}  non-empty-patch={ne}")
    print(f"  total_wall={total_s}s  boot~={boot_s}s  harness_wall={harness_s}s")
    print(f"  sum(per-instance wall_clock_s)={sum_wall:.0f}s")
    if harness_s and len(rows):
        print(f"  throughput={len(rows)/(harness_s/3600):.2f} instances/hour")
    return harness_s, rows

tB, tA = arm_seconds("conc8"), arm_seconds("serial")
bB = boots[0] if len(boots) >= 1 else None
bA = boots[1] if len(boots) >= 2 else None

hB, rB = summarize("conc8",  tB, bB)
hA, rA = summarize("serial", tA, bA)

if hA and hB:
    print(f"\n>>> SPEEDUP (serial harness / conc8 harness) = {hA/hB:.2f}x")

# Correctness invariant: temperature 0 => identical solve set across arms.
common = set(rA) & set(rB)
same = sum(1 for i in common
           if (rA[i].get("model_patch") or "") == (rB[i].get("model_patch") or ""))
print(f"\nidentical patches across arms: {same}/{len(common)} "
      f"(expected all — temperature 0)")
PY

echo ""
echo "Done."
