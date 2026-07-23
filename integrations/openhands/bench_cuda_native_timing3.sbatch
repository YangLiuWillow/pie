#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-timing3
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_timing3_%j.out

# Third and final timing probe. Splits the transport.
#
# 19245724 split the call INSIDE the inferlet: 92.9% decode, APC bookkeeping
# under 9 ms, 0.1 ms unattributed. The inferlet is not where time goes.
# 19247969 added host_ms around the whole round trip: 13.6 s/call host vs
# 3.6 s/call inferlet ⇒ 10.0 s/call (69% of instance wallclock) is transport,
# while agent+tools is only 6%. It also showed Pie's engine is FASTER than the
# baseline once normalized: 85.5 tok/s vs litellm's 64.2 (2.58 s vs 3.44 s per
# equal-length call). So the integration transport, not the engine, is the gap.
#
# This run splits that 10 s into connect / auth / launch / first_event / wait /
# close (llm.py `_transport`), against the inferlet's own total_ms. The residual
# `wait_ms - first_event_ms` vs `total_ms` separates queueing/admission from
# result delivery. Pure Python change — same wasm and same pie binary as
# 19247969, same instance and config, so all three runs are comparable.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18103
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_timing3
export PIE_DEBUG_LOG=logs/pie_debug_cuda_native_timing3_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+apc+timings3 \
OUTPUT=predictions/cuda_native_timing3_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_timing3 \
REQUEST_TIMEOUT_S=900 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== session modes seen (hit-rate) ==="
grep -oE "modes=\{[^}]*\}" logs/cuda_native_timing3_${SLURM_JOB_ID}.out | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" logs/cuda_native_timing3_${SLURM_JOB_ID}.out || echo 0

echo ""
echo "=== TRANSPORT SPLIT ==="
.venv/bin/python - \
  "logs/pie_debug_cuda_native_timing3_${SLURM_JOB_ID}.jsonl" \
  "logs/cuda_native_timing3_${SLURM_JOB_ID}.out" <<'PY'
import json, re, sys, statistics as st

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("timings") and r.get("transport")]
if not rows:
    print("no transport records — llm.py instrumentation did not take effect")
    raise SystemExit

n = len(rows)
def m(path, default=0.0):
    vals = []
    for r in rows:
        d = r
        for k in path:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        vals.append(d if isinstance(d, (int, float)) else default)
    return st.mean(vals), st.median(vals), max(vals)

wall = float("nan")
for line in open(sys.argv[2]):
    mm = re.search(r"ok \(([\d.]+)s, (\d+) iters", line)
    if mm:
        wall = float(mm.group(1))

host_mean = st.mean([r.get("host_ms") or 0.0 for r in rows])
tot_mean, _, _ = m(["timings", "total_ms"])

print(f"calls={n}  instance wallclock={wall:.1f}s")
print(f"host round trip = {host_mean:8.1f} ms/call   "
      f"({host_mean*n/1000:.1f} s over the run)\n")

print(f"{'step':<26} {'mean ms':>10} {'median':>10} {'max':>10} {'x calls (s)':>12}")
STEPS = [
    ("connect_ms",     "connect (TCP+ws)"),
    ("auth_ms",        "authenticate"),
    ("launch_ms",      "launch_process"),
    ("first_event_ms", "→ first event"),
    ("wait_ms",        "→ Return (total)"),
    ("close_ms",       "close/teardown"),
]
for key, label in STEPS:
    mean, med, mx = m(["transport", key])
    print(f"{label:<26} {mean:10.1f} {med:10.1f} {mx:10.1f} {mean*n/1000:12.1f}")

print(f"\n{'inferlet total_ms':<26} {tot_mean:10.1f}")

wait_mean, _, _ = m(["transport", "wait_ms"])
fe_mean, _, _ = m(["transport", "first_event_ms"])
conn, _, _ = m(["transport", "connect_ms"])
auth, _, _ = m(["transport", "auth_ms"])
lau, _, _ = m(["transport", "launch_ms"])
clo, _, _ = m(["transport", "close_ms"])

print("\n--- attribution ---")
print(f"connect + auth + launch + close : {conn+auth+lau+clo:8.1f} ms/call")
print(f"first_event (queue + delivery)  : {fe_mean:8.1f} ms/call")
print(f"  of which inferlet ran for     : {tot_mean:8.1f} ms/call")
print(f"  ⇒ pre-entry + result delivery : {fe_mean-tot_mean:8.1f} ms/call")
print(f"stream after first event        : {wait_mean-fe_mean:8.1f} ms/call")
print(f"unaccounted (host - the above)  : "
      f"{host_mean-(conn+auth+lau+clo+wait_mean):8.1f} ms/call")
print("\nReference: 19247969 measured 10014.6 ms/call of transport, "
      "inferlet 3596.3 ms/call.")
PY

echo ""
echo "=== patch summary (correctness must not regress) ==="
.venv/bin/python - "predictions/cuda_native_timing3_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
PY
