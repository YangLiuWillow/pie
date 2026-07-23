#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-timing2
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_timing2_%j.out

# Follow-up to job 19245724. That run split the call INSIDE the inferlet and
# closed to 0.1 ms unattributed: 92.9% decode, 6.9% prefill, and the entire APC
# apparatus (render 6.8 ms, hash 0.3, open 0.6, save 0.6, fork 0.0) under 9 ms.
# So the inferlet is not where time goes. But the inferlet only accounted for
# 130.6 s of that instance's 456.4 s wallclock — 71% was outside it.
#
# This run closes that gap. llm.py now records, per call:
#   host_ms          — whole round trip as the host sees it
#   tokens_generated — real token count (19245724 had to infer it from chars)
#   prompt_tokens
# host_ms - timings.total_ms is exactly the transport + process-launch cost the
# inferlet cannot see, and instance_wallclock - sum(host_ms) is agent/tool time.
#
# Reference points: litellm+vLLM measured 3.44 s/call (mean latency_sec over its
# own 492 completion logs, 221 completion tokens, 64.2 tok/s); Pie's inferlet
# measured 4.35 s/call on this same instance.
#
# Same instance, same config as 19245724 so the two are directly comparable.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18102
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_timing2
export PIE_DEBUG_LOG=logs/pie_debug_cuda_native_timing2_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+apc+timings2 \
OUTPUT=predictions/cuda_native_timing2_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_timing2 \
REQUEST_TIMEOUT_S=900 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== session modes seen (hit-rate) ==="
grep -oE "modes=\{[^}]*\}" logs/cuda_native_timing2_${SLURM_JOB_ID}.out | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" logs/cuda_native_timing2_${SLURM_JOB_ID}.out || echo 0

echo ""
echo "=== WHERE THE TIME GOES ==="
.venv/bin/python - \
  "logs/pie_debug_cuda_native_timing2_${SLURM_JOB_ID}.jsonl" \
  "logs/cuda_native_timing2_${SLURM_JOB_ID}.out" <<'PY'
import json, re, sys, statistics as st

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        r = json.loads(line)
        if r.get("timings"):
            rows.append(r)

if not rows:
    print("no timings recorded — is the rebuilt wasm installed?")
    raise SystemExit

n = len(rows)
def m(f): return st.mean([f(r) for r in rows])

total   = m(lambda r: r["timings"]["total_ms"])
decode  = m(lambda r: r["timings"]["decode_ms"])
prefill = m(lambda r: r["timings"]["prefill_ms"])
apc     = m(lambda r: sum(r["timings"][k] for k in
                          ("setup_ms","render_ms","hash_ms","open_ms","save_ms","fork_ms")))

have_host = all(r.get("host_ms") is not None for r in rows)
host = m(lambda r: r["host_ms"]) if have_host else float("nan")

toks = [r.get("tokens_generated") for r in rows if r.get("tokens_generated") is not None]
mean_tok = st.mean(toks) if toks else float("nan")

# Instance wallclock from the driver's own summary line.
wall = float("nan"); iters = float("nan")
for line in open(sys.argv[2]):
    mm = re.search(r"ok \(([\d.]+)s, (\d+) iters", line)
    if mm:
        wall, iters = float(mm.group(1)), int(mm.group(2))

print(f"calls={n}  iters={iters}  instance wallclock={wall:.1f}s\n")
print(f"{'layer':<34} {'mean ms':>10} {'x calls (s)':>12}")
print(f"{'decode (GPU)':<34} {decode:10.1f} {decode*n/1000:12.1f}")
print(f"{'prefill (GPU)':<34} {prefill:10.1f} {prefill*n/1000:12.1f}")
print(f"{'APC bookkeeping (all phases)':<34} {apc:10.1f} {apc*n/1000:12.1f}")
print(f"{'= inferlet total':<34} {total:10.1f} {total*n/1000:12.1f}")
if have_host:
    print(f"{'transport (host_ms - total_ms)':<34} {host-total:10.1f} {(host-total)*n/1000:12.1f}")
    print(f"{'= host round trip':<34} {host:10.1f} {host*n/1000:12.1f}")
    print(f"\n{'agent/tools (wallclock - host)':<34} {'':>10} {wall - host*n/1000:12.1f}")
    print(f"\nshare of instance wallclock:")
    print(f"  Pie round trip : {100*host*n/1000/wall:5.1f}%")
    print(f"  agent + tools  : {100*(wall - host*n/1000)/wall:5.1f}%")
else:
    print("\nhost_ms MISSING — llm.py instrumentation did not take effect.")

print(f"\ntokens generated/call: mean={mean_tok:.0f}")
if toks:
    print(f"decode rate: {mean_tok/(decode/1000):.1f} tok/s")
    print(f"per-token latency vs litellm (221 tok in 3.44 s = 64.2 tok/s):")
    print(f"  Pie {mean_tok/(decode/1000):.1f} tok/s   litellm 64.2 tok/s")
    print(f"  length-normalized Pie call at litellm's token count: "
          f"{221/(mean_tok/(decode/1000)):.2f} s vs litellm 3.44 s")
PY

echo ""
echo "=== patch summary (correctness must not regress) ==="
.venv/bin/python - "predictions/cuda_native_timing2_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
PY
