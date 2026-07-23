#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-timing
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_timing_%j.out

# ONE instance, instrumented — attribute the per-call non-decode time.
#
# A/B job 19212030 measured, on the 4 instances that completed: Pie 13.5 s per
# LLM call vs the litellm/vLLM baseline's 3.9 s. That gap is NOT decode
# (63 vs 65 tok/s, measured by jobs 19078674/19078675), NOT extra output
# (~250 vs 221 generated tokens/call), NOT prefill (APC cut it to ~5%), and NOT
# the phase-2 fallback (debug_phase2_fired was 0 across all 169 calls). So
# ~9.5 s/call is unattributed.
#
# The coder-session inferlet now emits a per-phase `timings` block on every
# call (setup / render / hash / open / prefill / save / fork / decode / total,
# ms), which llm.py writes into $PIE_DEBUG_LOG. django-13028 is the proven
# instance (~8 min, ~39 calls) — enough samples to attribute the residual
# without spending 13-instance GPU time.
#
# Prime suspects, in the order the numbers would implicate them:
#   render_ms — the whole history is re-tokenized every call (one tokenizer
#               round-trip per message), because each call is a fresh wasm
#               instance that cannot carry the previous render forward.
#   fork_ms   — ctx.fork() is taken on EVERY call for a phase-2 fallback that
#               never fired; forking copies the cued prompt's working pages.
#   open_ms   — up to 8 Context::open round-trips per call.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18101
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_timing
export PIE_DEBUG_LOG=logs/pie_debug_cuda_native_timing_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+apc+timings \
OUTPUT=predictions/cuda_native_timing_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_timing \
REQUEST_TIMEOUT_S=900 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== session modes seen (hit-rate) ==="
grep -oE "modes=\{[^}]*\}" logs/cuda_native_timing_${SLURM_JOB_ID}.out | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" logs/cuda_native_timing_${SLURM_JOB_ID}.out || echo 0

echo ""
echo "=== PER-PHASE TIMING BREAKDOWN (the point of this run) ==="
.venv/bin/python - "logs/pie_debug_cuda_native_timing_${SLURM_JOB_ID}.jsonl" <<'PY'
import json, sys

PHASES = ["setup_ms", "render_ms", "hash_ms", "open_ms",
          "prefill_ms", "save_ms", "fork_ms", "decode_ms"]

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    t = (json.loads(line) or {}).get("timings")
    if t:
        rows.append(t)

if not rows:
    print("no timings recorded — is the rebuilt wasm installed?")
    raise SystemExit

n = len(rows)
def col(k): return sorted(r.get(k) or 0.0 for r in rows)
def mean(k): return sum(r.get(k) or 0.0 for r in rows) / n

total = mean("total_ms")
print(f"calls={n}   mean total={total:8.1f} ms\n")
print(f"{'phase':<12} {'mean ms':>10} {'median':>10} {'max':>10} {'% total':>8}")
acc = 0.0
for k in PHASES:
    c, m = col(k), mean(k)
    acc += m
    print(f"{k:<12} {m:10.1f} {c[n//2]:10.1f} {c[-1]:10.1f} {100*m/total:7.1f}%")
print(f"{'-'*54}")
print(f"{'accounted':<12} {acc:10.1f} {'':10} {'':10} {100*acc/total:7.1f}%")
print(f"{'unattributed':<12} {total-acc:10.1f} {'':10} {'':10} {100*(total-acc)/total:7.1f}%")
print(f"\nmean Context::open attempts/call: "
      f"{sum(r.get('open_attempts') or 0 for r in rows)/n:.2f}")
print("\nBaseline for comparison: litellm+vLLM ran 3.9 s/call on this model;")
print("A/B job 19212030 measured Pie at 13.5 s/call.")
PY

echo ""
echo "=== patch summary (correctness must not regress) ==="
.venv/bin/python - "predictions/cuda_native_timing_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
PY
