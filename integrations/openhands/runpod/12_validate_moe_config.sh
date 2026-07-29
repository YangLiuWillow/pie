#!/usr/bin/env bash
# =============================================================================
# Measure a borrowed MoE config against vLLM's default heuristic, on THIS GPU.
#
# WHY THIS EXISTS. The `fair` tier is defined as "CUDA graphs ON + a tuned MoE
# kernel". On H200 vLLM ships the config and there is nothing to prove. On any
# board where it does not ship one (A100, H100), the tier is only honest if the
# substitute is MEASURED to beat the fallback rather than trusted because its
# filename matches the device. This is the A100 precedent from
# runpod/tuned_moe/VALIDATION.md, generalised so it can be re-run anywhere.
#
# METHOD. `benchmark_moe.py` WITHOUT `--tune` benchmarks whatever config the
# loader resolves, so the only variable between runs is VLLM_TUNED_CONFIG_FOLDER.
#
# Usage:
#   bash 12_validate_moe_config.sh <label>=<folder> [<label>=<folder> ...]
# The baseline run (no folder) is always performed first and labelled `default`.
#
# --tp-size 1 IS MANDATORY. The script defaults to tp_size=2, which computes
# E=128//2=64 and shard_intermediate_size=768, i.e. it silently benchmarks
# E=64,N=384 — the wrong shape. We serve tp=1, which needs E=128,N=768.
# (RUN_STATE.md §6; caught once already, ~20 s into a launch.)
# =============================================================================
set -euo pipefail

BENCH=${BENCH:-/workspace/benchmark_moe.py}
PIE_VENV=${PIE_VENV:-/root/venvs/pie-vllm}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
OUTDIR=${OUTDIR:-/workspace/pie/integrations/openhands/logs}
TS=$(date +%Y%m%d_%H%M%S)

[ -f "$BENCH" ] || { echo "FATAL: no $BENCH — fetch it version-matched:"; \
  echo "  curl -sL https://raw.githubusercontent.com/vllm-project/vllm/v\$(\"$PIE_VENV/bin/python\" -c 'import vllm;print(vllm.__version__)')/benchmarks/kernels/benchmark_moe.py -o $BENCH"; exit 2; }
[ $# -ge 1 ] || { echo "usage: $0 <label>=<folder> [...]"; exit 2; }

mkdir -p "$OUTDIR"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "=== MoE config validation on: $GPU"
echo "=== vLLM $("$PIE_VENV/bin/python" -c 'import vllm;print(vllm.__version__)') / Triton $("$PIE_VENV/bin/python" -c 'import triton;print(triton.__version__)')"

run_one() {
    local label=$1 folder=$2 log="$OUTDIR/moe_bench_${1}_${TS}.log"
    echo "--- $label ${folder:+(VLLM_TUNED_CONFIG_FOLDER=$folder)}" >&2
    if [ -z "$folder" ]; then
        env -u VLLM_TUNED_CONFIG_FOLDER "$PIE_VENV/bin/python" "$BENCH" \
            --model "$MODEL" --dtype auto --tp-size 1 > "$log" 2>&1
    else
        VLLM_TUNED_CONFIG_FOLDER="$folder" "$PIE_VENV/bin/python" "$BENCH" \
            --model "$MODEL" --dtype auto --tp-size 1 > "$log" 2>&1
    fi
    # Prove which config was actually resolved, rather than assuming. A silent
    # fallback to the default would otherwise show up as "the borrowed config
    # is exactly as fast as the default", which reads as a null result and is
    # really a plumbing bug.
    if grep -q "Using configuration from" "$log"; then
        grep -m1 -o "Using configuration from.*for MoE layer" "$log" | sed 's/^/    RESOLVED: /' >&2
    elif grep -q "Using default MoE config" "$log"; then
        echo "    RESOLVED: default heuristic (no config file)" >&2
    else
        echo "    RESOLVED: *** could not tell from the log — inspect $log ***" >&2
    fi
    echo "$log"
}

LOGS=(); LABELS=()
LOGS+=("$(run_one default "" | tail -1)"); LABELS+=(default)
for spec in "$@"; do
    lbl=${spec%%=*}; fld=${spec#*=}
    [ -d "$fld" ] || { echo "FATAL: not a directory: $fld"; exit 2; }
    LOGS+=("$(run_one "$lbl" "$fld" | tail -1)"); LABELS+=("$lbl")
done

echo ""
echo "=== results (kernel time, us — lower is better) ==="
"$PIE_VENV/bin/python" - "$TS" "${#LABELS[@]}" "${LABELS[@]}" "${LOGS[@]}" <<'PY'
import re, sys
ts = sys.argv[1]; n = int(sys.argv[2])
labels = sys.argv[3:3+n]; logs = sys.argv[3+n:3+2*n]

def parse(path):
    bs, out = None, {}
    for line in open(path, errors="replace"):
        m = re.search(r"Batch size:\s*(\d+)", line)
        if m: bs = int(m.group(1)); continue
        m = re.search(r"Kernel time:\s*([\d.]+)\s*us", line)
        if m and bs is not None: out[bs] = float(m.group(1)); bs = None
    return out

data = {l: parse(p) for l, p in zip(labels, logs)}
base = data[labels[0]]
if not base:
    print("no measurements parsed — inspect the logs"); raise SystemExit(1)
sizes = sorted(base)

hdr = f"{'batch':>6} " + " ".join(f"{l:>14}" for l in labels)
print(hdr); print("-" * len(hdr))
tot = {l: 0.0 for l in labels}
for b in sizes:
    row = f"{b:>6} "
    for l in labels:
        v = data[l].get(b)
        if v is None: row += f"{'-':>14} "; continue
        tot[l] += v
        row += f"{v:>8.2f}" + (f" {(v/base[b]-1)*100:+5.1f}%" if l != labels[0] else " " * 6)
    print(row)
print("-" * len(hdr))
print(f"{'SUM':>6} " + " ".join(
    f"{tot[l]:>8.2f}" + (f" {(tot[l]/tot[labels[0]]-1)*100:+5.1f}%" if l != labels[0] else " "*6)
    for l in labels))

def regime(l, lo, hi):
    ks = [b for b in sizes if lo <= b <= hi and b in data[l]]
    if not ks: return None
    return (sum(data[l][b] for b in ks) / sum(base[b] for b in ks) - 1) * 100

print()
for name, lo, hi in (("decode  (bs 1-8)", 1, 8), ("mid     (bs 16-512)", 16, 512),
                     ("prefill (bs >=1024)", 1024, 10**9)):
    parts = [f"{l} {regime(l,lo,hi):+.1f}%" for l in labels[1:] if regime(l, lo, hi) is not None]
    if parts: print(f"  {name}: " + ", ".join(parts))
print()
print("Decide on bs=1-8 (single-stream agent decode) and bs>=1024 (chunked")
print("prefill) — those are the regimes the A/B actually runs in.")
PY
