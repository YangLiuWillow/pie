#!/usr/bin/env bash
#SBATCH --job-name=mask-debug-fix
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_debug_%j.out

# Verify the 2026-07-18 tool-call grammar fix (schema-constrained arguments
# + whitespace tolerance in qwen3.rs build_tool_call_grammar): same probe
# suite as bench_mask_debug.sbatch, but the server binary comes from the
# separate grammar-fix target dir so the live target/release/pie (used by
# still-running job restarts) is untouched.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

PIE=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
SCRATCH=/nfs/roberts/scratch/pi_ql324/ly337
PIE_LOG=logs/mask_debug_pie_serve_${SLURM_JOB_ID}.log
export HF_HOME=$SCRATCH/hf_cache
export PIE_MASK_DEBUG=$SCRATCH/mask_debug_${SLURM_JOB_ID}.jsonl

rm -f "$PIE_MASK_DEBUG" "$PIE_MASK_DEBUG.err"

PYTHONPATH=$SCRATCH/dbgdrv \
  $PIE serve --config tests/fixtures/pie_cuda_vllm_config.toml --port 18085 --no-auth \
  > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'kill $PIE_PID 2>/dev/null || true' EXIT

ready=0
for i in $(seq 1 300); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        ready=1; break
    fi
    if ! kill -0 $PIE_PID 2>/dev/null; then
        echo "pie serve died during startup:"; tail -40 "$PIE_LOG"; exit 1
    fi
    sleep 2
done
[ "$ready" -eq 1 ] || { echo "server not ready in time"; tail -40 "$PIE_LOG"; exit 1; }
echo "pie serve ready"

PYTHONPATH="" .venv/bin/python $SCRATCH/probe_suite.py ws://127.0.0.1:18085

echo ""
echo "=== mask debug dump: $PIE_MASK_DEBUG ==="
wc -l "$PIE_MASK_DEBUG" 2>/dev/null || echo "(no mask debug lines written!)"
[ -f "$PIE_MASK_DEBUG.err" ] && { echo "instrumentation errors:"; head "$PIE_MASK_DEBUG.err"; }
exit 0
