#!/usr/bin/env bash
#SBATCH --job-name=slice3-deleg
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/slice3_deleg_%j.out

# Slice 3 — agent-level critic-continue fork delegation (real generation).
# Boots pie serve, then runs bench_slice3_delegation.py: for each instance a
# REAL OpenHands agent solves the SWE-bench issue (building a large KV
# snapshot), then a fork-stamped child (PieLLM.model_copy, the Slice-2 glue)
# runs a real critic-continue generation from the parent's full context while a
# no-fork control runs the same prompt cold. Reports, per instance:
#   - behaviour-neutrality: sha256(fork output) == sha256(cold output) at temp 0
#   - prefill saved: forked suffix-only vs cold full re-prefill
#   - wall saved: forked critic call vs cold critic call
# Uses the SAME PIE_BIN (target-grammarfix) + coder-session wasm (both
# unchanged) and the aligned vllm config as the acceleration eval.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
# Deterministic workspace path: the ws path leaks into prompts via tool output,
# so both fork/cold arms only see identical prompts if the path is identical.
export SWEB_WS_ROOT=/tmp/sweb_slice3_${SLURM_JOB_ID}
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe_aligned.toml

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session
mkdir -p logs "$SWEB_WS_ROOT"

echo "[1/3] Building coder-session inferlet (wasm32-wasip2)..."
( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )

echo "[2/3] Booting pie serve on port $PIE_PORT..."
PIE_LOG="logs/pie_serve_slice3_deleg_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'echo "Stopping pie serve (PID $PIE_PID)"; kill "$PIE_PID" 2>/dev/null || true; rm -rf "$SWEB_WS_ROOT" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then
        echo "  pie serve is up ($((i * 2))s)"
        break
    fi
    if ! kill -0 "$PIE_PID" 2>/dev/null; then
        echo "ERROR: pie serve died during startup. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
    sleep 2
    if [ "$i" -eq 120 ]; then
        echo "ERROR: pie serve not ready within timeout. Check $PIE_LOG"; tail -40 "$PIE_LOG"; exit 1
    fi
done

echo "[3/3] Running Slice-3 critic-continue fork delegation..."
PYTHONPATH="" .venv/bin/python bench_slice3_delegation.py \
    --max-iterations 100 \
    --out logs/slice3_deleg_${SLURM_JOB_ID}.json
