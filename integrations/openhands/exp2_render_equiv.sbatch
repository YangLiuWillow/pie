#!/usr/bin/env bash
#SBATCH --job-name=exp2-render
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/exp2_render_%j.out

# Exp 2 — render equivalence. Boots pie serve, sends one fixed (messages,tools)
# via PieLLM to read the inferlet's render len/hash, then compares to vLLM's
# chat-template render of the same. A real render divergence = layer (C).
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe_aligned.toml
INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session
mkdir -p logs

echo "[1/3] Building coder-session inferlet..."
( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )

echo "[2/3] Booting pie serve..."
PIE_LOG="logs/pie_serve_exp2_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    if grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null; then echo "  up ($((i*2))s)"; break; fi
    if ! kill -0 "$PIE_PID" 2>/dev/null; then echo "ERROR: pie serve died"; tail -40 "$PIE_LOG"; exit 1; fi
    sleep 2
done

echo "[3/3] Running render-equivalence check..."
PYTHONPATH="" .venv/bin/python exp2_render_equiv.py
