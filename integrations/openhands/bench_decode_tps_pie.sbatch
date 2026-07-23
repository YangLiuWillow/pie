#!/usr/bin/env bash
#SBATCH --job-name=dtps-pie
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/dtps_pie_%j.out

# Decode-throughput vs context-length — Pie arm. Does Pie's decode tok/s
# collapse as attended context grows (the OpenHands-slowdown hypothesis)?
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export PIE_URI=ws://127.0.0.1:18099
export PIE_USER=local-dev
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe_ctx64k.toml

DB=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/decode-bench
export PIE_DB_WASM="$DB/target/wasm32-wasip2/release/decode_bench.wasm"
export PIE_DB_MANIFEST="$DB/Pie.toml"

export BENCH_ARM=pie
export DT_L=${DT_L:-2000,5000,20000,50000}
export DT_M=${DT_M:-64,512}
export DT_REPEATS=${DT_REPEATS:-3}
export BENCH_OUT=logs/decode_tps_pie_${SLURM_JOB_ID}.jsonl
mkdir -p logs

echo "[1/3] Building decode-bench inferlet..."; ( cd "$DB" && cargo build --release --target wasm32-wasip2 )
ls -la "$PIE_DB_WASM"
echo "[2/3] Booting pie serve (ctx 64k)..."
PIE_LOG="logs/pie_serve_dtps_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!; trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 "$PIE_PID" 2>/dev/null || { echo "ERROR: pie died"; tail -50 "$PIE_LOG"; exit 1; }
    sleep 2
done
echo "[3/3] Running decode-tps (pie)..."
PYTHONPATH="" .venv/bin/python decode_tps_bench.py
