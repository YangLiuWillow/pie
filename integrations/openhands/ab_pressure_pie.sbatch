#!/usr/bin/env bash
#SBATCH --job-name=abp-pie
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:20:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/abp_pie_%j.out

# MEMORY-PRESSURE A/B — Pie B2 fork arm. Large DISTINCT prefixes (~8k tok/parent)
# swept over concurrency 8..64 → working set 64k..512k vs 244,704-token KV cache
# (overcommit 0.26x..2.1x). Tiny decode isolates prefill/reuse. Model set by
# MODEL_CFG (default 30B-A3B).
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export PIE_URI=ws://127.0.0.1:18099
export PIE_USER=local-dev
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=${MODEL_CFG:-tests/fixtures/pie_cuda_vllm_config_30b_moe_aligned.toml}

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation
export PIE_OE_WASM="$INFERLET_DIR/target/wasm32-wasip2/release/openevolve_generation.wasm"
export PIE_OE_MANIFEST="$INFERLET_DIR/Pie.toml"

export BENCH_ARM=pie_batch
export BENCH_EXAMPLES=${BENCH_EXAMPLES:-function_minimization}
export BENCH_N=${BENCH_N:-4}
export BENCH_CONC=${BENCH_CONC:-8,16,32,64}
export BENCH_REPEATS=${BENCH_REPEATS:-2}
export BENCH_PREFIX_PAD_TOKENS=${BENCH_PREFIX_PAD_TOKENS:-8000}
export BENCH_MAXTOK=${BENCH_MAXTOK:-16}
export BENCH_ANALYSIS_MAXTOK=${BENCH_ANALYSIS_MAXTOK:-64}
export BENCH_OUT=logs/abp_pie_${SLURM_JOB_ID}.jsonl
mkdir -p logs

echo "[1/3] Building inferlet..."; ( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )
echo "[2/3] Booting pie serve ($CFG)..."
PIE_LOG="logs/pie_serve_abp_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!; trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 "$PIE_PID" 2>/dev/null || { echo "ERROR: pie died"; tail -40 "$PIE_LOG"; exit 1; }
    sleep 2
done
echo "[3/3] Running pressure sweep (pie)..."
PYTHONPATH="" .venv/bin/python ab_bench.py
echo "=== report (pie) ==="
PYTHONPATH="" .venv/bin/python ab_bench_report.py "$BENCH_OUT"
