#!/usr/bin/env bash
#SBATCH --job-name=ab-pie
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:10:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/ab_pie_%j.out

# A/B harness — Pie arm (B2 fork batch). Boots pie serve (30B), then runs
# ab_bench.py sweeping N and concurrency across real OpenEvolve examples.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18099
export PIE_URI=ws://127.0.0.1:18099
export PIE_USER=local-dev
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe_aligned.toml

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation
export PIE_OE_WASM="$INFERLET_DIR/target/wasm32-wasip2/release/openevolve_generation.wasm"
export PIE_OE_MANIFEST="$INFERLET_DIR/Pie.toml"

export BENCH_ARM=pie_batch
export BENCH_EXAMPLES=${BENCH_EXAMPLES:-function_minimization,circle_packing,signal_processing}
export BENCH_N=${BENCH_N:-1,2,4,8}
export BENCH_CONC=${BENCH_CONC:-1,2,4,8}
export BENCH_REPEATS=${BENCH_REPEATS:-3}
export BENCH_MAXTOK=${BENCH_MAXTOK:-256}
export BENCH_OUT=logs/ab_bench_pie_${SLURM_JOB_ID}.jsonl
mkdir -p logs

echo "[1/3] Building inferlet..."; ( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )
echo "[2/3] Booting pie serve..."
PIE_LOG="logs/pie_serve_ab_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!; trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 200); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 "$PIE_PID" 2>/dev/null || { echo "ERROR: pie died"; tail -40 "$PIE_LOG"; exit 1; }
    sleep 2
done
echo "[3/3] Running A/B (pie)..."
PYTHONPATH="" .venv/bin/python ab_bench.py
echo "=== report (pie only) ==="
PYTHONPATH="" .venv/bin/python ab_bench_report.py "$BENCH_OUT"
