#!/usr/bin/env bash
#SBATCH --job-name=fanout-pie
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:10:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/fanout_pie_%j.out

# HIGH-N SINGLE-CALL FAN-OUT headline — Pie B2 arm.
#
# Thesis: at a large shared prefix + high branching factor N, Pie's ONE forked
# server call (gen analysis A once, fork N leaves off P+A in-memory, batch the
# N child decodes) should beat vLLM's two-phase path, which pays N round-trips
# + N re-sends/re-tokenizations of P+A + an eviction window between phase1/2.
#
# conc=1 deliberately: isolates the fan-out mechanism (round-trips, re-send,
# decode batching) from cross-parent APC eviction. The N=1 cell doubles as the
# driver-overhead isolation (Pie-vLLM-driver vs standalone vLLM at one call).
# Working set at conc=1,N=64 ~= 12k(P)+256(A)+64*~100 ~= 19k tok << 244k wall.
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
export BENCH_EXAMPLES=${BENCH_EXAMPLES:-function_minimization}
export BENCH_N=${BENCH_N:-1,4,8,16,32,64}
export BENCH_CONC=${BENCH_CONC:-1}
export BENCH_REPEATS=${BENCH_REPEATS:-5}
# NOTE: the padding filler is hex-dense → ~1.11 chars/token (measured), so
# actual prefix tokens ≈ pad*4/1.11 ≈ pad*3.60. pad=2750 → ~9.9k filler tok →
# ~10.5k total prompt, safely under vLLM's 32768 ctx and Pie's limits.
export BENCH_PREFIX_PAD_TOKENS=${BENCH_PREFIX_PAD_TOKENS:-2750}
export BENCH_MAXTOK=${BENCH_MAXTOK:-64}
export BENCH_ANALYSIS_MAXTOK=${BENCH_ANALYSIS_MAXTOK:-256}
export BENCH_OUT=logs/fanout_pie_${SLURM_JOB_ID}.jsonl
mkdir -p logs

echo "[1/3] Building inferlet..."; ( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )
echo "[2/3] Booting pie serve..."
PIE_LOG="logs/pie_serve_fanout_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!; trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 "$PIE_PID" 2>/dev/null || { echo "ERROR: pie died"; tail -50 "$PIE_LOG"; exit 1; }
    sleep 2
done
echo "[3/3] Running fan-out sweep (pie)..."
PYTHONPATH="" .venv/bin/python ab_bench.py
echo "=== report (pie only) ==="
PYTHONPATH="" .venv/bin/python ab_bench_report.py "$BENCH_OUT"
