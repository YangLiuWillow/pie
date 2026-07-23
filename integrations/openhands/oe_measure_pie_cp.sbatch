#!/usr/bin/env bash
#SBATCH --job-name=oe-meas-pie-cp
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/oe_meas_pie_cp_%j.out

# MEASUREMENT — Pie/cuda_native + content-addressed APC, function_minimization.
# Boots pie serve once, then runs the evolution for each seed. Time + accuracy
# are parsed from the per-run "OE_MEASURE ..." line + per-iteration log lines by
# oe_measure_report.py. Head-to-head partner = oe_measure_vllm_fm.sbatch.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18096
export PIE_URI=ws://127.0.0.1:18096
export PIE_USER=local-dev
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml

INFERLET_DIR=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation
export PIE_OE_WASM="$INFERLET_DIR/target/wasm32-wasip2/release/openevolve_generation.wasm"
export PIE_OE_MANIFEST="$INFERLET_DIR/Pie.toml"
export PIE_OE_BACKEND=pie
export PIE_OE_SKIP_ANALYSIS=0

export OE_EXAMPLE=circle_packing
export OE_CONFIG=config_phase_1.yaml
export OE_ITERS=8
export OE_MAX_TOKENS=8192
export OE_TIMEOUT=900
SEEDS="0 1"
mkdir -p logs

( cd "$INFERLET_DIR" && cargo build --release --target wasm32-wasip2 )

PIE_LOG="logs/pie_serve_meas_pie_cp_${SLURM_JOB_ID}.log"
"$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
trap 'kill "$PIE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "pie up (${i}x2s)"; break; }
    kill -0 "$PIE_PID" 2>/dev/null || { echo "pie died"; tail -40 "$PIE_LOG"; exit 1; }
    sleep 2
done

for SEED in $SEEDS; do
    echo "########## SEED $SEED ##########"
    OE_SEED=$SEED PYTHONPATH="" .venv/bin/python run_pie_evolution.py || echo "seed $SEED FAILED"
done

echo "=== OE_MEASURE lines (this job) ==="
grep -h "^OE_MEASURE " "logs/oe_meas_pie_cp_${SLURM_JOB_ID}.out" || true
