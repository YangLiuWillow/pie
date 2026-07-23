#!/usr/bin/env bash
#SBATCH --job-name=mask-traj-replay
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_traj_replay_%j.out

# FIXED-TRAJECTORY REPLAY — the confound-free mask-vs-rebuild quality number.
# PHASE 1: capture ONE real SWE-bench trajectory (openhands-agent, NO condense).
# PHASE 2: tokenize the fixed transcript, replay through mask-condense-bench
# `traj_replay` at a LOW context_limit → full/mask/rebuild perplexity of the REAL
# continuation at each condensation. Both condensers see identical tokens, so the
# Stage-2 A/B (job 19059956) trajectory-nondeterminism confound is gone.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_traj_replay_ws_${SLURM_JOB_ID}
mkdir -p "$SWEB_WS_ROOT" logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
VENV=.venv
AINF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-agent
AWASM=$AINF/target/wasm32-wasip2/release/openhands_agent.wasm
AMANIFEST=$AINF/Pie.toml
BINF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/mask-condense-bench
BWASM=$BINF/target/wasm32-wasip2/release/mask_condense_bench.wasm
BMANIFEST=$BINF/Pie.toml
PIE_LOG=logs/pie_serve_traj_replay_${SLURM_JOB_ID}.log

# Capture on a real, multi-step instance; replay squeezes it to force condense.
# django-14373 is well-characterized (~44 iters) → a long linear trajectory that
# reaches tens of thousands of tokens without a trivial early finish.
# CAPTURE_CONTEXT_LIMIT stays under the ~65k model max so FULL/MASK arms (which
# keep the whole history resident) can prefill every probe point.
export INSTANCE_ID=${INSTANCE_ID:-django__django-14373}
export CAPTURE_CONTEXT_LIMIT=${CAPTURE_CONTEXT_LIMIT:-60000}
export MAX_STEPS=${MAX_STEPS:-50}
export REPLAY_CONTEXT_LIMIT=${REPLAY_CONTEXT_LIMIT:-16000}
export KEEP_RECENT_TURNS=${KEEP_RECENT_TURNS:-12}
export PROBE_TOKENS=${PROBE_TOKENS:-48}
export TRAJ_JSON=${TRAJ_JSON:-logs/traj_capture_${INSTANCE_ID//\//_}.json}

echo "[build] openhands-agent + mask-condense-bench wasm..."
( cd "$AINF" && cargo build --release --target wasm32-wasip2 )
( cd "$BINF" && cargo build --release --target wasm32-wasip2 )

PIE_PID=""
cleanup() { [ -n "$PIE_PID" ] && { kill "$PIE_PID" 2>/dev/null || true; wait "$PIE_PID" 2>/dev/null || true; }; rm -rf "$SWEB_WS_ROOT" 2>/dev/null || true; }
trap cleanup EXIT

echo "[boot] pie serve..."
PYTHONPATH="" HF_HOME=$HF_HOME \
  $PIE_BIN serve --config "$CFG" --port $PIE_PORT --no-auth > "$PIE_LOG" 2>&1 &
PIE_PID=$!
for i in $(seq 1 1800); do
    grep -q "pie-server serving on" "$PIE_LOG" 2>/dev/null && { echo "  up ($((i*2))s)"; break; }
    kill -0 $PIE_PID 2>/dev/null || { echo "ERROR: serve died"; tail -30 "$PIE_LOG"; exit 1; }
    sleep 2
done

echo "[install] both inferlets..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 \
  AWASM="$AWASM" AMANIFEST="$AMANIFEST" BWASM="$BWASM" BMANIFEST="$BMANIFEST" PIE_PORT="$PIE_PORT" \
  $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import PieClient
async def install():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["AWASM"], os.environ["AMANIFEST"], force_overwrite=True)
        await c.install_program(os.environ["BWASM"], os.environ["BMANIFEST"], force_overwrite=True)
        print("  installed openhands-agent + mask-condense-bench")
asyncio.run(install())
PY

echo "[run] capture + replay..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  $VENV/bin/python mask_traj_replay.py
RC=$?

echo "=== traj-replay exit code: $RC ==="
exit $RC
