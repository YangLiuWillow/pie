#!/usr/bin/env bash
#SBATCH --job-name=mask-condense-compound
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_condense_compound_%j.out

# B2-mask COMPOUNDING microbench: a long-horizon agent condenses many times.
# Sweeps RUN LENGTH (turns) to show rebuild re-prefill (what vLLM+APC must pay to
# drop the middle) SUMS over condensations while Pie's mask stays 0 — the clean,
# non-prefix "Pie beats APC" number that fork could not give. Also reports
# coherence (mask vs rebuild decoded-token agreement). Controlled: synthetic
# tokens, no agent loop / tool time / trajectory nondeterminism.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
mkdir -p logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
VENV=.venv
INF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/mask-condense-bench
WASM=$INF/target/wasm32-wasip2/release/mask_condense_bench.wasm
MANIFEST=$INF/Pie.toml
PIE_LOG=logs/pie_serve_mask_compound_${SLURM_JOB_ID}.log

echo "[build] mask-condense-bench wasm..."
( cd "$INF" && cargo build --release --target wasm32-wasip2 )

PIE_PID=""
cleanup() { [ -n "$PIE_PID" ] && { kill "$PIE_PID" 2>/dev/null || true; wait "$PIE_PID" 2>/dev/null || true; }; }
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

echo "[run] compounding sweep over run length..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 \
  WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import Event, PieClient

# Fixed condensation regime; sweep RUN LENGTH so #condensations (and thus the
# rebuild re-prefill that compounds) grows while mask stays 0.
BASE = {"turn_tokens": 300, "decode_per_turn": 8, "context_limit": 8000,
        "sink_size": 64, "keep_recent": 2000}
TURNS_SWEEP = [40, 120, 240]

async def run_one(client, cfg):
    proc = await client.launch_process("mask-condense-bench@0.1.0", input=cfg)
    lines = []
    while True:
        ev, val = await proc.recv()
        if ev == Event.Stdout:
            lines.append(val.decode() if isinstance(val, (bytes, bytearray)) else str(val))
        elif ev == Event.Return:
            break
        elif ev == Event.Error:
            print("  ERROR:", val); break
    return "".join(lines)

async def main():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed\n")
        # warm-up (first launch pays JIT/alloc)
        await run_one(c, {**BASE, "turns": 4})
        for turns in TURNS_SWEEP:
            cfg = {**BASE, "turns": turns}
            print(f">>> turns={turns}  cfg={cfg}")
            out = await run_one(c, cfg)
            for line in out.splitlines():
                if "=" in line or line.startswith("==="):
                    print("   ", line.strip())
            print()

asyncio.run(main())
PY

echo "=== done ==="
