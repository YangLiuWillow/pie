#!/usr/bin/env bash
#SBATCH --job-name=mask-condense-bench
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_condense_bench_%j.out

# B2-mask Stage-1: mask-condense (0 re-prefill) vs rebuild-condense (re-prefill).
# Sweeps the kept-window size to show the re-prefill vLLM+APC must pay to drop
# the middle grows with kept size, while Pie's mask stays 0.

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
PIE_LOG=logs/pie_serve_mask_condense_${SLURM_JOB_ID}.log

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

echo "[run] sweep..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 \
  WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  $VENV/bin/python - <<'PY'
import asyncio, os
from pie_client import Event, PieClient

CONFIGS = [
    {"history_tokens": 8000, "sink_size": 64, "window_size": 512,  "decode_steps": 128},
    {"history_tokens": 8000, "sink_size": 64, "window_size": 2048, "decode_steps": 128},
    {"history_tokens": 8000, "sink_size": 64, "window_size": 4096, "decode_steps": 128},
]

async def run_one(client, cfg):
    proc = await client.launch_process("mask-condense-bench@0.1.0", input=cfg)
    lines = []
    while True:
        ev, val = await proc.recv()
        if ev == Event.Stdout:
            s = (val.decode() if isinstance(val, (bytes, bytearray)) else str(val))
            lines.append(s)
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
        # warm-up (first launch pays JIT/alloc) then real runs
        await run_one(c, {"history_tokens": 1000, "sink_size": 32, "window_size": 128, "decode_steps": 8})
        for cfg in CONFIGS:
            print(f">>> config: {cfg}")
            out = await run_one(c, cfg)
            for line in out.splitlines():
                if "=" in line or line.startswith("==="):
                    print("   ", line.strip())
            print()

asyncio.run(main())
PY

echo "=== done ==="
