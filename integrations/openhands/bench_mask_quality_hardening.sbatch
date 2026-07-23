#!/usr/bin/env bash
#SBATCH --job-name=mask-quality-hard
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_quality_hard_%j.out

# HARDENING run for the mask-condensation quality claim. Addresses the two big
# caveats from job 19068808: (1) FUTURE-GENERATION queries (predict tokens AFTER
# the kept window, not re-predict inside it); (2) sweep history length P up
# toward the model's max range to find where MASK's ever-growing original
# positions break vs REBUILD's bounded compact positions. Perplexity on gold
# continuation tokens, three ways (full / mask / rebuild).

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
mkdir -p logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
INF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/mask-condense-bench
WASM=$INF/target/wasm32-wasip2/release/mask_condense_bench.wasm
MANIFEST=$INF/Pie.toml
PIE_LOG=logs/pie_serve_mask_hard_${SLURM_JOB_ID}.log

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

echo "[run] hardening sweep..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  .venv/bin/python - <<'PY'
import asyncio, glob, json, math, os
from tokenizers import Tokenizer
from pie_client import Event, PieClient

# Build a LONG document (need > max P + num_queries tokens). Concatenate Python
# across the repo; repeat if still short.
files = sorted(glob.glob("/nfs/roberts/project/pi_ql324/ly337/pie/**/*.py", recursive=True))
text = ""
for f in files:
    try: text += open(f, errors="ignore").read() + "\n\n"
    except Exception: pass
    if len(text) > 400000: break

tj = sorted(glob.glob(os.environ["HF_HOME"]
            + "/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/tokenizer.json"))[0]
tok = Tokenizer.from_file(tj)
ids = tok.encode(text, add_special_tokens=False).ids
while len(ids) < 60000:
    ids = ids + ids
print(f"corpus tokens: {len(ids)}")

SINK, KEEP, NQ = 64, 256, 48
# Sweep history length up toward max_model_len (65536); >32k probes the
# beyond-base-trained regime. 3 live contexts, so cap to keep KV in budget.
P_SWEEP = [4000, 12000, 24000, 40000, 56000]

def stats(lp):
    xs = [x for x in lp if x == x]
    m = sum(xs) / max(1, len(xs))
    return m, math.exp(-m)

async def run_one(client, P):
    cfg = {"token_ids": ids[:P + NQ + 2], "sink_size": SINK, "keep_recent": KEEP,
           "num_queries": NQ, "history_len": P}
    proc = await client.launch_process("mask-condense-bench@0.1.0", input=cfg)
    ret = err = None
    while True:
        ev, val = await proc.recv()
        if ev == Event.Return:
            ret = val.decode() if isinstance(val, (bytes, bytearray)) else str(val); break
        elif ev == Event.Error:
            err = val; break
    return ret, err

async def main():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed\n")
        print(f"{'P':>7} {'full_ppl':>10} {'mask_ppl':>10} {'rebuild_ppl':>12} {'mask/rebuild':>13}")
        for P in P_SWEEP:
            if P + NQ + 2 > len(ids):
                print(f"{P:>7}  (skip: corpus too short)"); continue
            ret, err = await run_one(c, P)
            if err:
                print(f"{P:>7}  ERROR: {err}"); continue
            r = json.loads(ret)
            _, fppl = stats(r["full_lp"]); _, mppl = stats(r["mask_lp"]); _, rppl = stats(r["rebuild_lp"])
            ratio = mppl / rppl if rppl else float("nan")
            flag = "  <-- mask worse" if ratio > 1.05 else ("  <-- mask better" if ratio < 0.95 else "")
            print(f"{P:>7} {fppl:>10.4f} {mppl:>10.4f} {rppl:>12.4f} {ratio:>13.4f}{flag}")
        print("\nINTERPRETATION: mask/rebuild < 1 => mask better (original positions win); "
              "> 1 => mask worse (positions exceeded useful range, rebuild's compact "
              "re-positioning wins). The crossover P is mask-condensation's operating limit.")

asyncio.run(main())
PY

echo "=== done ==="
