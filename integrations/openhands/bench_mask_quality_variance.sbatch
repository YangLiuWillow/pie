#!/usr/bin/env bash
#SBATCH --job-name=mask-quality-var
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_quality_var_%j.out

# VARIANCE-TIGHTENING pass for the mask-vs-rebuild quality comparison. The
# hardening sweep (19069519) was noisy (48 queries, different text per P). Here:
# fix P, use NQ=256 gold continuation tokens, and AVERAGE over many windows (each
# a different corpus offset), so the mask/rebuild perplexity ratio comes with a
# tight mean +/- SEM instead of "roughly even". Two P values (mid + high) confirm
# it holds across context length.

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
PIE_LOG=logs/pie_serve_mask_var_${SLURM_JOB_ID}.log

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

echo "[run] variance-tightening..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  .venv/bin/python - <<'PY'
import asyncio, glob, json, math, os, statistics as st
from tokenizers import Tokenizer
from pie_client import Event, PieClient

files = sorted(glob.glob("/nfs/roberts/project/pi_ql324/ly337/pie/**/*.py", recursive=True))
text = ""
for f in files:
    try: text += open(f, errors="ignore").read() + "\n\n"
    except Exception: pass
    if len(text) > 700000: break
tj = sorted(glob.glob(os.environ["HF_HOME"]
            + "/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/tokenizer.json"))[0]
tok = Tokenizer.from_file(tj)
ids = tok.encode(text, add_special_tokens=False).ids
while len(ids) < 200000:  # need room for many high-P windows
    ids = ids + ids
print(f"corpus tokens: {len(ids)}")

SINK, KEEP, NQ = 64, 256, 256
PLAN = [(24000, 16), (48000, 10)]  # (history_len P, num_windows)

def ppl(lp):
    xs = [x for x in lp if x == x]
    return math.exp(-sum(xs) / max(1, len(xs)))

async def run_one(client, sl, P):
    cfg = {"token_ids": sl, "sink_size": SINK, "keep_recent": KEEP,
           "num_queries": NQ, "history_len": P}
    proc = await client.launch_process("mask-condense-bench@0.1.0", input=cfg)
    while True:
        ev, val = await proc.recv()
        if ev == Event.Return:
            return json.loads(val.decode() if isinstance(val, (bytes, bytearray)) else str(val)), None
        if ev == Event.Error:
            return None, val

async def main():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed\n")
        for P, W in PLAN:
            span = P + NQ + 2
            max_off = len(ids) - span
            offs = [int(k * max_off / max(1, W - 1)) for k in range(W)]
            fulls, masks, rebuilds, ratios = [], [], [], []
            for off in offs:
                sl = ids[off:off + span]
                r, err = await run_one(c, sl, P)
                if err:
                    print(f"  P={P} off={off} ERROR: {err}"); continue
                fp, mp, rp = ppl(r["full_lp"]), ppl(r["mask_lp"]), ppl(r["rebuild_lp"])
                fulls.append(fp); masks.append(mp); rebuilds.append(rp); ratios.append(mp / rp)
            n = len(ratios)
            if not n:
                print(f"P={P}: no windows succeeded"); continue
            def ms(xs):
                m = st.mean(xs); sem = (st.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
                return m, sem
            fm, fs = ms(fulls); mm, msem = ms(masks); rm, rs = ms(rebuilds); ratm, rsem = ms(ratios)
            wins = sum(1 for x in ratios if x < 1.0)
            print(f"=== P={P}  ({n} windows, NQ={NQ}) ===")
            print(f"  full_ppl    = {fm:.4f} +/- {fs:.4f}")
            print(f"  mask_ppl    = {mm:.4f} +/- {msem:.4f}")
            print(f"  rebuild_ppl = {rm:.4f} +/- {rs:.4f}")
            print(f"  mask/rebuild ratio = {ratm:.4f} +/- {rsem:.4f}   (mask better in {wins}/{n} windows)")
            print()
        print("READING: ratio +/- SEM straddling 1.0 => mask == rebuild (statistically "
              "quality-neutral, and mask is FREE). Consistently <1 => mask better; >1 => worse.")

asyncio.run(main())
PY

echo "=== done ==="
