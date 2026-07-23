#!/usr/bin/env bash
#SBATCH --job-name=mask-quality
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/mask_quality_%j.out

# Offline QUALITY microbench for mask- vs rebuild-condensation. Feeds REAL
# tokenized text through the inferlet's quality mode and scores next-token
# predictions three ways: FULL (gold), MASK (sink+window, ORIGINAL positions,
# Pie's free condensation), REBUILD (sink+window, COMPACT positions, stock
# condenser). The MASK-vs-REBUILD agreement isolates the positional-encoding
# effect; both-vs-FULL isolates the dropped-middle information loss. Answers the
# open question from the compounding bench: does mask's original-position keeping
# degrade quality vs rebuild's compact re-positioning?

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
mkdir -p logs

CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml
PIE_VENV=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env
INF=/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/mask-condense-bench
WASM=$INF/target/wasm32-wasip2/release/mask_condense_bench.wasm
MANIFEST=$INF/Pie.toml
PIE_LOG=logs/pie_serve_mask_quality_${SLURM_JOB_ID}.log

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

echo "[run] quality eval..."
PYTHONPATH="" OPENHANDS_SUPPRESS_BANNER=1 HF_HOME=$HF_HOME \
  WASM="$WASM" MANIFEST="$MANIFEST" PIE_PORT="$PIE_PORT" \
  .venv/bin/python - <<'PY'
import asyncio, glob, json, os
from tokenizers import Tokenizer
from pie_client import Event, PieClient

# Real long text with genuine long-range structure (a big source file). We want
# L > sink + keep + num_queries so there's a real "middle" to drop.
SRC_FILES = sorted(glob.glob("benchmarks/*.py") + glob.glob("pie_openhands/*.py"))
text = ""
for f in SRC_FILES:
    text += open(f).read() + "\n\n"
    if len(text) > 40000:
        break

# Load the model's tokenizer.json straight from the HF cache (no transformers).
tj = sorted(glob.glob(os.environ["HF_HOME"]
            + "/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/tokenizer.json"))[0]
tok = Tokenizer.from_file(tj)
ids = tok.encode(text, add_special_tokens=False).ids

L = 6000
# Small keep_recent so there's a LARGE dropped middle (real info loss to detect)
# and a large positional gap between sink and recent (the positional effect).
SINK, KEEP, NQ = 64, 256, 128
ids = ids[:L]
print(f"tokenized {len(ids)} tokens; sink={SINK} keep_recent={KEEP} num_queries={NQ}")

async def main():
    async with PieClient(f"ws://127.0.0.1:{os.environ['PIE_PORT']}") as c:
        await c.authenticate("local-dev")
        await c.install_program(os.environ["WASM"], os.environ["MANIFEST"], force_overwrite=True)
        print("  installed")
        cfg = {"token_ids": ids, "sink_size": SINK, "keep_recent": KEEP, "num_queries": NQ}
        proc = await c.launch_process("mask-condense-bench@0.1.0", input=cfg)
        ret, err = None, None
        while True:
            ev, val = await proc.recv()
            if ev == Event.Return:
                ret = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
                break
            elif ev == Event.Error:
                err = val; break
        if err:
            print("ERROR:", err); return
        import math
        r = json.loads(ret)
        full, mask, rebuild = r["full_lp"], r["mask_lp"], r["rebuild_lp"]
        def stats(lp):
            xs = [x for x in lp if x == x]  # drop NaN
            m = sum(xs) / max(1, len(xs))
            return m, math.exp(-m)  # mean logprob, perplexity
        fm, fppl = stats(full)
        mm, mppl = stats(mask)
        rm, rppl = stats(rebuild)
        # Per-query gold-logprob deltas (how much each mode loses vs full).
        def mean_delta(a): return sum((f - x) for f, x in zip(full, a) if f == f and x == x) / max(1, len(a))
        print("\n=== mask-quality result (perplexity on gold tokens) ===")
        print(f"queries={len(full)}")
        print(f"full_ppl    = {fppl:.4f}   (mean logprob {fm:.4f})   # upper bound (full context)")
        print(f"mask_ppl    = {mppl:.4f}   (mean logprob {mm:.4f})   # sink+window, ORIGINAL positions")
        print(f"rebuild_ppl = {rppl:.4f}   (mean logprob {rm:.4f})   # sink+window, COMPACT positions")
        print(f"mask_logprob_loss_vs_full    = {mean_delta(mask):.4f}   # avg nats/token lost")
        print(f"rebuild_logprob_loss_vs_full = {mean_delta(rebuild):.4f}   # avg nats/token lost")
        print(f"mask_vs_rebuild_ppl_ratio    = {mppl / rppl:.4f}   # >1 means mask worse (positional cost)")
        print("\nINTERPRETATION: if mask_ppl ~= rebuild_ppl (ratio ~1), keeping "
              "ORIGINAL positions is quality-neutral => mask is a CLEAN win (free "
              "AND quality-neutral). If mask_ppl >> rebuild_ppl, original positions "
              "cost quality. Both-vs-full shows the info loss from dropping the "
              "middle (shared by mask AND rebuild — not a mask-specific cost).")

asyncio.run(main())
PY

echo "=== done ==="
