"""Decode-throughput vs context-length diagnostic.

Question: does Pie's decode tok/s collapse as the retained/attended context
grows, while vLLM's holds? That would explain the OpenHands 3.4x/iter loss
(long agentic contexts) despite the fan-out N=1 tie (short context), and would
mean mask-condense (shrinking the decode context) is the fix.

Method: a single call prefills an L-token prompt then decodes M tokens. Decode
attends over the full L-token context whether it's session-retained or freshly
prefilled, so one call measures decode-over-L for both engines. We run M_lo and
M_hi at each L and take the slope
    decode_s_per_tok = (t_hi - t_lo) / (M_hi - M_lo)
which cancels the fixed prefill(L) cost -> pure decode throughput. Both engines
are forced to emit EXACTLY M tokens (decode-bench inferlet has no stop; vLLM
uses ignore_eos + min_tokens).

Env:
  BENCH_ARM     pie | vllm
  DT_L          comma list of context lengths, e.g. "2000,5000,20000,50000"
  DT_M          "lo,hi" decode lengths, e.g. "64,512"
  DT_REPEATS    repeats per cell (default 3)
  BENCH_OUT     results .jsonl
  pie:  PIE_URI/PIE_USER/PIE_DB_WASM/PIE_DB_MANIFEST/PIE_DB_INFERLET
  vllm: OPENAI_API_BASE/OPENAI_API_KEY/VLLM_MODEL
"""
import asyncio
import hashlib
import json
import os
import sys
import time

INFERLET = os.environ.get("PIE_DB_INFERLET", "decode-bench@0.1.0")


def make_prompt(target_tokens):
    """Deterministic hex-dense filler ~1.11 chars/token, so chars ~= 1.11*T.
    Identical string handed to both engines -> identical L."""
    seed = hashlib.sha256(b"decode-tps").hexdigest()
    target_chars = int(target_tokens * 1.11)
    lines, n, i = [], 0, 0
    while n < target_chars:
        line = f"# ctx-{seed[:16]}-{i:06d}-{seed[16:48]}"
        lines.append(line)
        n += len(line) + 1
        i += 1
    return "Continue this log.\n" + "\n".join(lines)


# ---------------- Pie ----------------
class PieRunner:
    def __init__(self):
        from pie_client import Event, PieClient
        self._Event, self._PieClient = Event, PieClient
        self.uri = os.environ.get("PIE_URI", "ws://127.0.0.1:18099")
        self.user = os.environ.get("PIE_USER", "local-dev")
        self.client = None

    async def __aenter__(self):
        self.client = await self._PieClient(self.uri).__aenter__()
        await self.client.authenticate(self.user)
        wasm, manifest = os.environ.get("PIE_DB_WASM"), os.environ.get("PIE_DB_MANIFEST")
        if wasm and manifest:
            await self.client.install_program(wasm, manifest, force_overwrite=True)
        return self

    async def __aexit__(self, *a):
        await self.client.__aexit__(*a)

    async def call(self, prompt, M):
        payload = {"prompt": prompt, "decode_tokens": M}
        t0 = time.time()
        proc = await self.client.launch_process(INFERLET, input=payload)
        result = None
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=1200)
            if event == self._Event.Return:
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode()
                result = json.loads(value) if isinstance(value, str) else value
                break
            if event == self._Event.Error:
                raise RuntimeError(f"pie error: {value!r}")
        dt = time.time() - t0
        return dt, result.get("prefill_tokens"), result.get("decode_tokens_generated")


# ---------------- vLLM ----------------
class VllmRunner:
    def __init__(self):
        import openai
        self.base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self.key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        self.client = openai.AsyncOpenAI(base_url=self.base, api_key=self.key, max_retries=0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def call(self, prompt, M):
        t0 = time.time()
        r = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=M, temperature=0.9, top_p=0.95,
            extra_body={"ignore_eos": True, "min_tokens": M},
        )
        dt = time.time() - t0
        u = r.usage
        return dt, u.prompt_tokens, u.completion_tokens


async def run():
    arm = os.environ.get("BENCH_ARM", "pie")
    Ls = [int(x) for x in os.environ.get("DT_L", "2000,5000,20000,50000").split(",")]
    M_lo, M_hi = [int(x) for x in os.environ.get("DT_M", "64,512").split(",")]
    reps = int(os.environ.get("DT_REPEATS", "3"))
    out_path = os.environ.get("BENCH_OUT", f"logs/decode_tps_{arm}.jsonl")
    runner_cm = PieRunner() if arm == "pie" else VllmRunner()
    print(f"arm={arm} L={Ls} M=({M_lo},{M_hi}) reps={reps}", flush=True)

    async with runner_cm as runner:
        # warmup (load weights)
        try:
            await runner.call(make_prompt(2000), M_lo)
        except Exception as e:
            print(f"warmup error: {e}", flush=True)

        for L in Ls:
            prompt = make_prompt(L)
            cell = {"arm": arm, "L_target": L}
            for M in (M_lo, M_hi):
                times, pfs, gens = [], [], []
                for _ in range(reps):
                    try:
                        dt, pf, gen = await runner.call(prompt, M)
                        times.append(dt); pfs.append(pf); gens.append(gen)
                    except Exception as e:
                        print(f"  L={L} M={M} ERROR {e}", flush=True)
                if times:
                    times.sort()
                    med = times[len(times) // 2]
                    cell[f"t_M{M}"] = med
                    cell[f"prefill_M{M}"] = pfs[0]
                    cell[f"gen_M{M}"] = gens[0]
            # slope: isolate decode
            if f"t_M{M_hi}" in cell and f"t_M{M_lo}" in cell:
                dsec = cell[f"t_M{M_hi}"] - cell[f"t_M{M_lo}"]
                dtok = M_hi - M_lo
                s_per_tok = dsec / dtok if dtok else float("nan")
                cell["decode_s_per_tok"] = s_per_tok
                cell["decode_tps"] = (1.0 / s_per_tok) if s_per_tok > 0 else float("nan")
            with open(out_path, "a") as f:
                f.write(json.dumps(cell) + "\n")
            print(f"  L~{cell.get('prefill_M64', L):6} "
                  f"t64={cell.get('t_M64', float('nan')):6.2f}s "
                  f"t512={cell.get('t_M512', float('nan')):6.2f}s "
                  f"decode_tps={cell.get('decode_tps', float('nan')):7.1f} tok/s", flush=True)
    print(f"DONE arm={arm} -> {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
