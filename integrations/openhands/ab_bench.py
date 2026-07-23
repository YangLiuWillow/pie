"""A/B microbenchmark: Pie fork (B2) vs vLLM two-phase+APC (A1) vs naive (A0).

Controlled replay — NOT the stochastic OpenEvolve loop. A fixed workload of
(parent, N-children) generation batches is replayed identically against each
system arm, so the only variable is the engine's KV-reuse mechanism.

Parents are drawn from REAL OpenEvolve examples (their initial_program.py +
system_message), giving realistic, varied prefix sizes. We sweep:
  * N            — children per parent (branching factor)
  * concurrency  — how many distinct parents' batches run at once (the pressure
                   that evicts vLLM's APC but not Pie's pinned snapshots)

Metrics per batch:
  * wallclock
  * Pie:  inferlet telemetry (l1p/l1g mode, prefill, decode, shared_prefix,
          saved, kv_avoided)
  * vLLM: usage.prompt_tokens / completion_tokens and
          prompt_tokens_details.cached_tokens  (direct APC-hit measurement;
          needs `--enable-prompt-tokens-details` on the server)

Env:
  BENCH_ARM        pie_batch | vllm_twophase | vllm_naive
  BENCH_EXAMPLES   comma list of example dir names (default a small set)
  BENCH_N          comma list, e.g. "1,2,4,8"
  BENCH_CONC       comma list, e.g. "1,2,4,8"
  BENCH_REPEATS    repeats per (N,conc) cell (default 3)
  BENCH_MAXTOK     child max tokens (default 256)
  BENCH_ANALYSIS_MAXTOK  analysis max tokens (default 256)
  BENCH_OUT        results .jsonl path
  PIE_URI / PIE_USER / PIE_OE_WASM / PIE_OE_MANIFEST  (pie arm)
  OPENAI_API_BASE / OPENAI_API_KEY / VLLM_MODEL       (vllm arms)
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid

OE_ROOT = "/nfs/roberts/project/pi_ql324/ly337/gemini/openevolve"
EX_DIR = f"{OE_ROOT}/examples"

DEFAULT_EXAMPLES = "function_minimization,circle_packing,signal_processing"
INFERLET = os.environ.get("PIE_OE_INFERLET", "openevolve-generation@0.1.0")

ANALYSIS_INSTRUCTION = (
    "First, briefly analyze the program's weaknesses and list 4 distinct, "
    "concrete improvement directions, numbered 1-4. Do not write code yet."
)
DIFF_TASK = (
    "Improve the program via SEARCH/REPLACE diffs. Use the exact format:\n"
    "<<<<<<< SEARCH\n# code to find\n=======\n# replacement\n>>>>>>> REPLACE"
)


def load_examples(names):
    """Return [{name, system, parent_block}] drawn from real examples."""
    out = []
    for name in names:
        d = f"{EX_DIR}/{name}"
        prog = None
        for fn in ("initial_program.py", "initial_program.rs", "initial_program.txt"):
            p = f"{d}/{fn}"
            if os.path.exists(p):
                with open(p) as f:
                    prog = f.read()
                break
        if prog is None:
            print(f"  (skip {name}: no initial_program)", flush=True)
            continue
        system = "You are an expert programmer improving algorithms by evolution."
        cfgp = f"{d}/config.yaml"
        if os.path.exists(cfgp):
            try:
                import yaml
                with open(cfgp) as f:
                    y = yaml.safe_load(f) or {}
                system = ((y.get("prompt") or {}).get("system_message")) or system
            except Exception:
                pass
        parent_block = f"# Current Program\n```\n{prog}\n```\n\n# Task\n{DIFF_TASK}"
        out.append({"name": name, "system": system, "parent_block": parent_block})
    return out


def steer(k):
    return f"Now apply improvement direction #{(k % 4) + 1} as SEARCH/REPLACE diff(s). Output only the diff."


def make_parent_block(ex, parent_uid, pad_tokens):
    """Parent prompt block, optionally padded to ~pad_tokens with filler that is
    UNIQUE per parent (distinct KV — not deduped by APC or Pie's radix trie) yet
    DETERMINISTIC given parent_uid (so the N children of a parent, and any
    revisit, share an identical prefix). ~4 chars/token."""
    base = ex["parent_block"]
    if pad_tokens <= 0:
        return base
    seed = hashlib.sha256(parent_uid.encode()).hexdigest()  # unique per parent
    target_chars = pad_tokens * 4
    lines, n, i = [], 0, 0
    while n < target_chars:
        # each line distinct per (parent, i); seed makes it distinct across parents
        line = f"# ctx-{seed[:16]}-{i:06d}-{seed[16:48]}"
        lines.append(line)
        n += len(line) + 1
        i += 1
    filler = "# --- unique context padding ---\n" + "\n".join(lines)
    return base + "\n\n" + filler


# ───────────────────────── Pie arm ─────────────────────────

class PieRunner:
    def __init__(self):
        from pie_client import Event, PieClient
        self._Event = Event
        self._PieClient = PieClient
        self.uri = os.environ.get("PIE_URI", "ws://127.0.0.1:18099")
        self.user = os.environ.get("PIE_USER", "local-dev")
        self.max_tok = int(os.environ.get("BENCH_MAXTOK", "256"))
        self.a_max_tok = int(os.environ.get("BENCH_ANALYSIS_MAXTOK", "256"))
        self.pad_tokens = int(os.environ.get("BENCH_PREFIX_PAD_TOKENS", "0"))
        self.client = None

    async def __aenter__(self):
        self.client = await self._PieClient(self.uri).__aenter__()
        await self.client.authenticate(self.user)
        wasm, manifest = os.environ.get("PIE_OE_WASM"), os.environ.get("PIE_OE_MANIFEST")
        if wasm and manifest:
            await self.client.install_program(wasm, manifest, force_overwrite=True)
        return self

    async def __aexit__(self, *a):
        await self.client.__aexit__(*a)

    async def _launch(self, payload):
        proc = await self.client.launch_process(INFERLET, input=payload)
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=600)
            if event == self._Event.Return:
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode()
                if isinstance(value, str):
                    value = json.loads(value)
                return value
            if event == self._Event.Error:
                raise RuntimeError(f"pie error: {value!r}")

    async def batch(self, ex, N, parent_uid):
        """B2: one forked call, num_children=N, shared generated analysis."""
        pblock = make_parent_block(ex, parent_uid, self.pad_tokens)
        payload = {
            "run_id": "bench", "parent_id": parent_uid, "topk_sig": parent_uid,
            "sections": {
                "system": ex["system"], "task": "",
                "parent_block": pblock, "analysis_instruction": ANALYSIS_INSTRUCTION,
            },
            "num_children": N, "steers": [steer(k) for k in range(N)],
            "analysis_max_tokens": self.a_max_tok, "child_max_tokens": self.max_tok,
            "analysis_temperature": 0.7, "child_temperature": [0.9], "child_top_p": [0.95],
        }
        t0 = time.time()
        out = await self._launch(payload)
        dt = time.time() - t0
        t = out.get("telemetry") or {}
        return {"wallclock_s": dt, "pie": t, "n_children": len(out.get("children", []))}


# ──────────────────────── vLLM arms ────────────────────────

class VllmRunner:
    def __init__(self, two_phase):
        import openai
        self.two_phase = two_phase
        self.base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self.key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        self.max_tok = int(os.environ.get("BENCH_MAXTOK", "256"))
        self.a_max_tok = int(os.environ.get("BENCH_ANALYSIS_MAXTOK", "256"))
        self.pad_tokens = int(os.environ.get("BENCH_PREFIX_PAD_TOKENS", "0"))
        self.client = openai.AsyncOpenAI(base_url=self.base, api_key=self.key, max_retries=0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def _chat(self, messages, max_tokens):
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=0.9, top_p=0.95,
        )
        u = r.usage
        cached = 0
        det = getattr(u, "prompt_tokens_details", None)
        if det is not None:
            cached = getattr(det, "cached_tokens", 0) or 0
        return (r.choices[0].message.content or ""), {
            "prompt": u.prompt_tokens, "completion": u.completion_tokens, "cached": cached,
        }

    async def batch(self, ex, N, parent_uid):
        sys_msg = ex["system"]
        pblock = make_parent_block(ex, parent_uid, self.pad_tokens)
        p_instr = f"{pblock}\n\n{ANALYSIS_INSTRUCTION}"
        acc = {"phase1_prompt": 0, "phase1_completion": 0, "phase1_cached": 0,
               "phase2_prompt": 0, "phase2_completion": 0, "phase2_cached": 0}
        t0 = time.time()
        if self.two_phase:
            # Phase 1: shared analysis (one), then N phase-2 children (concurrent).
            analysis, u1 = await self._chat(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": p_instr}],
                self.a_max_tok)
            acc["phase1_prompt"] += u1["prompt"]; acc["phase1_completion"] += u1["completion"]
            acc["phase1_cached"] += u1["cached"]

            async def child(k):
                msgs = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": p_instr},
                    {"role": "assistant", "content": analysis},
                    {"role": "user", "content": steer(k)},
                ]
                _, u = await self._chat(msgs, self.max_tok)
                return u
            us = await asyncio.gather(*[child(k) for k in range(N)])
        else:
            # Naive: N single-shot completions (no analysis), concurrent.
            async def child(k):
                msgs = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"{pblock}\n\n{steer(k)}"},
                ]
                _, u = await self._chat(msgs, self.max_tok)
                return u
            us = await asyncio.gather(*[child(k) for k in range(N)])
        for u in us:
            acc["phase2_prompt"] += u["prompt"]; acc["phase2_completion"] += u["completion"]
            acc["phase2_cached"] += u["cached"]
        dt = time.time() - t0
        return {"wallclock_s": dt, "vllm": acc, "n_children": N}


# ───────────────────────── driver ─────────────────────────

async def run():
    arm = os.environ.get("BENCH_ARM", "pie_batch")
    examples = load_examples(os.environ.get("BENCH_EXAMPLES", DEFAULT_EXAMPLES).split(","))
    Ns = [int(x) for x in os.environ.get("BENCH_N", "1,2,4,8").split(",")]
    concs = [int(x) for x in os.environ.get("BENCH_CONC", "1,2,4").split(",")]
    repeats = int(os.environ.get("BENCH_REPEATS", "3"))
    out_path = os.environ.get("BENCH_OUT", f"logs/ab_bench_{arm}.jsonl")
    print(f"arm={arm} examples={[e['name'] for e in examples]} N={Ns} conc={concs} reps={repeats}", flush=True)

    if arm == "pie_batch":
        runner_cm = PieRunner()
    elif arm == "vllm_twophase":
        runner_cm = VllmRunner(two_phase=True)
    elif arm == "vllm_naive":
        runner_cm = VllmRunner(two_phase=False)
    else:
        raise SystemExit(f"unknown BENCH_ARM={arm}")

    results = []
    async with runner_cm as runner:
        # warm up (load weights / prime) — one batch, discarded.
        try:
            await runner.batch(examples[0], 1, f"warmup-{uuid.uuid4().hex[:6]}")
        except Exception as e:
            print(f"warmup error: {e}", flush=True)

        for ex in examples:
            for N in Ns:
                for conc in concs:
                    for rep in range(repeats):
                        # `conc` distinct parents' batches run concurrently.
                        uids = [f"{ex['name']}-N{N}-c{conc}-r{rep}-{i}-{uuid.uuid4().hex[:6]}"
                                for i in range(conc)]
                        t0 = time.time()
                        batches = await asyncio.gather(
                            *[runner.batch(ex, N, u) for u in uids], return_exceptions=True)
                        total_dt = time.time() - t0
                        ok = [b for b in batches if not isinstance(b, Exception)]
                        errs = [repr(b) for b in batches if isinstance(b, Exception)]
                        rec = {
                            "arm": arm, "example": ex["name"], "N": N, "concurrency": conc,
                            "repeat": rep, "total_wallclock_s": total_dt,
                            "per_batch": ok, "errors": errs,
                        }
                        results.append(rec)
                        with open(out_path, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        per = (sum(b["wallclock_s"] for b in ok) / len(ok)) if ok else float("nan")
                        print(f"  {ex['name']:20s} N={N} conc={conc} rep={rep} "
                              f"total={total_dt:5.1f}s per_batch={per:5.1f}s errs={len(errs)}", flush=True)
    print(f"DONE arm={arm} -> {out_path} ({len(results)} cells)", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, OE_ROOT)
    asyncio.run(run())
