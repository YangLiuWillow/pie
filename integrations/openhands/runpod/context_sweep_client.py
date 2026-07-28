#!/usr/bin/env python3
"""Decode cost vs context length, measured identically on Pie and vLLM.

WHY DIFFERENCING. We want ms/decoded-token as a function of KV length. Measuring
one call and dividing does not give that: the call also carries prefill, chat
rendering, transport and process launch, and those differ per engine. So for each
context we issue the SAME prompt twice, once with a short generation and once
with a long one, and take

    decode_ms_per_token = (latency_long - latency_short)
                          / (tokens_long - tokens_short)

Everything context-dependent but generation-independent cancels: prefill, render,
transport, launch. What survives is decode. The subtraction is what makes the two
arms comparable despite completely different client stacks -- a constant per-call
overhead, however large, drops out. Actual token counts come from each engine's
own report, so an early EOS shortens the long run without biasing the result.

PREFIX CACHE. Both engines cache prefixes, so an uncached first call would put
full prefill in the short run and none in the long one, and the subtraction would
silently return decode minus prefill. Every context therefore gets a warmup call
on the same prompt first, leaving both measured calls equally cached.

The slope of the resulting line against context is KV read bandwidth; the
intercept is the context-independent per-token cost (weights, MoE, router,
launch). Those two numbers are the whole point -- they say which half of decode
to attack, and comparing them across arms says how much is recoverable.

Usage:
  context_sweep_client.py --arm pie   --uri ws://127.0.0.1:18080
  context_sweep_client.py --arm vllm  --base-url http://127.0.0.1:18000/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
SYSTEM = "You are a coding agent."

# Filler that tokenizes densely and predictably. Real code-ish text, so the
# prompt looks like the SWE-bench workload rather than a pathological repeat of
# one token (which would compress oddly under BPE and understate prompt length).
FILLER = (
    "def process_record(record, index, config):\n"
    "    value = record.get('value', 0)\n"
    "    if value > config.threshold and index % 2 == 0:\n"
    "        return {'ok': True, 'value': value * 2, 'index': index}\n"
    "    return {'ok': False, 'value': value, 'index': index}\n\n"
)


def build_prompt(tok, target_tokens: int) -> tuple[str, int]:
    """Return filler text whose rendered length is close to target_tokens."""
    per = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    reps = max(1, target_tokens // per)
    text = FILLER * reps
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    # Trim/extend by whole repetitions until within ~2% of the target.
    while n > target_tokens * 1.02 and reps > 1:
        reps -= 1
        text = FILLER * reps
        n = len(tok(text, add_special_tokens=False)["input_ids"])
    while n < target_tokens * 0.98:
        reps += 1
        text = FILLER * reps
        n = len(tok(text, add_special_tokens=False)["input_ids"])
    return text, n


def messages_for(filler: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            filler + "\n\nSummarize what the function above does, in detail."},
    ]


# --------------------------------------------------------------------------
# Pie arm
# --------------------------------------------------------------------------
async def run_pie(args, tok, contexts):
    # pie_client is installed in the harness venv (/root/venvs/harness), which is
    # the interpreter this must run under. The pie-vllm venv does not have it.
    from pie_client import Event, PieClient  # noqa: E402

    inferlet_dir = os.path.join(args.repo, "inferlets", "openhands-coder-session")
    wasm = os.path.join(inferlet_dir, "target", "wasm32-wasip2", "release",
                        "openhands_coder_session.wasm")
    manifest = os.path.join(inferlet_dir, "Pie.toml")

    async def one(client, msgs, max_tokens, session_id):
        payload = {"messages": msgs, "max_tokens": max_tokens,
                   "temperature": 0.0, "session_id": session_id}
        t0 = time.perf_counter()
        proc = await client.launch_process(args.inferlet, input=payload)
        out = None
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=args.timeout)
            if event == Event.Return:
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode()
                out = json.loads(value) if isinstance(value, str) else value
                break
            if event == Event.Error:
                raise RuntimeError(f"inferlet error: {value!r}")
        lat = (time.perf_counter() - t0) * 1000.0
        ntok = int(out.get("tokens_generated") or 0)
        timings = out.get("timings") or {}
        session = out.get("session") or {}
        return lat, ntok, timings, session

    rows = []
    async with PieClient(args.uri) as client:
        await client.authenticate("local-dev")
        await client.install_program(wasm, manifest, force_overwrite=True)
        for ctx in contexts:
            filler, want = build_prompt(tok, ctx)
            msgs = messages_for(filler)
            sid = f"sweep_{ctx}"
            await one(client, msgs, args.short_tokens, sid)  # warm the prefix
            shorts, longs, decode_ms, klen = [], [], [], None
            for _ in range(args.reps):
                ls, ns, _, sess = await one(client, msgs, args.short_tokens, sid)
                ll, nl, tl, _ = await one(client, msgs, args.long_tokens, sid)
                if nl <= ns:
                    continue
                shorts.append((ls, ns))
                longs.append((ll, nl))
                decode_ms.append(float(tl.get("decode_ms") or 0.0) / max(1, nl))
                klen = sess.get("len") or klen
            rows.append(summarize(ctx, want, klen, shorts, longs, decode_ms))
            report(rows[-1])
    return rows


# --------------------------------------------------------------------------
# vLLM arm
# --------------------------------------------------------------------------
def run_vllm(args, tok, contexts):
    import urllib.request

    def one(msgs, max_tokens):
        body = json.dumps({
            "model": args.model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{args.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            out = json.loads(r.read().decode())
        lat = (time.perf_counter() - t0) * 1000.0
        usage = out.get("usage") or {}
        return lat, int(usage.get("completion_tokens") or 0), int(
            usage.get("prompt_tokens") or 0)

    rows = []
    for ctx in contexts:
        filler, want = build_prompt(tok, ctx)
        msgs = messages_for(filler)
        one(msgs, args.short_tokens)  # warm the prefix
        shorts, longs, klen = [], [], None
        for _ in range(args.reps):
            ls, ns, pt = one(msgs, args.short_tokens)
            ll, nl, _ = one(msgs, args.long_tokens)
            if nl <= ns:
                continue
            shorts.append((ls, ns))
            longs.append((ll, nl))
            klen = pt
        rows.append(summarize(ctx, want, klen, shorts, longs, []))
        report(rows[-1])
    return rows


# --------------------------------------------------------------------------
def summarize(ctx, want, klen, shorts, longs, self_decode_ms):
    per = [(ll - ls) / (nl - ns)
           for (ls, ns), (ll, nl) in zip(shorts, longs) if nl > ns]
    return {
        "ctx_requested": ctx,
        "filler_tokens": want,
        "prompt_tokens": klen,
        "samples": len(per),
        "decode_ms_per_token": st.median(per) if per else None,
        "decode_ms_per_token_spread": (max(per) - min(per)) if len(per) > 1 else None,
        # Pie only: the engine's own decode attribution, as an independent check
        # on the differencing. vLLM exposes no equivalent, so it stays null.
        "self_reported_decode_ms_per_token":
            st.median(self_decode_ms) if self_decode_ms else None,
    }


def report(r):
    d = r["decode_ms_per_token"]
    s = r["self_reported_decode_ms_per_token"]
    print(f"  ctx~{r['ctx_requested']:>6} prompt_tokens={str(r['prompt_tokens']):>7} "
          f"decode={d if d is None else round(d, 4)} ms/tok"
          + (f"  (self-reported {round(s, 4)})" if s else ""), flush=True)


def fit(rows):
    """Least-squares line through (prompt_tokens, ms/token)."""
    pts = [(r["prompt_tokens"], r["decode_ms_per_token"]) for r in rows
           if r["prompt_tokens"] and r["decode_ms_per_token"]]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    denom = sum((p[0] - mx) ** 2 for p in pts)
    if denom == 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / denom
    intercept = my - slope * mx
    # 96 KiB of KV per token for this model: 48 layers x 4 kv heads x 128 dim
    # x 2 (K and V) x 2 bytes.
    kv_bytes_per_token = 48 * 4 * 128 * 2 * 2
    gbps = (kv_bytes_per_token / (slope / 1000.0)) / 1e9 if slope > 0 else None
    return {"slope_ms_per_ktoken": slope * 1000.0,
            "intercept_ms": intercept,
            "kv_read_GB_per_s": gbps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["pie", "vllm"], required=True)
    p.add_argument("--uri", default="ws://127.0.0.1:18097")
    p.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--repo", default="/workspace/pie")
    p.add_argument("--inferlet", default="openhands-coder-session@0.1.0")
    p.add_argument("--contexts", default="1000,4000,8000,16000,24000,32000")
    p.add_argument("--short-tokens", type=int, default=8)
    p.add_argument("--long-tokens", type=int, default=136)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    print(f"=== context sweep: arm={args.arm} contexts={contexts} "
          f"short={args.short_tokens} long={args.long_tokens} reps={args.reps}")

    if args.arm == "pie":
        rows = asyncio.run(run_pie(args, tok, contexts))
    else:
        rows = run_vllm(args, tok, contexts)

    f = fit(rows)
    print(f"\n=== {args.arm} fit: {json.dumps(f)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"arm": args.arm, "rows": rows, "fit": f}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
