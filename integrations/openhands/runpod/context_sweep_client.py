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
# Prefill mode — the same differencing idea, turned 90 degrees.
#
# WHY IT IS NEEDED. `context` mode deliberately CANCELS prefill: it holds the
# prompt fixed and varies generation, so everything generation-independent drops
# out. That made decode comparable across two unrelated client stacks, and it is
# why prefill has never been compared between the arms at all. With decode now at
# ~1.06x of vLLM (AGENT_HANDOVER_20260728.md 6a-nonies), prefill is the largest
# remaining stage: it is 21.7% of Pie's per-call time on the H100 c1 arm.
#
# THE MEASUREMENT. Hold GENERATION fixed and vary the number of NEW prompt tokens:
#
#     prefill_ms_per_token = (latency(S_long) - latency(S_short))
#                            / (S_long - S_short)
#
# Per-call fixed cost, transport, render, launch and the cached-prefix lookup are
# all independent of S, so they cancel exactly as prefill did in `context` mode.
#
# CONTROLLING CACHE STATE IS THE WHOLE GAME. If the shared prefix is not already
# resident, the "new tokens" count is wrong and this silently measures cold
# prefill on one arm and warm on the other. So every size gets its own session /
# prefix warm on the SAME base prompt first, and only the appended suffix is new.
# That also matches how the agent workload actually behaves -- an append-only
# conversation that grows -- rather than inventing a pattern neither engine sees.
#
# WHAT SURVIVES THE SUBTRACTION BUT SHOULDN'T. Decode at the longer context costs
# marginally more (the ctx-mode slope, ~0.024 ms per 1k KV tokens). Over a 4k
# suffix and 8 generated tokens that is ~0.8 ms against hundreds of ms of
# prefill, i.e. well under 1%. Reported so the reader can check rather than
# assume, but not corrected for.
# --------------------------------------------------------------------------
def build_prefill_case(tok, base_tokens: int, suffix_tokens: int, uid: str):
    """Base prompt (shared, cacheable) + a unique suffix of ~suffix_tokens.

    The uid marker sits at the START of the suffix so the prefix cache matches
    exactly the base and nothing after it -- the suffix is genuinely new work on
    both engines. Returns (warm_messages, measured_messages, actual_new_tokens).
    """
    base, _ = build_prompt(tok, base_tokens)
    tail = "\n\nSummarize what the function above does, in detail."

    warm_content = base + tail
    if suffix_tokens <= 0:
        return messages_for(base), messages_for(base), 0

    marker = f"\n# variant {uid}\n"
    suffix, _ = build_prompt(tok, suffix_tokens)
    measured_content = base + marker + suffix + tail

    warm_msgs = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": warm_content}]
    meas_msgs = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": measured_content}]

    # The warm call is  system + base + tail;  the measured call is
    # system + base + marker + suffix + tail. Their longest common prefix is
    # system + base, because the two diverge at the first character after `base`.
    # So the cache covers `base` and the new work is marker + suffix + tail.
    #
    # Counted with the tokenizer rather than assumed: BPE does not guarantee that
    # asking for N tokens of filler yields exactly N, and the exact figure is the
    # denominator of the whole measurement.
    n_base = len(tok(base, add_special_tokens=False)["input_ids"])
    n_meas = len(tok(measured_content, add_special_tokens=False)["input_ids"])
    return warm_msgs, meas_msgs, n_meas - n_base


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
# Pie batch arm — decode cost vs CONCURRENCY at fixed context.
#
# This exists to test one specific cliff. Routing decode through the prefill
# kernel (PIE_QWEN35_TENSOR_CORE_DECODE) only keeps CUDA graphs while
# total_tokens <= qwen35_small_spec_graph_tokens() (default 17), and on decode
# total_tokens IS the request count -- so graphs are expected to drop above
# R=17. The context sweep runs at batch 1 and cannot see this at all.
#
# Same differencing trick, one level up: fire N streams short, then N streams
# long, and subtract. With equal per-stream generation lengths the batch wall is
# prefill + T * t_forward(R=N), so the subtraction yields t_forward at that R
# directly. Streams share prompt text (distinct session ids) so their output
# lengths match and R stays flat for the whole measured window -- unequal
# lengths would let R decay mid-run and quietly average two batch sizes.
# --------------------------------------------------------------------------
async def run_pie_batch(args, tok, concurrencies):
    from pie_client import Event, PieClient  # noqa: E402

    inferlet_dir = os.path.join(args.repo, "inferlets", "openhands-coder-session")
    wasm = os.path.join(inferlet_dir, "target", "wasm32-wasip2", "release",
                        "openhands_coder_session.wasm")
    manifest = os.path.join(inferlet_dir, "Pie.toml")

    async def one(client, msgs, max_tokens, session_id):
        payload = {"messages": msgs, "max_tokens": max_tokens,
                   "temperature": 0.0, "session_id": session_id}
        proc = await client.launch_process(args.inferlet, input=payload)
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=args.timeout)
            if event == Event.Return:
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode()
                out = json.loads(value) if isinstance(value, str) else value
                return int(out.get("tokens_generated") or 0)
            if event == Event.Error:
                raise RuntimeError(f"inferlet error: {value!r}")

    async def wave(client, msgs, max_tokens, n, tag):
        t0 = time.perf_counter()
        counts = await asyncio.gather(*[
            one(client, msgs, max_tokens, f"batch_{tag}_{i}") for i in range(n)])
        return (time.perf_counter() - t0) * 1000.0, sum(counts)

    filler, want = build_prompt(tok, args.batch_context)
    msgs = messages_for(filler)
    rows = []
    async with PieClient(args.uri) as client:
        await client.authenticate("local-dev")
        await client.install_program(wasm, manifest, force_overwrite=True)
        for n in concurrencies:
            await wave(client, msgs, args.short_tokens, n, f"w{n}")  # warm
            per = []
            for rep in range(args.reps):
                ws, ns = await wave(client, msgs, args.short_tokens, n, f"s{n}_{rep}")
                wl, nl = await wave(client, msgs, args.long_tokens, n, f"l{n}_{rep}")
                if nl > ns:
                    # (wall_long - wall_short) / tokens-per-stream-delta
                    per.append((wl - ws) / ((nl - ns) / n))
            t_fwd = st.median(per) if per else None
            rows.append({
                "concurrency": n,
                "ctx_requested": args.batch_context,
                "filler_tokens": want,
                "samples": len(per),
                "t_forward_ms": t_fwd,
                "aggregate_tok_s": (n / t_fwd * 1000.0) if t_fwd else None,
            })
            r = rows[-1]
            print(f"  R={n:>3} t_forward={('' if t_fwd is None else round(t_fwd,3)):>8} ms"
                  f"   aggregate={('' if not r['aggregate_tok_s'] else round(r['aggregate_tok_s'],1)):>8} tok/s",
                  flush=True)
    return rows


# --------------------------------------------------------------------------
# Prefill sweep — Pie arm.
#
# Each suffix size gets a FRESH session id. Within that session the warm call
# establishes the base prompt and the measured call extends it, so Pie takes its
# `extended` path (the one the agent workload uses) rather than `rebuilt`. Using
# one session across different suffixes would diverge the context and force a
# rebuild, which prefills EVERYTHING and would silently report the wrong number
# -- larger, and in a way that looks like a real result.
# --------------------------------------------------------------------------
async def run_pie_prefill(args, tok, suffixes):
    from pie_client import Event, PieClient  # noqa: E402

    inferlet_dir = os.path.join(args.repo, "inferlets", "openhands-coder-session")
    wasm = os.path.join(inferlet_dir, "target", "wasm32-wasip2", "release",
                        "openhands_coder_session.wasm")
    manifest = os.path.join(inferlet_dir, "Pie.toml")

    async def one(client, msgs, session_id):
        payload = {"messages": msgs, "max_tokens": args.short_tokens,
                   "temperature": 0.0, "session_id": session_id}
        t0 = time.perf_counter()
        proc = await client.launch_process(args.inferlet, input=payload)
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
        return lat, (out.get("timings") or {}), (out.get("session") or {})

    rows = []
    async with PieClient(args.uri) as client:
        await client.authenticate("local-dev")
        await client.install_program(wasm, manifest, force_overwrite=True)
        for s in suffixes:
            lats, self_pref, news = [], [], []
            for rep in range(args.reps):
                uid = f"{s}_{rep}"
                sid = f"prefillsweep_{args.prefill_base}_{uid}"
                warm, meas, n_new = build_prefill_case(
                    tok, args.prefill_base, s, uid)
                await one(client, warm, sid)              # cache the base
                lat, tim, sess = await one(client, meas, sid)
                lats.append(lat)
                news.append(n_new)
                # Pie's own attribution, as an independent check on the
                # differencing -- exactly as `context` mode does for decode.
                self_pref.append(float(tim.get("prefill_ms") or 0.0))
                if sess.get("mode") == "rebuilt":
                    print(f"    WARN suffix={s} rep={rep}: pie reported "
                          f"mode=rebuilt, not extended -- this sample prefilled "
                          f"the whole prompt and must not be trusted", flush=True)
            rows.append(prefill_row(s, news, lats, self_pref))
            report_prefill(rows[-1])
    return rows


# --------------------------------------------------------------------------
# Prefill sweep — vLLM arm. Same procedure; APC supplies the prefix reuse that
# Pie gets from the session id, so the warm call is what makes the two
# comparable rather than any engine-specific flag.
# --------------------------------------------------------------------------
def run_vllm_prefill(args, tok, suffixes):
    import urllib.request

    def one(msgs):
        body = json.dumps({
            "model": args.model, "messages": msgs,
            "max_tokens": args.short_tokens, "temperature": 0.0, "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{args.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            out = json.loads(r.read().decode())
        return (time.perf_counter() - t0) * 1000.0, (out.get("usage") or {})

    rows = []
    for s in suffixes:
        lats, news = [], []
        for rep in range(args.reps):
            uid = f"{s}_{rep}"
            warm, meas, n_new = build_prefill_case(tok, args.prefill_base, s, uid)
            one(warm)                                     # cache the base
            lat, _ = one(meas)
            lats.append(lat)
            news.append(n_new)
        rows.append(prefill_row(s, news, lats, []))
        report_prefill(rows[-1])
    return rows


def prefill_row(suffix, news, lats, self_prefill_ms):
    return {
        "suffix_requested": suffix,
        "new_tokens": int(st.median(news)) if news else None,
        "samples": len(lats),
        "latency_ms": st.median(lats) if lats else None,
        "latency_ms_spread": (max(lats) - min(lats)) if len(lats) > 1 else None,
        # Pie only; vLLM exposes no per-phase attribution, so it stays null.
        "self_reported_prefill_ms":
            st.median(self_prefill_ms) if self_prefill_ms else None,
    }


def report_prefill(r):
    s = r["self_reported_prefill_ms"]
    print(f"  suffix~{r['suffix_requested']:>6} new_tokens={str(r['new_tokens']):>7} "
          f"latency={r['latency_ms'] if r['latency_ms'] is None else round(r['latency_ms'], 1)} ms"
          + (f"  (self-reported prefill {round(s, 1)} ms)" if s else ""), flush=True)


def fit_prefill(rows):
    """Least-squares line through (new_tokens, latency_ms).

    slope     = ms per NEW prompt token  -> 1000/slope = prefill tok/s
    intercept = per-call cost that does not depend on how much is prefilled
                (transport, render, scheduling, and the fixed generation)
    """
    pts = [(r["new_tokens"], r["latency_ms"]) for r in rows
           if r["new_tokens"] and r["latency_ms"]]
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
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
    return {
        "prefill_ms_per_token": slope,
        "prefill_tokens_per_s": (1000.0 / slope) if slope > 0 else None,
        "fixed_per_call_ms": intercept,
        "r2": (1 - ss_res / ss_tot) if ss_tot else None,
    }


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
    p.add_argument("--mode", choices=["context", "batch", "prefill"],
                   default="context")
    p.add_argument("--concurrencies", default="1,4,8,16,20,24,32,48")
    p.add_argument("--batch-context", type=int, default=16000)
    # Prefill mode. The base is the cached shared prefix; the suffixes are the
    # NEW tokens whose cost we are measuring. Defaults bracket the agent
    # workload: the H100 c1 arm prefilled ~1,008 new tokens per call on average.
    p.add_argument("--prefill-base", type=int, default=8000)
    p.add_argument("--prefill-suffixes", default="256,512,1024,2048,4096")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    if args.mode == "batch":
        conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
        print(f"=== batch sweep: arm={args.arm} concurrencies={conc} "
              f"ctx={args.batch_context} reps={args.reps}")
        if args.arm != "pie":
            print("batch mode is implemented for the pie arm only")
            raise SystemExit(2)
        rows = asyncio.run(run_pie_batch(args, tok, conc))
        f = None
    elif args.mode == "prefill":
        suffixes = [int(s) for s in args.prefill_suffixes.split(",") if s.strip()]
        print(f"=== prefill sweep: arm={args.arm} base={args.prefill_base} "
              f"suffixes={suffixes} gen={args.short_tokens} reps={args.reps}")
        if args.arm == "pie":
            rows = asyncio.run(run_pie_prefill(args, tok, suffixes))
        else:
            rows = run_vllm_prefill(args, tok, suffixes)
        f = fit_prefill(rows)
        print(f"\n=== {args.arm} prefill fit: {json.dumps(f)}")
    else:
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
            json.dump({"arm": args.arm, "mode": args.mode,
                       "rows": rows, "fit": f}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
