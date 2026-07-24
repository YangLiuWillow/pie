#!/usr/bin/env python3
# =============================================================================
# Identical-measurement decode microbenchmark.
#
# Fixes the measurement-fairness bug called out in the writeup: it currently
# compares 85.5 tok/s (Pie, GPU-only) against 64.2 tok/s (vLLM, end-to-end incl.
# HTTP + tokenization) — not apples-to-apples. This script measures BOTH engines
# the SAME way: end-to-end from one client, over an OpenAI-compatible endpoint,
# streaming, generating exactly N tokens at batch 1 (ignore_eos so both emit
# the same count). It reports TTFT (prefill-ish) and steady-state decode tok/s.
#
#   - vLLM: point --base-url at the vLLM serve (http://localhost:18000/v1).
#   - Pie:  point --base-url at Pie's OpenAI-compatible adapter endpoint.
#           (Both go through the same client codepath here, so the comparison is
#            honest. If your Pie build has no OpenAI HTTP shim, take Pie's number
#            from the harness's own per-call decode timing measured client-side —
#            same clock, same place — and compare that instead.)
#
# Run each engine separately (fresh server, no cross-contention) and record both:
#   python 20_decode_microbench.py --base-url http://localhost:18000/v1 \
#          --model Qwen/Qwen3-Coder-30B-A3B-Instruct --label vllm-fair
#   python 20_decode_microbench.py --base-url http://localhost:18100/v1 \
#          --model Qwen/Qwen3-Coder-30B-A3B-Instruct --label pie
# =============================================================================
from __future__ import annotations

import argparse
import json
import time
import urllib.request


PROMPT = (
    "You are a senior software engineer. Explain, in careful and thorough "
    "detail, how a copy-on-write B-tree works, including node splitting, "
    "reference counting, and how concurrent readers are isolated from writers. "
    "Then walk through a worked insertion example step by step."
)


def stream_completion(base_url: str, model: str, max_tokens: int, api_key: str):
    """Yield (token_text, wallclock_time) for each streamed chunk."""
    url = base_url.rstrip("/") + "/completions"
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,          # force exactly max_tokens on both engines
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or choices[0].get("delta", {}).get("content") or ""
            yield text, time.perf_counter()


def run_once(base_url, model, max_tokens, api_key):
    t0 = time.perf_counter()
    ttft = None
    n = 0
    last = t0
    for text, t in stream_completion(base_url, model, max_tokens, api_key):
        if ttft is None:
            ttft = t - t0        # time to first token ~ prefill + queue + net
        n += 1
        last = t
    total = last - t0
    decode_time = max(total - (ttft or 0.0), 1e-9)
    # tokens-after-first over the decode window = steady-state decode rate
    decode_toks = max(n - 1, 0)
    return {
        "tokens": n,
        "ttft_s": ttft,
        "total_s": total,
        "decode_tok_s": decode_toks / decode_time,
        "e2e_tok_s": n / max(total, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base, e.g. http://localhost:18000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="engine")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--api-key", default="dummy")
    args = ap.parse_args()

    print(f"=== decode microbench :: {args.label} ===")
    print(f"  base_url={args.base_url}  model={args.model}")
    print(f"  max_tokens={args.max_tokens}  warmup={args.warmup}  reps={args.reps}  batch=1")

    for _ in range(args.warmup):
        run_once(args.base_url, args.model, args.max_tokens, args.api_key)

    rows = [run_once(args.base_url, args.model, args.max_tokens, args.api_key)
            for _ in range(args.reps)]

    def med(key):
        xs = sorted(r[key] for r in rows)
        return xs[len(xs) // 2]

    print()
    print(f"  {'rep':>3} {'tokens':>7} {'ttft_s':>8} {'total_s':>8} {'decode_tok/s':>13} {'e2e_tok/s':>10}")
    for i, r in enumerate(rows):
        print(f"  {i:>3} {r['tokens']:>7} {r['ttft_s']:>8.3f} {r['total_s']:>8.3f} "
              f"{r['decode_tok_s']:>13.1f} {r['e2e_tok_s']:>10.1f}")
    print("  " + "-" * 56)
    print(f"  MEDIAN decode_tok/s = {med('decode_tok_s'):.1f}   "
          f"e2e_tok/s = {med('e2e_tok_s'):.1f}   ttft = {med('ttft_s'):.3f}s")
    print()
    print("  Report decode_tok/s for BOTH engines (measured this same way).")


if __name__ == "__main__":
    main()
