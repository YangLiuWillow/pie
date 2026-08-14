#!/usr/bin/env python3
"""Decode rate by slope, so an agent call can be split into prefill and decode.

## Why a slope and not a division

A single request's `output_tokens / elapsed` folds in prefill of the prompt and
the fixed per-call cost, neither of which scales with the number of tokens
generated. Asking for two different output lengths from the SAME prompt makes
both cancel:

    elapsed(n) = fixed + prefill(prompt) + n / decode_rate
    decode_rate = (n_long - n_short) / (t_long - t_short)

Everything that does not depend on `n` drops out of the difference. This is the
same reason `profile_prefill.py` fits a line over prompt length rather than
dividing an aggregate.

## What it is for

With a prefill rate and a decode rate for a stack, each agent turn recorded in
opencode's store can be attributed:

    predicted = input_tokens / prefill_rate + output_tokens / decode_rate

and the residual against the measured call duration is the part neither
explains — per-call overhead, queueing, or a wrong assumption about caching.
Reporting the residual is the point: it is what stops the decomposition from
being a story that always fits.

Usage:
    python3 decode_rate.py --base-url http://127.0.0.1:8080 --model qwen3-coder-30b
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

# Short and fixed, so prefill is identical across both points and cancels.
PROMPT = "Count slowly and describe each number in one short sentence."


def call(base_url, model, max_tokens, timeout=900):
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        # `ignore_eos` where supported keeps the model from stopping early and
        # making the two points incomparable; absent it, the check below catches
        # a short generation instead of silently reporting a wrong slope.
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer probe"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    elapsed = time.time() - t0
    return elapsed, d["usage"]["completion_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--long", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    pairs = []
    for i in range(args.repeat):
        ts, ns = call(args.base_url, args.model, args.short)
        tl, nl = call(args.base_url, args.model, args.long)
        if nl <= ns:
            print(f"  trial {i+1}: model stopped at {nl} tokens (<= {ns}); "
                  f"cannot form a slope from this pair — skipped")
            continue
        rate = (nl - ns) / (tl - ts)
        pairs.append(rate)
        print(f"  trial {i+1}: {ns} tok in {ts:.2f}s, {nl} tok in {tl:.2f}s "
              f"-> {rate:.1f} tok/s")

    if not pairs:
        print("no usable pair; the model stops before the long target every time")
        return 1
    # Max across trials: the fastest run is the one least perturbed by other
    # load, matching profile_prefill.py's min-of-repeats reasoning.
    best = max(pairs)
    print(f"\ndecode rate: {best:.1f} tok/s (best of {len(pairs)}, "
          f"median {statistics.median(pairs):.1f})")
    if args.out:
        json.dump({"decode_tok_s": best, "trials": pairs}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
