#!/usr/bin/env python3
"""Profile prefill: separate the fixed per-call cost from the per-token cost.

## Why a slope and not a ratio

"pie does 186 tok/s, vLLM does 1183" is a ratio of aggregates, and the OpenHands
evaluation's first method lesson is that dividing an aggregate by a count is how
three wrong conclusions got published there ("measure the layer, don't divide the
aggregate"). A single prompt's `tokens / ttfc` folds together per-call overhead
that does not scale with length and per-token compute that does. Those two are
fixed by completely different work, so a number that blends them cannot point at
a bottleneck.

So this sweeps prompt length and fits a line:

    ttfc(n) = intercept + n / marginal_rate

- **intercept** — everything paid once per call regardless of prompt size:
  wasm instantiation, render, address hashing, launch/dispatch, the first
  submit's round trip.
- **marginal_rate** — the actual prefill throughput of the forward pass, which
  is what a kernel or chunk-size lever moves.

A stack can lose badly on either, and the fix is unrelated in each case.

## Reading the result

If the intercept is large relative to a real turn, the bottleneck is per-call
machinery and no kernel work will help. If the marginal rate is low, it is the
forward pass, and chunk size / tile shape are the levers — OpenHands took CUDA
prefill 12.0k → 24.1k tok/s with "chunk 2048 + 64-row tiles" at concurrency 1.

Usage:
    python3 profile_prefill.py --base-url http://127.0.0.1:8080 --model coder30b
"""

import argparse
import json
import os
import sys
import time
import urllib.request

# Filler that tokenizes densely and predictably. Prose, not repeated tokens: a
# repeated single token can hit tokenizer merges and undercount, and some
# backends special-case highly redundant input.
FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer reviews "
    "the serving stack and records what the profiler actually measured. "
)


def prompt_of(words, nonce):
    """Build a prompt of roughly `words` words that shares NO prefix with any
    other prompt in the sweep.

    The nonce leads, and that placement is the whole point. Building the sweep
    by repeating one filler makes every prompt a literal prefix of the longer
    ones, so a backend with prefix caching answers each point almost entirely
    from cache and the sweep measures the cache instead of the prefill. It does
    not look wrong, it looks *fast*: measured at 80,543 tok/s marginal on a 30B
    MoE laptop, which is impossible by about two orders of magnitude, and
    0.045 s for 440 tokens.

    A trailing nonce would not fix it — the shared part would still be the
    prefix. It has to lead.
    """
    n = max(1, words // len(FILLER.split()) + 1)
    return f"[{nonce}] " + (FILLER * n).strip()


def one_call(base_url, model, text, timeout, max_tokens=1):
    """Send one request; return (ttfc, total, prompt_tokens).

    `max_tokens=1` so decode contributes a single token and the measurement is
    dominated by prefill. TTFC is time to first CONTENT, not first chunk — a
    role chunk arrives in milliseconds on both stacks and measures nothing.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer prof"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttfc = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("tool_calls")) and ttfc is None:
                    ttfc = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttfc, total, (usage or {}).get("prompt_tokens", 0)


def fit_line(xs, ys):
    """Least squares slope/intercept. Returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - slope * mx, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--model", default=os.environ.get("PIE_MODEL", "pie"))
    ap.add_argument("--label", default="pie")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--words", default="200,800,1600,3200,4800",
                    help="approximate prompt sizes, in words")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Warm-up on a SHORT unrelated prompt: pays JIT/compile without priming a
    # prefix cache for any measured point.
    try:
        one_call(args.base_url, args.model, "warm up please", args.timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[{args.label}] warm-up failed: {e!r}", file=sys.stderr)
        return 1

    rows = []
    print(f"[{args.label}] prefill sweep against {args.base_url}", flush=True)
    for w in [int(x) for x in args.words.split(",")]:
        best = None
        for rep in range(args.repeat):
            # A fresh nonce per repeat too, or the second sample of each point
            # is a pure cache hit.
            text = prompt_of(w, f"{args.label}-{w}-{rep}-{int(time.time()*1000)}")
            try:
                ttfc, total, ptoks = one_call(args.base_url, args.model, text, args.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"[{args.label}] {w} words FAILED: {e!r}", file=sys.stderr)
                return 1
            if ttfc is None:
                print(f"[{args.label}] {w} words: no content delta", file=sys.stderr)
                continue
            # Min across repeats: the fastest run is the one least perturbed by
            # other load on a shared machine.
            if best is None or ttfc < best[0]:
                best = (ttfc, total, ptoks)
        if best is None:
            continue
        ttfc, total, ptoks = best
        rows.append({"prompt_tokens": ptoks, "ttfc": round(ttfc, 4), "total": round(total, 4)})
        print(f"  {ptoks:>6} tok  ttfc={ttfc:>7.3f}s  ({ptoks/ttfc:>8.1f} tok/s naive)", flush=True)

    if len(rows) >= 2:
        xs = [r["prompt_tokens"] for r in rows]
        ys = [r["ttfc"] for r in rows]
        intercept, slope = fit_line(xs, ys)
        marginal = 1.0 / slope if slope > 0 else float("inf")
        print()
        print(f"[{args.label}] fixed per-call cost : {intercept*1000:8.1f} ms")
        print(f"[{args.label}] marginal prefill    : {marginal:8.1f} tok/s")
        naive = xs[-1] / ys[-1]
        print(f"[{args.label}] naive at {xs[-1]} tok  : {naive:8.1f} tok/s "
              f"({'overhead-dominated' if naive < marginal * 0.7 else 'compute-dominated'})")
        result = {
            "label": args.label,
            "fixed_ms": round(intercept * 1000, 1),
            "marginal_tok_s": round(marginal, 1),
            "rows": rows,
        }
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[{args.label}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
