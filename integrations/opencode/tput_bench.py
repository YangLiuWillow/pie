#!/usr/bin/env python3
"""Concurrent-throughput comparison across OpenAI-compatible engines.

## What this measures, and why it is a separate harness

`bench_ab.py` replays ONE growing conversation and reports per-turn latency.
That is the agentic shape and it is what an opencode session actually does, but
it says nothing about what an engine does when several requests are in flight —
which is the number `pie-project.org/preview` quotes for Apple Silicon ("8
concurrent", 108.5-390.7 tok/s). This is that measurement, on our hardware and
our weights, so the two can be compared like for like.

## The design decisions, each of which can bias the answer

**Unique prompts.** Every request gets a distinct prefix. Otherwise the engines
with a prompt cache (all three of ours) serve later requests from cache and the
number becomes a cache-hit-rate measurement wearing a throughput costume. This
follows the upstream bench's own fairness default, which disables prefix caching
outright; we cannot disable it uniformly across three engines, so we defeat it
with the input instead.

**Aggregate tokens/second is computed from the SERVER's own usage counts**, not
from a token estimate. `completion_tokens` summed over all requests, divided by
the wall time from first submit to last completion. An engine that stops early
scores lower, correctly — it produced fewer tokens in that window.

**`max_tokens` is a cap, not a target**, and the engines will not agree on how
many tokens they emit for the same prompt. That is why the metric is
tokens-per-second rather than time-to-finish: dividing by the actual token count
makes runs with different output lengths comparable.

**One engine at a time.** Two 30B-class servers do not fit on a 48 GB machine —
that is measured, not assumed: a vLLM arm OOM'd and dead-latched on
2026-08-14 while a peer held 12.5 GB. Boot, measure, kill, next.

Usage:
    python3 tput_bench.py --base-url http://127.0.0.1:8001 \
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
        --arm mlx --concurrency 8 --requests 32 --max-tokens 128
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# A prompt long enough to make prefill matter but short enough that 32 of them
# do not dominate the run. ~300-400 tokens once the unique preamble is added.
BODY = (
    "You are reviewing a Python service. Explain, in careful prose, how you "
    "would investigate a latency regression in a request handler that calls a "
    "database, a cache, and a downstream HTTP service. Cover instrumentation, "
    "how to separate the three dependencies, and what you would measure first. "
)


def make_prompt(i: int) -> str:
    # The unique part goes FIRST. A shared prefix with a unique suffix is
    # exactly what a prefix cache is built to exploit, so putting the unique
    # token at the front defeats every one of them symmetrically.
    return f"Case {i}-{i * 7919 % 100003}: incident report {i}. " + BODY


def one_request(url, model, prompt, max_tokens, timeout, out, idx):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer bench"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d.get("usage") or {}
        out[idx] = {
            "ok": True,
            "latency": time.perf_counter() - t0,
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
        }
    except Exception as e:  # noqa: BLE001
        out[idx] = {"ok": False, "latency": time.perf_counter() - t0,
                    "completion_tokens": 0, "prompt_tokens": 0,
                    "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # Warm-up is NOT timed and uses prompts outside the measured set: the first
    # request after a boot pays kernel/program compilation, and charging that to
    # whichever arm booted last is how a tuning difference gets misread as an
    # engine difference.
    if a.warmup:
        print(f"[{a.arm}] warming up ({a.warmup})...", flush=True)
        w = [None] * a.warmup
        ts = [threading.Thread(target=one_request,
                               args=(a.base_url, a.model, f"warmup {i}: {BODY}",
                                     16, a.timeout, w, i))
              for i in range(a.warmup)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if not all(x and x["ok"] for x in w):
            print(f"[{a.arm}] warm-up FAILED: {w}", file=sys.stderr)
            return 1

    results = [None] * a.requests
    sem = threading.Semaphore(a.concurrency)
    threads = []

    def worker(i):
        with sem:
            one_request(a.base_url, a.model, make_prompt(i),
                        a.max_tokens, a.timeout, results, i)

    print(f"[{a.arm}] {a.requests} requests at concurrency {a.concurrency}", flush=True)
    t0 = time.perf_counter()
    for i in range(a.requests):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and r["ok"]]
    bad = [r for r in results if not (r and r["ok"])]
    gen = sum(r["completion_tokens"] for r in ok)
    prm = sum(r["prompt_tokens"] for r in ok)
    lat = sorted(r["latency"] for r in ok)

    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0.0

    summary = {
        "arm": a.arm, "concurrency": a.concurrency, "requests": a.requests,
        "completed": len(ok), "failed": len(bad),
        "wall_s": round(wall, 3),
        "output_tokens": gen, "prompt_tokens": prm,
        "output_tok_per_s": round(gen / wall, 1) if wall else 0.0,
        "prompt_tok_per_s": round(prm / wall, 1) if wall else 0.0,
        "req_per_s": round(len(ok) / wall, 3) if wall else 0.0,
        "latency_mean_s": round(statistics.fmean(lat), 2) if lat else None,
        "latency_p50_s": round(pct(0.50), 2),
        "latency_p95_s": round(pct(0.95), 2),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if bad:
        print(f"[{a.arm}] {len(bad)} FAILED, first: {bad[0].get('error')}", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "requests": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
