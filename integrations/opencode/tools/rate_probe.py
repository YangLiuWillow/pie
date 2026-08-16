#!/usr/bin/env python3
"""TTFT and decode rate at fixed prompt sizes, identically on any engine.

## Why this rather than the agentic harness

`tools/pie_ab.sh` showed that one SWE-bench run cannot price a change: opencode's
own prompt varies run to run, so greedy decoding diverges and the agent takes 3
turns or 19. Its wall clock carries a variance larger than most of the
differences it would be used to argue.

This asks a narrower question that IS reproducible: given THE SAME prompt, how
long until the first content byte, and how fast do tokens come after it. Prompt
text is generated from a seed, so every engine and every run sees the same bytes;
the only thing that differs is the server.

TTFT is prefill plus queue plus whatever the engine does before it starts, and
the decode rate is generation. Those are the two halves an agentic turn is made
of, and separating them is the point -- a single wall-clock number hides which
one moved.

## Reading the output

`prompt` is the server's OWN count, not an estimate, so a tokenizer difference
between engines shows up rather than hiding. Decode rate divides the server's
own `completion_tokens` by the time from first content byte to last.

Usage:
    python3 rate_probe.py --base-url http://127.0.0.1:8080 \
        --model qwen3-coder-30b --label pie [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# Three sizes spanning what an agentic turn actually runs at: a first prompt, a
# mid-session context, and a long one. Block counts, not token counts, because
# the tokenizer is the server's business -- the generated TEXT is what is held
# fixed across engines.
BLOCKS = [150, 400, 700]
MAX_TOKENS = 200


def prompt_for(nblocks: int) -> str:
    body = "\n".join(
        f"def handler_{i}(request, *a, **kw):\n"
        f"    ctx = build_context(request, index={i})\n"
        f"    return render(ctx, template='page_{i}.html')\n"
        for i in range(nblocks))
    return "Read this code and write a detailed numbered refactoring plan:\n" + body


def one(base_url: str, model: str, nblocks: int, timeout: float) -> dict:
    req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(nblocks)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    r = urllib.request.urlopen(
        urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            json.dumps(req).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer local"}),
        timeout=timeout)

    t0 = time.time()
    first = None
    n = 0
    usage = None
    buf = b""
    text = []
    while True:
        # read1, not read: `read(n)` blocks until it has n bytes, which would
        # accumulate the whole generation and report TTFT == total.
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, _, buf = buf.partition(b"\n\n")
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if first is None:
                            first = time.time()
                        n += 1
                        text.append(piece)
    t1 = time.time()
    if first is None:
        return {"blocks": nblocks, "error": "no content returned"}
    out = (usage or {}).get("completion_tokens") or n
    dec = t1 - first
    return {
        "blocks": nblocks,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": out,
        "ttft_s": round(first - t0, 3),
        "decode_s": round(dec, 3),
        "decode_tok_s": round(out / dec, 1) if dec > 0 else None,
        # The generated TEXT, so one run answers both questions a kernel change
        # raises. `temperature: 0` makes it a function of the model and the
        # arithmetic alone, which is what lets two arms of an A/B be compared
        # for CORRECTNESS and not only for rate -- a kernel that got faster by
        # computing something else is the failure this catches, and a timing
        # cannot see it.
        "text": "".join(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--json", default=None)
    # Three sizes are enough to compare engines and NOT enough to tell an
    # outlier from a curve. Split-K decode attention read 1.02x / 0.94x / 1.09x
    # across the default three, and a dip between two gains is exactly the shape
    # that needs points either side of it before it means anything.
    ap.add_argument("--blocks", default=None,
                    help="comma-separated block counts, overriding the default three")
    a = ap.parse_args()

    blocks = ([int(x) for x in a.blocks.split(",") if x.strip()]
              if a.blocks else BLOCKS)
    rows = []
    for nb in blocks:
        try:
            row = one(a.base_url, a.model, nb, a.timeout)
        except Exception as e:  # a dead server is a result, and must say so
            row = {"blocks": nb, "error": f"{type(e).__name__}: {e}"}
        row["label"] = a.label
        rows.append(row)
        if "error" in row:
            print(f"  [{a.label}] blocks={nb} FAILED: {row['error']}", flush=True)
        else:
            print(f"  [{a.label}] prompt={row['prompt_tokens']} out={row['output_tokens']} "
                  f"ttft={row['ttft_s']:.2f}s decode={row['decode_s']:.2f}s "
                  f"-> {row['decode_tok_s']} tok/s", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
