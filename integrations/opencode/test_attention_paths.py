#!/usr/bin/env python3
"""Greedy gate: do pie's attention implementations agree, token for token?

## Why this has to exist before the MMA change

The Metal driver has a fast matrix-unit attention (`sdpa_paged_mma.metal`) that
is instantiated only for gpt-oss's head width. Wiring it for Qwen is a four-file
change and would be worth ~1.9× on prefill — but the driver warns that the
matrix path "depends on the register layout of `simdgroup_matrix<T,8,8>` … a
machine whose layout differs would produce **wrong numbers rather than slow
ones**", and a new head width is exactly where that bites.

Nothing else in this repo would catch it. The acceptance suite asserts wire
shape; the resume suite asserts cache behaviour. **Both pass at 25/25 and 5/5 on
subtly wrong attention output**, because neither ever looks at a logit.

So: compare the actual generated tokens across attention implementations, at
temperature 0. Greedy decoding is the most sensitive cheap probe there is — it
turns any numerical drift large enough to flip one argmax into a visible,
exactly-located divergence, and drift too small to flip an argmax anywhere in a
long generation is drift that does not matter.

## How to use it

Capture one run per attention path, then compare:

```sh
# path A — the tiled scalar kernel (today's default for a prefill)
PIE_BASE_URL=... python3 test_attention_paths.py --capture /tmp/attn_tiled.json

# path B — force the per-row kernel by raising the tiling threshold above the
# prompt length, so `sdpa_should_tile` never fires
#   (boot pie with PIE_METAL_SDPA_TILE_MIN_ROWS=1000000)
PIE_BASE_URL=... python3 test_attention_paths.py --capture /tmp/attn_perrow.json

python3 test_attention_paths.py --compare /tmp/attn_tiled.json /tmp/attn_perrow.json
```

When the MMA path lands, capture a third with `PIE_METAL_SDPA_MMA=1` and compare
it against the tiled capture. Same harness, no changes.

**A caveat worth stating**: agreement here is evidence, not proof. Two paths can
agree on these prompts and differ on a shape neither exercises. Widen `PROMPTS`
rather than trusting a green run on three of them.
"""

import argparse
import json
import os
import sys
import urllib.request

BASE = os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("PIE_MODEL", "q06")
TIMEOUT = float(os.environ.get("PIE_TIMEOUT", "600"))

# Prompts chosen to put the attention kernel in different regimes: short enough
# to stay under a tiling threshold, long enough to exceed it comfortably, and
# long enough to span more than one prefill chunk.
_FILLER = (
    "The engineer reviewed the serving stack carefully and wrote down what the "
    "profiler actually measured, rather than what everyone expected it to say. "
)
PROMPTS = [
    ("short", "Count from one to ten, then stop."),
    ("mid", _FILLER * 12 + "\n\nSummarise the paragraph above in one sentence."),
    ("long", _FILLER * 60 + "\n\nSummarise the paragraph above in one sentence."),
]


def generate(prompt, max_tokens=48):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Exact greedy: the engine takes reduce_argmax at temperature 0, so any
        # run-to-run difference is numerical, not sampling.
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer attn"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    usage = d.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def capture(path):
    out = {"base_url": BASE, "model": MODEL, "runs": {}}
    for name, prompt in PROMPTS:
        r = generate(prompt)
        out["runs"][name] = r
        print(f"  {name:>6}: prompt={r['prompt_tokens']:>5} gen={r['completion_tokens']:>3} "
              f"{r['content'][:60]!r}")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    return 0


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    bad = 0
    for name, _ in PROMPTS:
        ra, rb = a["runs"].get(name), b["runs"].get(name)
        if ra is None or rb is None:
            print(f"[SKIP] {name}: missing from one capture")
            continue
        if ra["prompt_tokens"] != rb["prompt_tokens"]:
            print(f"[FAIL] {name}: prompts differ ({ra['prompt_tokens']} vs "
                  f"{rb['prompt_tokens']}) — not an attention comparison")
            bad += 1
            continue
        if ra["content"] == rb["content"]:
            print(f"[PASS] {name}: identical, {ra['completion_tokens']} tokens")
            continue
        bad += 1
        # Locate the divergence rather than dumping two blobs.
        i = next((k for k, (x, y) in enumerate(zip(ra["content"], rb["content"]))
                  if x != y), min(len(ra["content"]), len(rb["content"])))
        lo = max(0, i - 40)
        print(f"[FAIL] {name}: diverges at char {i}")
        print(f"          A: ...{ra['content'][lo:i+40]!r}")
        print(f"          B: ...{rb['content'][lo:i+40]!r}")
    print()
    if bad:
        print(f"{bad} of {len(PROMPTS)} prompts DIVERGED — the two attention paths do "
              f"not agree. Do not ship the faster one.")
    else:
        print(f"all {len(PROMPTS)} prompts identical — the paths agree on this set.")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", metavar="OUT")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.capture:
        return capture(args.capture)
    if args.compare:
        return compare(*args.compare)
    ap.error("pass --capture OUT or --compare A B")


if __name__ == "__main__":
    sys.exit(main())
