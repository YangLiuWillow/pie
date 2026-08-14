#!/usr/bin/env python3
"""Does the per-request prefix cache reuse KV, and does the graft change what
the model says?

Two claims, measured separately, because a prefix cache fails in two unrelated
ways and one of them is invisible:

  REUSE        report cached/prompt as a FRACTION, never a hit flag. A turn that
               resumes at a shallow cut and re-prefills most of the history
               still reports a hit; both prefix-cache defects RatioThink shipped
               did exactly that, one behind a 25.7x TTFT regression.

  AGREEMENT    a resumed turn should produce what the cold turn produced. A
               graft onto a prefix that ends in the wrong place still decodes
               fluent text, so "it answered" proves nothing.

AGREEMENT is a rate over many prompts, not one anecdote — a single mismatch
cannot distinguish a broken graft from a near-tie argmax flip, and greedy
decoding on int4 weights has plenty of near-ties. Two runs make it decidable:

  --mode agree     cold vs warm on the same prompt (graft vs full prefill)
  --mode control   cold only. Run it twice under DIFFERENT max_forward_tokens
                   and diff. That changes the prefill chunk shape and nothing
                   else — no cache involved. Whatever disagreement it produces
                   is the floor the graft should be judged against.

  python3 apc_probe.py --mode agree   --uid r1
  python3 apc_probe.py --mode control --uid r1   # under each chunk size
"""

import argparse
import json
import sys
import time
import urllib.request

BULK = "\n".join(
    f"Repository note {i}: module pkg/mod_{i}.py defines helper_{i}(x) which "
    f"returns x * {i} and is covered by tests/test_mod_{i}.py."
    for i in range(220)
)

QUESTIONS = [
    "What does helper_7 return for x=3? Answer with the number only.",
    "Which file covers helper_44? Answer with the path only.",
    "List the modules for helper_2, helper_3 and helper_4, one path per line.",
    "What does helper_15 return for x=4? Show the multiplication then the result.",
    "Name the helper defined in pkg/mod_91.py and say what it returns for x=1.",
    "Which two files would you edit to change helper_5's multiplier? Paths only.",
]


def system_for(uid, i):
    # The unique marker leads the system message so EVERY cut differs between
    # cases. Sharing a prefix would make case 2 onwards warm, and a "cold" run
    # that is actually warm measures nothing.
    return (
        f"Session {uid}-{i}. You are a precise coding assistant working in a "
        "Python repository. Answer with the shortest correct answer and no "
        "preamble.\n\n" + BULK
    )


def call(base, model, messages, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            # The gateway requires one; opencode.json sends this exact key.
            "Authorization": "Bearer pie-local",
        },
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.load(r)
    usage = payload.get("usage", {})
    return {
        "text": (payload["choices"][0]["message"].get("content") or "").strip(),
        "prompt": usage.get("prompt_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "s": time.monotonic() - t0,
    }


def first_diff(a, b):
    """Index of the first differing character, or None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3-coder-30b")
    ap.add_argument("--mode", choices=["agree", "control"], default="agree")
    ap.add_argument("--uid", default="p1", help="same uid = same prompts across runs")
    ap.add_argument("--n", type=int, default=len(QUESTIONS))
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    cases = QUESTIONS[: args.n]
    results = []
    for i, q in enumerate(cases):
        msgs = [
            {"role": "system", "content": system_for(args.uid, i)},
            {"role": "user", "content": q},
        ]
        cold = call(args.base, args.model, msgs, args.max_tokens)
        if args.mode == "control":
            print(f"case {i}  prompt {cold['prompt']:>6}  {cold['s']:6.2f}s  {cold['text']!r}")
            results.append(cold)
            continue
        warm = call(args.base, args.model, msgs, args.max_tokens)
        results.append((cold, warm))
        frac = 100.0 * warm["cached"] / warm["prompt"] if warm["prompt"] else 0.0
        same = cold["text"] == warm["text"]
        print(
            f"case {i}  prompt {warm['prompt']:>6}  reuse {frac:5.1f}%  "
            f"{cold['s']:6.2f}s -> {warm['s']:5.2f}s  "
            f"{'agree' if same else 'DIFFER'}"
        )
        if not same:
            d = first_diff(cold["text"], warm["text"])
            print(f"          first divergence at char {d}")
            print(f"          cold {cold['text']!r}")
            print(f"          warm {warm['text']!r}")

    if args.mode == "control":
        print("\ncontrol: rerun under a different max_forward_tokens and diff these lines.")
        print("a difference here is chunk shape alone — no cache, no graft.")
        return 0

    print()
    warms = [w for _, w in results]
    fracs = [100.0 * w["cached"] / w["prompt"] for w in warms if w["prompt"]]
    agree = sum(1 for c, w in results if c["text"] == w["text"])
    print(f"  reuse      median {sorted(fracs)[len(fracs) // 2]:.1f}% of prompt tokens "
          f"(min {min(fracs):.1f}%, max {max(fracs):.1f}%)")
    print(f"  agreement  {agree}/{len(results)} prompts byte-identical cold vs warm")
    cold_s = sorted(c["s"] for c, _ in results)
    warm_s = sorted(w["s"] for _, w in results)
    mid = len(results) // 2
    print(f"  latency    median {cold_s[mid]:.2f}s cold -> {warm_s[mid]:.2f}s warm")
    print("             one machine, one prompt shape: a latency ratio, not a "
          "throughput claim.")
    print()
    # Reuse is the claim this script can settle on its own. Agreement is not:
    # judging it needs the control run's floor, so report it and do not grade it.
    return 0 if fracs and min(fracs) > 50.0 else 1


if __name__ == "__main__":
    sys.exit(main())
