#!/usr/bin/env python3
"""A/B one arm of the opencode integration: replay a growing agentic
conversation and time every turn.

## What this measures, and why it is a replay rather than stock opencode

The claim under test is that ~70% of an agentic task is re-prefill of a history
that changed by a few hundred tokens, and that a resumed KV working set removes
it. Testing that needs the two arms to see **byte-identical prompts** — the
handover is emphatic about this, because qwen-code's run 2 compared two arms
that turned out to differ in their renderers rather than their servers, and
measured nothing.

Driving stock opencode twice cannot give that: turn N+1's request contains turn
N's assistant reply, so the moment the two arms answer differently — a token of
sampling noise is enough — every later turn is comparing different prompts.

So this replays a **canned** transcript. Every turn's request is fixed in
advance from the real captured opencode wire fixtures
(`tests/inferlets/fixtures/opencode/wire/`, a genuine `opencode/1.18.16`
session: 10 tools, ~7.5k-token system+user turn, then an assistant tool call and
its result). The model's own output is timed and then discarded; the next turn
appends the canned assistant turn instead. Both arms therefore see exactly the
same bytes at every turn, and the only difference is where the KV lives.

That the canned reply is not the model's own does not favour either arm. Under
Strategy A nothing is retained at all. Under Strategy B the retention address
hashes only the CLIENT's messages — no server output — so a canned assistant
turn resumes exactly as a real one would. (Under the earlier seal-through-the-
generated-turn design it would not have, which is one more reason that design
was wrong.)

## What is reported

Per turn: wall time, time to first *content* byte, and the usage the server
reported. TTFB is deliberately not "time to first chunk": the role chunk lands
in ~3 ms on both arms and says nothing — the 2026-08-12 run recorded a 0.003 s
TTFB against a real 12.4 s wait for first content.

Usage:
    PIE_BASE_URL=http://127.0.0.1:8080 python3 bench_ab.py --arm b --turns 4
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = REPO / "tests/inferlets/fixtures/opencode/wire"


def load_fixture(name):
    d = json.loads((FIXTURES / name).read_text())
    body = d["body"]
    return json.loads(body) if isinstance(body, str) else body


def build_conversation(turns):
    """The canned transcript, as a list of message-lists (one per turn).

    Turn 1 and 2 are the real capture: the opening agent turn, then the same
    conversation after one tool round-trip. Further turns extend it in the same
    shape — assistant tool call, tool result — so history grows the way a real
    agent loop grows, a few hundred tokens at a time.
    """
    first = load_fixture("req-004.json")   # system + user, 10 tools
    second = load_fixture("req-005.json")  # + assistant(tool_call) + tool result

    base_msgs = second["messages"]
    assert [m["role"] for m in base_msgs] == ["system", "user", "assistant", "tool"], (
        f"req-005 shape changed: {[m['role'] for m in base_msgs]}"
    )
    canned_assistant, canned_tool = base_msgs[2], base_msgs[3]

    seqs = [first["messages"], base_msgs]
    while len(seqs) < turns:
        prev = seqs[-1]
        k = len(seqs)
        # A fresh call id per round, so nothing dedups and the ids stay valid.
        a = json.loads(json.dumps(canned_assistant))
        t = json.loads(json.dumps(canned_tool))
        cid = f"call_bench_{k}"
        if a.get("tool_calls"):
            a["tool_calls"] = [dict(a["tool_calls"][0], id=cid)]
        t["tool_call_id"] = cid
        seqs.append(prev + [a, t])
    return seqs[:turns], second.get("tools", first.get("tools", []))


def stream_turn(base_url, body, timeout):
    """POST one streaming turn. Returns (total_s, ttfc_s, usage, text_len)."""
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer bench"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttfc = None
    usage = None
    text = []
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
                delta = ch.get("delta", {})
                # First CONTENT byte, not the role chunk.
                if delta.get("content"):
                    if ttfc is None:
                        ttfc = time.perf_counter() - t0
                    text.append(delta["content"])
                if delta.get("tool_calls") and ttfc is None:
                    ttfc = time.perf_counter() - t0
    return time.perf_counter() - t0, ttfc, usage, sum(len(s) for s in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--arm", required=True, help="label for this run (a|b|...)")
    ap.add_argument("--model", default=os.environ.get("PIE_MODEL", "pie"))
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96,
                    help="capped so the comparison is dominated by prefill, which is "
                         "what the strategies differ in; decode rate is already known")
    ap.add_argument("--decode-probe", action="store_true",
                    help="Append a request for a long answer and drop the tool "
                         "definitions, so the turn is dominated by DECODE rather "
                         "than prefill. Without this the canned transcript ends in "
                         "a tool result and the model replies with a ~10-token tool "
                         "call — which is the right shape for comparing prefill "
                         "reuse and the wrong one for anything about decoding, "
                         "speculative or otherwise. The appended text is identical "
                         "on every arm, so it cannot favour one.")
    ap.add_argument("--decode-tail", default=None,
                    help="Override the --decode-probe request. Use to set the "
                         "DRAFTABILITY of the output deliberately: an answer that "
                         "copies text already in the context is the best case for "
                         "prompt-lookup drafting, and free-form prose is the "
                         "worst. Reporting only one of them describes a workload, "
                         "not an engine.")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs, tools = build_conversation(args.turns)

    # A long answer that an agent turn would plausibly ask for, and that leans on
    # text already in the context (file paths, identifiers) — which is exactly
    # the condition prompt-lookup drafting needs. Making it easy to draft is
    # deliberate: a speculation comparison on text nothing can draft measures the
    # fallback path on both sides and reports a tie that means nothing.
    DECODE_TAIL = (
        "Now, without calling any tools, write out a numbered plan for this task. "
        "For each step give the file path it touches and one sentence on why. "
        "Then repeat the same list as a checklist. Be thorough and specific."
    )
    if args.decode_probe:
        tail = args.decode_tail or DECODE_TAIL
        seqs = [m + [{"role": "user", "content": tail}] for m in seqs]

    def body_for(msgs):
        return {
            "model": args.model,
            "messages": msgs,
            # Tools steer the model into a short tool call. The decode probe
            # wants prose, so it asks for the same turn without them.
            **({} if args.decode_probe else {"tools": tools}),
            "max_tokens": args.max_tokens,
            # Greedy: with identical prompts the two arms should generate the
            # same tokens, which makes completion_tokens a cross-check rather
            # than a confound.
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    # Warm-up: the first request after a boot pays wasm JIT / kernel compile and
    # reads as a hang. Timing it would put a one-off cost into turn 1 of
    # whichever arm ran first.
    #
    # It MUST NOT share a prefix with any measured turn. Sending turn 1's own
    # messages here primes a prefix cache, which silently converts turn 1 from a
    # cold prefill into a cache hit — and only on the arms that HAVE a cache, so
    # it favours them. Measured: it took vLLM's turn-1 TTFC on a 7.5k-token
    # prompt to 0.34 s, an implausible ~22k tok/s prefill, while pie's turn 1
    # stayed cold because pie excludes its own final boundary as a resume
    # candidate. Two arms, two different meanings for "turn 1".
    print(f"[{args.arm}] warming up (unrelated short prompt)...", flush=True)
    warm = {
        "model": args.model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        stream_turn(args.base_url, warm, args.timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[{args.arm}] warm-up failed: {e!r}", file=sys.stderr)
        return 1

    rows = []
    total = 0.0
    print(f"[{args.arm}] {args.turns} turns against {args.base_url}", flush=True)
    for i, msgs in enumerate(seqs, 1):
        try:
            secs, ttfc, usage, nchars = stream_turn(args.base_url, body_for(msgs), args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[{args.arm}] turn {i} FAILED: {e!r}", file=sys.stderr)
            return 1
        u = usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        gen = u.get("completion_tokens", 0)
        # Decode rate over the tokens AFTER the first: everything before first
        # content is prefill, and charging it to decode would make a fast prefill
        # look like fast decoding. Undefined below two tokens rather than
        # reported as a number that is really a prefill measurement.
        dec_tps = None
        if ttfc is not None and gen > 1 and secs > ttfc:
            dec_tps = round((gen - 1) / (secs - ttfc), 1)
        row = {
            "turn": i,
            "messages": len(msgs),
            "seconds": round(secs, 3),
            "ttfc": round(ttfc, 3) if ttfc is not None else None,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "cached_tokens": cached,
            "completion_tokens": gen,
            "decode_tps": dec_tps,
            "content_chars": nchars,
        }
        rows.append(row)
        total += secs
        print(
            f"  turn {i}: {secs:7.2f}s  ttfc={row['ttfc']}  "
            f"prompt={row['prompt_tokens']} cached={cached} "
            f"gen={gen} dec_tps={dec_tps}",
            flush=True,
        )

    print(f"[{args.arm}] TOTAL {total:.2f}s over {args.turns} turns", flush=True)
    result = {
        "arm": args.arm,
        "base_url": args.base_url,
        "model": args.model,
        "turns": args.turns,
        "max_tokens": args.max_tokens,
        "total_seconds": round(total, 3),
        "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[{args.arm}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
