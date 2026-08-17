#!/usr/bin/env python3
"""Throughput and latency from `turnlog.py` records, aggregated so short calls
cannot lie.

## The trap this exists to avoid

A `decode_tok_s` computed per call is meaningless on an agentic workload. The
median completion in a SWE-bench run is ~63 tokens and many calls are two-token
tool acknowledgements; dividing 2 tokens by a millisecond yields 4816 tok/s, and
a mean or median over those numbers reports the JITTER OF SHORT CALLS, not the
engine's generation speed. The same artifact already appeared as `dec_tps=434`
in `bench_ab` output, on a 30B model whose memory bandwidth caps it near 174.

So throughput here is TOKEN-WEIGHTED:

    sum(completion_tokens) / sum(decode_s)

which is the rate the engine actually sustained, and is immune to short-call
division. The per-call distribution is still reported, but as LATENCY, where a
per-call figure is the thing a user feels.

## What each number means

* **TTFT** -- request sent to first content byte: prefill, queue, and whatever
  the engine does before generating. This is what an agent waits on before
  anything appears, and on long prompts it dominates.
* **call total** -- the whole request; what agent wall clock is made of.
* **decode tok/s (weighted)** -- generation speed, tokens the SERVER says it
  produced over the time it spent producing them.
* **calls/instance** -- how many round trips the agent needed. An engine that
  provokes more turns loses wall clock without being slower per call, and that
  distinction is invisible in an instance timing.

`usage_degraded` counts calls where the server refused `include_usage` and
tokens had to be delta-counted; those undercount completions, so they are
reported rather than silently mixed in.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

MIN_TOKENS_FOR_RATE = 16  # a per-call rate below this is division noise


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def summarize(tag, rows, instances):
    ok = [r for r in rows if r.get("status") == 200]
    deg = sum(1 for r in ok if r.get("usage_degraded"))
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    tot = [r["total_s"] for r in ok if r.get("total_s") is not None]
    ptok = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    ct = sum(r.get("completion_tokens") or 0 for r in ok)
    ds = sum(r.get("decode_s") or 0.0 for r in ok)
    percall = [
        r["decode_tok_s"] for r in ok
        if r.get("decode_tok_s") and (r.get("completion_tokens") or 0) >= MIN_TOKENS_FOR_RATE
    ]
    return {
        "arm": tag, "calls": len(rows), "failed": len(rows) - len(ok), "degraded": deg,
        "instances": instances,
        "calls_per_instance": len(ok) / instances if instances else float("nan"),
        "ttft_med": st.median(ttft) if ttft else float("nan"),
        "ttft_p90": pct(ttft, 90), "ttft_max": max(ttft) if ttft else float("nan"),
        "total_med": st.median(tot) if tot else float("nan"),
        "total_p90": pct(tot, 90),
        "prompt_med": st.median(ptok) if ptok else float("nan"),
        "tok_weighted": ct / ds if ds > 0 else float("nan"),
        "percall_med": st.median(percall) if percall else float("nan"),
        "percall_n": len(percall),
        "completion_total": ct, "decode_s_total": ds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    stats = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, "calls-*.jsonl"))):
        tag = os.path.basename(f)[len("calls-"):-len(".jsonl")]
        preds = os.path.join(a.out_dir, f"preds-{tag}.jsonl")
        n = sum(1 for _ in open(preds)) if os.path.exists(preds) else 0
        stats.append(summarize(tag, load(f), n))
    if not stats:
        print(f"no calls-*.jsonl in {a.out_dir}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(stats, indent=2))
        return 0

    print(f"\n{'arm':<6} {'inst':>5} {'calls':>6} {'c/inst':>7} {'fail':>5} {'degr':>5}")
    for s in stats:
        print(f"{s['arm']:<6} {s['instances']:>5} {s['calls']:>6} "
              f"{s['calls_per_instance']:>7.1f} {s['failed']:>5} {s['degraded']:>5}")

    print("\nLATENCY (seconds, per call)")
    print(f"{'arm':<6} {'TTFT med':>9} {'TTFT p90':>9} {'TTFT max':>9} "
          f"{'total med':>10} {'total p90':>10} {'prompt med':>11}")
    for s in stats:
        print(f"{s['arm']:<6} {s['ttft_med']:>9.2f} {s['ttft_p90']:>9.2f} {s['ttft_max']:>9.2f} "
              f"{s['total_med']:>10.2f} {s['total_p90']:>10.2f} {s['prompt_med']:>11.0f}")

    print("\nTHROUGHPUT")
    print(f"{'arm':<6} {'tok/s weighted':>15} {'tok/s percall med':>19} {'n>=16tok':>9} "
          f"{'tokens':>9} {'decode s':>9}")
    for s in stats:
        print(f"{s['arm']:<6} {s['tok_weighted']:>15.1f} {s['percall_med']:>19.1f} "
              f"{s['percall_n']:>9} {s['completion_total']:>9} {s['decode_s_total']:>9.0f}")
    print(f"""
  tok/s weighted = sum(completion_tokens)/sum(decode_s): the rate actually
  sustained. The per-call median covers only calls of >= {MIN_TOKENS_FOR_RATE} tokens; unrestricted
  it reaches four figures on two-token tool replies, which is division noise.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
