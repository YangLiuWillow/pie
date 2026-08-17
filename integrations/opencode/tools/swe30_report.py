#!/usr/bin/env python3
"""One table: accuracy, throughput, latency, turns, degeneration — three engines.

Merges the four instruments that each answer part of the question, because no
single one of them can:

* `<arm>.<run_tag>_<arm>.json`  — graded `resolved_ids` from the SWE-bench
  Docker harness. The ONLY thing that says "accuracy". Patch bytes do not: a
  previous run had 2/5 non-empty patches in both arms while grading 4/5 to 1/5.
* `calls-<arm>.jsonl`          — per-call TTFT, decode wall and server-reported
  token counts from `turnlog.py`, identical across engines.
* `wd-<arm>/*.opencode.log`    — transcripts, for degeneration. A loop makes a
  run FASTER, so every timing here would score one as a win without this column.
* `preds-<arm>.jsonl`          — which instances the arm actually attempted.

## Two aggregation choices that are not cosmetic

**Throughput is token-weighted**, `sum(completion_tokens)/sum(decode_s)`. A
per-call rate on an agentic workload is dominated by two-token tool replies and
reaches four figures on a model whose bandwidth caps it near 174.

**Latency stays per-call**, because there the per-call figure is what a user
waits through. TTFT and total are reported as median and p90 — an agentic
workload mixes small tool round-trips with 29k-token prefills, so a mean
describes neither.

## What this refuses to do

It does not rank engines that are within a couple of instances of each other.
At n=30 a two-instance swing reorders the table, and the previous 10-instance
run had all three tied exactly. Where the spread is small the summary says
"parity", not a leaderboard.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

MIN_TOKENS_FOR_RATE = 16
LOOP_THRESHOLD = 10
ARMS = ("pie", "mlx", "vllm")


def graded(out, run_tag, arm):
    exact = os.path.join(out, f"{arm}.{run_tag}_{arm}.json")
    cands = [exact] if os.path.exists(exact) else sorted(
        glob.glob(os.path.join(out, f"{arm}.*.json")))
    for f in cands:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "resolved_ids" in d:          # identify by schema, not by filename
            return d
    return None


def calls(out, arm):
    p = os.path.join(out, f"calls-{arm}.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def loops(out, arm):
    d = os.path.join(out, f"wd-{arm}")
    if not os.path.isdir(d):
        return None
    n = tot = 0
    for f in glob.glob(os.path.join(d, "*.opencode.log")):
        try:
            lines = [l.rstrip("\n") for l in open(f, errors="replace")]
        except OSError:
            continue
        subst = [l for l in lines if len(l) > 60]
        tot += 1
        if subst:
            from collections import Counter
            if Counter(subst).most_common(1)[0][1] >= LOOP_THRESHOLD:
                n += 1
    return (n, tot)


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--run-tag", default="swe30")
    a = ap.parse_args()
    out = a.out_dir

    rows = []
    for arm in ARMS:
        g = graded(out, a.run_tag, arm)
        c = [r for r in calls(out, arm) if r.get("status") == 200]
        lp = loops(out, arm)
        preds = os.path.join(out, f"preds-{arm}.jsonl")
        n_inst = sum(1 for _ in open(preds)) if os.path.exists(preds) else 0
        ct = sum(r.get("completion_tokens") or 0 for r in c)
        ds = sum(r.get("decode_s") or 0.0 for r in c)
        ttft = [r["ttft_s"] for r in c if r.get("ttft_s") is not None]
        tot = [r["total_s"] for r in c if r.get("total_s") is not None]
        rows.append(dict(
            arm=arm, instances=n_inst,
            resolved=len(g["resolved_ids"]) if g else None,
            submitted=(g.get("submitted_instances") or len(g.get("submitted_ids") or [])) if g else None,
            resolved_ids=set(g["resolved_ids"]) if g else set(),
            calls=len(c), cpi=(len(c) / n_inst if n_inst else float("nan")),
            tokw=(ct / ds if ds else float("nan")),
            ttft_med=st.median(ttft) if ttft else float("nan"),
            ttft_p90=pct(ttft, 90),
            tot_med=st.median(tot) if tot else float("nan"),
            tot_p90=pct(tot, 90),
            degraded=sum(1 for r in c if r.get("usage_degraded")),
            loops=lp,
        ))

    print("\n=== ACCURACY (graded) ===")
    print(f"{'arm':<6} {'resolved':>9} {'submitted':>10}")
    for r in rows:
        res = "not graded" if r["resolved"] is None else f"{r['resolved']}"
        print(f"{r['arm']:<6} {res:>9} {str(r['submitted'] or r['instances']):>10}")
    got = [r for r in rows if r["resolved"] is not None]
    if len(got) > 1:
        spread = max(x["resolved"] for x in got) - min(x["resolved"] for x in got)
        n = max(x["submitted"] or x["instances"] or 1 for x in got)
        if spread <= 2:
            print(f"\n  spread is {spread} instance(s) over n={n}. That is PARITY, not a")
            print("  ranking -- a two-instance swing reorders this table.")
        else:
            print(f"\n  spread {spread} of {n}.")

    print("\n=== THROUGHPUT ===")
    print(f"{'arm':<6} {'tok/s (weighted)':>17} {'calls':>7} {'calls/inst':>11} {'usage degraded':>15}")
    for r in rows:
        print(f"{r['arm']:<6} {r['tokw']:>17.1f} {r['calls']:>7} {r['cpi']:>11.1f} {r['degraded']:>15}")

    print("\n=== LATENCY (seconds per call) ===")
    print(f"{'arm':<6} {'TTFT med':>9} {'TTFT p90':>9} {'call med':>9} {'call p90':>9}")
    for r in rows:
        print(f"{r['arm']:<6} {r['ttft_med']:>9.2f} {r['ttft_p90']:>9.2f} "
              f"{r['tot_med']:>9.2f} {r['tot_p90']:>9.2f}")

    print("\n=== DEGENERATION (loops; a loop makes a run FASTER) ===")
    print(f"{'arm':<6} {'looped':>8} {'of':>5} {'rate':>7}")
    for r in rows:
        if r["loops"]:
            n, t = r["loops"]
            print(f"{r['arm']:<6} {n:>8} {t:>5} {(n/t*100 if t else 0):>6.0f}%")
        else:
            print(f"{r['arm']:<6} {'-':>8} {'-':>5} {'-':>7}")

    if len(got) > 1:
        print("\n=== WHICH INSTANCES (graded) ===")
        allres = set().union(*[r["resolved_ids"] for r in got])
        for i in sorted(allres):
            who = " ".join(r["arm"] for r in got if i in r["resolved_ids"])
            print(f"  {i:<34} {who}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
