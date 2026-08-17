#!/usr/bin/env python3
"""Did the model DEGENERATE, or did the engine get faster? They look identical.

## Why this exists

On 2026-08-17 a pie SWE-bench arm finished ten instances in 27 minutes against
roughly ten minutes *per instance* in the graded run, and produced patches on
half of them. Read as throughput that is a spectacular result. Read as
transcripts it is four runs stuck in degenerate repetition -- one line emitted
132 times -- each producing exactly zero bytes. **A loop makes a run faster.**

Every instrument in this repo measures tokens per second, milliseconds per fire,
or wall clock. Not one of them can tell "the engine got quicker" from "the model
started repeating itself and hit the cap sooner". This closes that gap with the
cheapest possible check: the most-repeated substantial line in the agent
transcript.

## The metric

`maxrep` = how many times the most frequent line of >60 characters appears.

    1-3     healthy
    4-9     watch it; long agent runs legitimately restate things
    >=10    degenerate; in observed runs these produced 0-byte patches

Crude on purpose. It needs no tokenizer, no model, no engine, and it runs on any
transcript any engine produced, which is what makes it usable as a control
across pie, mlx-lm and vLLM alike.

## What it is NOT

It is not an accuracy metric. A clean transcript can still fail the task, and
`swebench.harness.run_evaluation` remains the only thing that says "resolved".
This answers one narrower question: is the model still generating, or is it
spinning?

## Usage

    transcript_health.py DIR [DIR2]          # one run, or two compared
    transcript_health.py --json DIR          # machine-readable
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

LOOP_THRESHOLD = 10
WATCH_THRESHOLD = 4
MIN_LINE = 60


def health(path):
    """(maxrep, total_lines, the repeated line) for one transcript."""
    try:
        with open(path, errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except OSError:
        return None
    subst = [ln for ln in lines if len(ln) > MIN_LINE]
    if not subst:
        return (0, len(lines), "")
    line, n = Counter(subst).most_common(1)[0]
    return (n, len(lines), line)


def scan(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.opencode.log"))):
        inst = os.path.basename(f)[: -len(".opencode.log")]
        h = health(f)
        if h:
            out[inst] = h
    return out


def verdict(n):
    if n >= LOOP_THRESHOLD:
        return "LOOP"
    if n >= WATCH_THRESHOLD:
        return "watch"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories of *.opencode.log")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runs = [(d, scan(d)) for d in args.dirs]
    for d, r in runs:
        if not r:
            print(f"no transcripts in {d}", file=sys.stderr)

    if args.json:
        print(json.dumps({d: {k: {"maxrep": v[0], "lines": v[1]} for k, v in r.items()}
                          for d, r in runs}, indent=2))
        return 0

    if len(runs) == 1:
        d, r = runs[0]
        print(f"\n{os.path.basename(d)}\n")
        print(f"{'instance':<28} {'maxrep':>7} {'lines':>7}   verdict")
        for inst in sorted(r):
            n, lines, _ = r[inst]
            print(f"{inst:<28} {n:>7} {lines:>7}   {verdict(n)}")
        loops = sum(1 for v in r.values() if v[0] >= LOOP_THRESHOLD)
        print(f"\n  looping: {loops}/{len(r)}")
        if loops:
            worst = max(r.items(), key=lambda kv: kv[1][0])
            print(f"  worst:   {worst[0]} at {worst[1][0]}x")
            print(f"    {worst[1][2][:110]}...")
        return 0

    # Two runs: the comparison is the point -- a maxrep that TRIPLES between
    # runs of the same engine is the signal, not its absolute value.
    (da, ra), (db, rb) = runs[0], runs[1]
    print(f"\nA = {os.path.basename(da)}\nB = {os.path.basename(db)}\n")
    print(f"{'instance':<28} {'A':>6} {'B':>6}   change")
    for inst in sorted(set(ra) | set(rb)):
        a = ra.get(inst, (None,))[0]
        b = rb.get(inst, (None,))[0]
        if a is None or b is None:
            note = "only in one run"
        elif b >= LOOP_THRESHOLD > a:
            note = "REGRESSED into a loop"
        elif a >= LOOP_THRESHOLD > b:
            note = "recovered"
        elif a and b >= a * 2 and b >= WATCH_THRESHOLD:
            note = f"worse ({b / a:.1f}x)"
        else:
            note = ""
        print(f"{inst:<28} {str(a):>6} {str(b):>6}   {note}")
    la = sum(1 for v in ra.values() if v[0] >= LOOP_THRESHOLD)
    lb = sum(1 for v in rb.values() if v[0] >= LOOP_THRESHOLD)
    print(f"\n  looping: A {la}/{len(ra)}   B {lb}/{len(rb)}")
    print("""
  A loop rate that matches between two DIFFERENT engines is a property of the
  checkpoint, not of either engine -- measured 2026-08-17, pie 4/10 against
  mlx-lm 3/10 on the same instances, overlapping on only one.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
