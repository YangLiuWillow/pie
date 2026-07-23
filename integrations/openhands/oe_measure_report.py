#!/usr/bin/env python3
"""Aggregate OpenEvolve-on-Pie vs vLLM measurement logs into a time+accuracy
head-to-head table.

Usage:
    python oe_measure_report.py logs/oe_meas_*_<jobid>.out ...

Parses, per log:
  - `OE_MEASURE backend=.. example=.. seed=.. iters=.. elapsed_s=.. best_combined=..`
    (one per run; the authoritative per-run summary)
  - `Iteration N: ... completed in Xs`  -> per-iteration latency distribution
  - `PIE_TELEMETRY ... l1p=<mode>/.. l1g=<mode>/..` -> KV-reuse mode counts (Pie)

Groups by (backend, example); reports across seeds: total wallclock, per-iter
median, best_combined (best + mean), valid-diff rate, and Pie l1p/l1g reuse.
"""
import re
import sys
import statistics as st
from collections import defaultdict

MEAS = re.compile(
    r"OE_MEASURE backend=(\S+) example=(\S+) seed=(\S+) iters=(\S+) "
    r"elapsed_s=(\S+) programs=(\S+) children=(\S+) best_combined=(\S+)"
)
ITER = re.compile(r"Iteration \d+:.*completed in ([\d.]+)s")
TELE = re.compile(r"PIE_TELEMETRY .*\bl1p=(\w+)/\S+ l1g=(\w+)/")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(paths):
    # runs keyed by (backend, example) -> list of per-run dicts
    runs = defaultdict(list)
    # per-log accumulators (a log may hold several seeds); attribute iter/tele
    # times to whichever run they precede by scanning sequentially per file.
    for p in paths:
        try:
            lines = open(p, errors="replace").read().splitlines()
        except OSError as e:
            print(f"!! skip {p}: {e}", file=sys.stderr)
            continue
        cur_iters, l1p, l1g = [], defaultdict(int), defaultdict(int)
        for ln in lines:
            mi = ITER.search(ln)
            if mi:
                cur_iters.append(float(mi.group(1)))
                continue
            mt = TELE.search(ln)
            if mt:
                l1p[mt.group(1)] += 1
                l1g[mt.group(2)] += 1
                continue
            mm = MEAS.search(ln)
            if mm:
                be, ex, seed, iters, elapsed, progs, kids, best = mm.groups()
                runs[(be, ex)].append(dict(
                    seed=seed, iters=int(iters), elapsed=fnum(elapsed),
                    programs=int(progs), children=int(kids),
                    best=fnum(best), iter_lat=cur_iters,
                    l1p=dict(l1p), l1g=dict(l1g),
                ))
                cur_iters, l1p, l1g = [], defaultdict(int), defaultdict(int)
    if not runs:
        print("No OE_MEASURE lines found. Did the runs finish?")
        return

    def med(xs):
        xs = [x for x in xs if x is not None]
        return st.median(xs) if xs else float("nan")

    examples = sorted({ex for (_, ex) in runs})
    for ex in examples:
        print(f"\n================  example: {ex}  ================")
        hdr = f"{'backend':<8} {'runs':>4} {'wall_s(med)':>11} {'iter_s(med)':>11} " \
              f"{'best(max)':>9} {'best(mean)':>10} {'valid_diff%':>11} {'l1p_open/built':>15} {'l1g_reuse/gen':>14}"
        print(hdr); print("-" * len(hdr))
        for be in ("pie", "vllm"):
            rs = runs.get((be, ex))
            if not rs:
                continue
            walls = [r["elapsed"] for r in rs]
            iters_all = [x for r in rs for x in r["iter_lat"]]
            bests = [r["best"] for r in rs if r["best"] is not None]
            vdr = [100.0 * r["children"] / r["iters"] for r in rs if r["iters"]]
            l1po = sum(r["l1p"].get("opened", 0) for r in rs)
            l1pb = sum(r["l1p"].get("built", 0) for r in rs)
            l1gr = sum(r["l1g"].get("reused", 0) for r in rs)
            l1gg = sum(r["l1g"].get("generated", 0) for r in rs)
            print(f"{be:<8} {len(rs):>4} {med(walls):>11.1f} {med(iters_all):>11.1f} "
                  f"{(max(bests) if bests else float('nan')):>9.4f} "
                  f"{(st.mean(bests) if bests else float('nan')):>10.4f} "
                  f"{med(vdr):>10.1f}% {f'{l1po}/{l1pb}':>15} {f'{l1gr}/{l1gg}':>14}")
        # speed ratio
        pw = [r["elapsed"] for r in runs.get(("pie", ex), [])]
        vw = [r["elapsed"] for r in runs.get(("vllm", ex), [])]
        if pw and vw and med(vw):
            print(f"  -> Pie/vLLM wallclock ratio (median): {med(pw)/med(vw):.2f}x "
                  f"({'Pie slower' if med(pw) > med(vw) else 'Pie faster'})")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
