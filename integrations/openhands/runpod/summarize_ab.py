#!/usr/bin/env python3
# =============================================================================
# Summarize A/B prediction jsonls into the writeup's headline numbers, the
# HONEST way: normalize by agent iteration and by per-call latency (both are
# recorded by the SAME harness for every arm, so the comparison is apples-to-
# apples regardless of trajectory divergence).
#
# Each prediction line carries _metadata: wall_clock_s, agent_iterations,
# num_llm_calls, prompt_tokens, completion_tokens, response_latencies[].
#
# Usage:
#   python summarize_ab.py pie=predictions/ab_a100_pie_*.jsonl \
#                          litellm-crippled=predictions/ab_a100_litellm_crippled_*.jsonl \
#                          litellm-graphs-only=predictions/..._graphs-only_*.jsonl \
#                          litellm-fair=predictions/..._fair_*.jsonl
# Any number of arms; each arg is  LABEL=GLOB.  Shared instances are used for the
# head-to-head totals so the arms are compared on the same work.
# =============================================================================
from __future__ import annotations

import glob
import json
import statistics
import sys


def load(path_glob):
    rows = {}
    for path in sorted(glob.glob(path_glob)):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = d.get("_metadata", {}) or {}
            rows[d["instance_id"]] = m  # last wins (resume-safe)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def main():
    args = sys.argv[1:]
    if not args or any("=" not in a for a in args):
        print(__doc__)
        sys.exit(1)

    arms = {}
    for a in args:
        label, g = a.split("=", 1)
        arms[label] = load(g)
        if not arms[label]:
            print(f"WARN: no rows matched for {label} ({g})")

    # Head-to-head on the instances every arm resolved (fair shared set).
    shared = set.intersection(*[set(r) for r in arms.values()]) if arms else set()
    print(f"# arms: {', '.join(arms)}")
    print(f"# shared instances (all arms): {len(shared)}")
    print()

    hdr = f"{'arm':<26} {'wall_s':>9} {'iters':>7} {'s/iter':>7} {'med_lat':>8} {'calls':>7} {'prompt_tok':>12} {'compl_tok':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, rows in arms.items():
        sub = {k: rows[k] for k in shared} if shared else rows
        wall = sum(m.get("wall_clock_s", 0) or 0 for m in sub.values())
        iters = sum(m.get("agent_iterations", 0) or 0 for m in sub.values())
        calls = sum(m.get("num_llm_calls", 0) or 0 for m in sub.values())
        ptok = sum(m.get("prompt_tokens", 0) or 0 for m in sub.values())
        ctok = sum(m.get("completion_tokens", 0) or 0 for m in sub.values())
        lat = [x for m in sub.values() for x in (m.get("response_latencies") or [])]
        s_iter = wall / iters if iters else float("nan")
        print(f"{label:<26} {wall:>9.1f} {iters:>7} {s_iter:>7.3f} {med(lat):>8.3f} "
              f"{calls:>7} {ptok:>12,} {ctok:>10,}")

    print()
    print("Read s/iter and med_lat, NOT raw wall_s: trajectories diverge, so wall")
    print("time conflates path length with serving speed. s/iter and per-call")
    print("median latency are the trajectory-robust serving-layer numbers.")

    # Per-instance wall for the shared set (spot divergence).
    if shared and len(arms) >= 2:
        print()
        labels = list(arms)
        print("per-instance wall_clock_s (shared set):")
        print(f"  {'instance':<40} " + " ".join(f"{l[:14]:>14}" for l in labels))
        for inst in sorted(shared):
            cells = " ".join(f"{arms[l][inst].get('wall_clock_s',0):>14.1f}" for l in labels)
            print(f"  {inst:<40} {cells}")


if __name__ == "__main__":
    main()
