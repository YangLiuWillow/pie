#!/usr/bin/env python3
"""Summarize [pie-qwen35-moe-profile] lines from a pie serve log.

The driver emits one line per forward when PIE_QWEN35_MOE_PROFILE=1, with a
per-stage cudaEventSynchronize around each stage. That sync SERIALIZES the
pipeline, so absolute ms here are inflated and must not be quoted as a rate --
use the shares. Measure rates in a separate unprofiled run.

Decode and prefill forwards are summarized separately (decode=1 vs 0): they have
completely different shapes and averaging them together is meaningless.

Usage: parse_moe_profile.py <pie_serve.log> [--min-seq N]
"""
from __future__ import annotations

import re
import statistics as st
import sys

FIELD = re.compile(r"(\w+)=([-\d.eE+]+)")

# Stages that partition the forward. moe_routed is the parent of route_setup /
# gate_up / act / down / reduce, so it is listed separately from its children.
TOP = ["embed_ms", "norm_ms", "full_attn_ms", "moe_router_ms", "moe_routed_ms",
       "moe_shared_ms", "residual_ms", "lm_head_ms", "other_ms"]
ROUTED_CHILDREN = ["moe_route_setup_ms", "moe_gate_up_ms", "moe_act_ms",
                   "moe_down_ms", "moe_reduce_ms"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    path = sys.argv[1]
    min_seq = 0
    if "--min-seq" in sys.argv:
        min_seq = int(sys.argv[sys.argv.index("--min-seq") + 1])

    rows = []
    for line in open(path, errors="replace"):
        if "pie-qwen35-moe-profile" not in line:
            continue
        d = {k: float(v) for k, v in FIELD.findall(line)}
        if d.get("seq", 0) < min_seq:
            continue
        rows.append(d)

    if not rows:
        print(f"no profile lines in {path}")
        raise SystemExit(1)

    for is_decode, label in ((1.0, "DECODE"), (0.0, "PREFILL")):
        sub = [d for d in rows if d.get("decode") == is_decode]
        if not sub:
            continue
        tot = [d["total_ms"] for d in sub]
        print(f"\n=== {label}: {len(sub)} forwards   "
              f"N median={st.median(d['N'] for d in sub):.0f} "
              f"R median={st.median(d['R'] for d in sub):.0f}   "
              f"total_ms median={st.median(tot):.3f} mean={st.mean(tot):.3f}")
        print(f"    {'stage':<22} {'mean_ms':>9} {'share':>7}")
        mean_total = st.mean(tot)
        for k in TOP:
            vals = [d.get(k, 0.0) for d in sub]
            m = st.mean(vals)
            if m <= 0:
                continue
            print(f"    {k:<22} {m:>9.3f} {100 * m / mean_total:>6.1f}%")
        print(f"    {'  -- routed breakdown':<22}")
        for k in ROUTED_CHILDREN:
            vals = [d.get(k, 0.0) for d in sub]
            m = st.mean(vals)
            if m <= 0:
                continue
            print(f"    {k:<22} {m:>9.3f} {100 * m / mean_total:>6.1f}%")


if __name__ == "__main__":
    main()
