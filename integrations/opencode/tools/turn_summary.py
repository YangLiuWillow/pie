#!/usr/bin/env python3
"""Decompose one arm's turn log into the three things wall clock confounds.

    wall  =  turns  x  (prefill + decode)

Prints per-arm totals and a per-turn table. The two columns that matter are
`ttft_s` (prefill, plus queue, plus whatever the engine does before it starts)
and `dec_tok/s` (generation). An engine that loses on `total` while matching on
both of those lost on TURN COUNT, which is an agent-behaviour result -- usually
tool-call fidelity -- and not a serving result at all.

Rows with a null TTFT are calls that produced no content: an error, or a
refusal. They are counted and shown, never dropped, because a fast arm made of
failed calls is the exact artifact this repo has been burned by.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(path: Path) -> None:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        print(f"  {path.name}: NO TURNS RECORDED — the arm made no model calls")
        return

    # Identity, not equality: two turns with the same prompt and the same
    # timings are two turns, and `r not in ok` would silently drop one.
    ok = [r for r in rows if r["status"] == 200 and r.get("ttft_s") is not None]
    ok_ids = {id(r) for r in ok}
    bad = [r for r in rows if id(r) not in ok_ids]

    print(f"\n  ── {path.name}  ({len(rows)} calls, {len(bad)} without content)")
    print(f"     {'#':>3} {'prompt':>7} {'cached':>7} {'out':>5} "
          f"{'ttft_s':>7} {'dec_s':>7} {'tok/s':>7}  finish")
    def num(v, fmt):
        return format(v, fmt) if v is not None else "-"

    for i, r in enumerate(rows, 1):
        print("     {:>3} {:>7} {:>7} {:>5} {:>7} {:>7} {:>7}  {}".format(
            i,
            num(r.get("prompt_tokens"), "d"),
            num(r.get("cached_tokens"), "d"),
            num(r.get("completion_tokens") or r.get("delta_count") or None, "d"),
            num(r.get("ttft_s"), ".2f"),
            num(r.get("decode_s"), ".2f"),
            num(r.get("decode_tok_s"), ".1f"),
            r.get("finish_reason") or ("HTTP " + str(r["status"]))))

    ttft = sum(r["ttft_s"] for r in ok)
    dec = sum(r["decode_s"] for r in ok)
    out = sum(r.get("completion_tokens") or 0 for r in ok)
    inp = sum(r.get("prompt_tokens") or 0 for r in ok)
    served = sum(r["total_s"] for r in rows)

    print(f"\n     turns              {len(rows)}"
          f"   ({len(bad)} produced no content)")
    print(f"     time in server     {served:8.1f} s")
    if served > 0:
        print(f"       of which TTFT    {ttft:8.1f} s   ({ttft / served * 100:.0f}%)")
        print(f"       of which decode  {dec:8.1f} s   ({dec / served * 100:.0f}%)")
    # Prompt tokens over TTFT is a PREFILL rate only to the extent the prompt
    # was actually prefilled. With a warm prefix cache most of it was not, so
    # `cached` above is what says whether this number means anything.
    if ttft > 0:
        print(f"     prompt tokens      {inp:8d}     -> {inp / ttft:8.0f} tok/s "
              f"over TTFT (cache-inflated; see cached column)")
    if dec > 0:
        print(f"     output tokens      {out:8d}     -> {out / dec:8.1f} tok/s during decode")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        p = Path(a)
        if p.exists():
            summarize(p)
        else:
            print(f"  {a}: MISSING — the arm wrote no turn log")
