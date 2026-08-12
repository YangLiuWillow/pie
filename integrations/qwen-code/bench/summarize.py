#!/usr/bin/env python3
"""Summarize and compare two benchmark arms (pie vs vLLM) for qwen-code.

Reads the per-task result dirs written by run_arm.sh and the qwen-code
`--openai-logging` captures inside them. Reports, per task and in total:
wall-clock, request count, prompt/completion tokens, cached (KV-reused)
prompt tokens, task success — and a trajectory-equivalence check in the
`compare_equivalence` discipline: at temperature 0 the two arms should
make the same tool calls in the same order; the serving stack changes
where prefill FLOPs happen, not what the model does.

Usage:
    python3 summarize.py results/pie-qwen3-coder results/vllm-Qwen_Qwen3-Coder-30B-A3B-Instruct
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_task(tdir: Path) -> dict:
    captures = sorted(tdir.glob("logs/openai-*.json")) or sorted(tdir.glob("logs/*.json"))
    reqs = prompt = completion = cached = 0
    calls: list[str] = []
    for p in captures:
        d = json.loads(p.read_text())
        resp = d.get("response") or {}
        u = resp.get("usage") or {}
        reqs += 1
        prompt += u.get("prompt_tokens") or 0
        completion += u.get("completion_tokens") or 0
        cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        for ch in resp.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append(fn.get("name", "?"))
    return {
        "wall": float((tdir / "wall.txt").read_text().strip()) if (tdir / "wall.txt").exists() else None,
        "ok": "check_rc=0" in ((tdir / "check.txt").read_text() if (tdir / "check.txt").exists() else ""),
        "reqs": reqs,
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "calls": calls,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a_name, b_name = a_dir.name, b_dir.name
    tasks = sorted(
        {p.name for p in a_dir.iterdir() if p.is_dir()}
        & {p.name for p in b_dir.iterdir() if p.is_dir()}
    )
    if not tasks:
        print("no common tasks between the two result dirs")
        return 2

    hdr = f"{'task':<16} {'arm':<6} {'ok':<4} {'wall_s':>7} {'reqs':>4} {'prompt':>8} {'cached':>8} {'compl':>6}  tool calls"
    print(hdr)
    print("-" * len(hdr))
    tot: dict[str, dict] = {a_name: dict(wall=0.0, ok=0, prompt=0, cached=0, compl=0),
                            b_name: dict(wall=0.0, ok=0, prompt=0, cached=0, compl=0)}
    mismatches = []
    for t in tasks:
        rows = {a_name: read_task(a_dir / t), b_name: read_task(b_dir / t)}
        for name, r in rows.items():
            print(f"{t:<16} {name.split('-')[0]:<6} {'y' if r['ok'] else 'N':<4} "
                  f"{r['wall'] if r['wall'] is not None else -1:>7.1f} {r['reqs']:>4} "
                  f"{r['prompt']:>8} {r['cached']:>8} {r['completion'] if 'completion' in r else r['compl']:>6}  "
                  f"{'>'.join(r['calls']) or '-'}")
            tt = tot[name]
            tt["wall"] += r["wall"] or 0
            tt["ok"] += r["ok"]
            tt["prompt"] += r["prompt"]
            tt["cached"] += r["cached"]
            tt["compl"] += r["completion"]
        if rows[a_name]["calls"] != rows[b_name]["calls"]:
            mismatches.append(t)

    print()
    for name, tt in tot.items():
        reuse = 100.0 * tt["cached"] / tt["prompt"] if tt["prompt"] else 0.0
        print(f"TOTAL {name}: ok {tt['ok']}/{len(tasks)}, wall {tt['wall']:.1f}s, "
              f"prompt {tt['prompt']}, cached {tt['cached']} ({reuse:.1f}% reuse), "
              f"completion {tt['compl']}")
    wa, wb = tot[a_name]["wall"], tot[b_name]["wall"]
    if wa and wb:
        print(f"\nwall-clock ratio {a_name} / {b_name}: {wa / wb:.3f}")
    if mismatches:
        print(f"\ntrajectory MISMATCH (tool-call sequences differ) on: {mismatches}")
        print("speed numbers on mismatched tasks compare different work — read with care")
        return 1
    print("\ntrajectories equivalent on all tasks (same tool-call sequences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
