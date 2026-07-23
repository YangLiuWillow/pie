"""Best-of-K helper for fork test-time-scaling predictions.

A fork run (``run_swe_bench.py --backend pie-agent --num-branches K ...``) writes
ONE predictions.jsonl per instance whose ``_metadata.candidates`` holds the K
per-branch patches. SWE-bench scores one patch per instance, so to grade all K:

  1. ``split`` — expand predictions.jsonl into K files
     ``<stem>.cand0.jsonl`` .. ``<stem>.cand{K-1}.jsonl``, each a normal
     predictions file (real instance_ids, one branch's model_patch). Score each
     with the usual scorer → K report.json files.
  2. ``combine`` — union the K reports: an instance is best-of-K resolved if ANY
     branch's patch resolved it. Prints best-of-K vs per-branch resolve rates.

Usage:
  python bestofk.py split   predictions/foo.jsonl
  python bestofk.py combine predictions/foo.cand0.report.json predictions/foo.cand1.report.json ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _rows(path: Path):
    for line in path.open():
        line = line.strip()
        if line:
            yield json.loads(line)


def split(pred_path: Path) -> list[Path]:
    rows = list(_rows(pred_path))
    # Max candidate count across instances (single-traj rows count as 1).
    max_k = 1
    for r in rows:
        cands = (r.get("_metadata") or {}).get("candidates") or []
        max_k = max(max_k, len(cands) or 1)

    stem = pred_path.with_suffix("")  # drop .jsonl
    out_paths: list[Path] = []
    for k in range(max_k):
        out = Path(f"{stem}.cand{k}.jsonl")
        n = 0
        with out.open("w") as f:
            for r in rows:
                meta = r.get("_metadata") or {}
                cands = meta.get("candidates") or []
                if cands:
                    if k >= len(cands):
                        continue  # this instance has fewer than k+1 candidates
                    patch = cands[k].get("model_patch", "")
                else:
                    # single-trajectory row: only present in cand0
                    if k > 0:
                        continue
                    patch = r.get("model_patch", "")
                f.write(json.dumps({
                    "instance_id": r["instance_id"],
                    "model_name_or_path": r.get("model_name_or_path", "pie-agent-fork"),
                    "model_patch": patch,
                }) + "\n")
                n += 1
        print(f"wrote {out}  ({n} rows)")
        out_paths.append(out)
    return out_paths


def _resolved_ids(report_path: Path) -> set[str]:
    d = json.loads(Path(report_path).read_text())
    return set(d.get("resolved_ids", []))


def combine(report_paths: list[Path]) -> None:
    per_branch = [(_resolved_ids(p), p) for p in report_paths]
    union: set[str] = set()
    for ids, _ in per_branch:
        union |= ids

    print("=== per-branch resolved ===")
    for ids, p in per_branch:
        print(f"  {Path(p).name}: {len(ids)} resolved")
    print(f"\n=== best-of-{len(report_paths)} resolved: {len(union)} ===")
    print("  ids:", sorted(union))

    # Which instances each branch uniquely rescued (only that branch solved).
    for ids, p in per_branch:
        others = set().union(*[o for o, q in per_branch if q != p]) if len(per_branch) > 1 else set()
        only = ids - others
        if only:
            print(f"  only {Path(p).name} solved: {sorted(only)}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    mode = argv[0]
    if mode == "split":
        split(Path(argv[1]))
    elif mode == "combine":
        combine([Path(a) for a in argv[1:]])
    else:
        print(f"unknown mode {mode!r} (expected split|combine)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
