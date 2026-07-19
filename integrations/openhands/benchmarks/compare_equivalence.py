"""Compare two SWE-Bench predictions JSONL files for trajectory equivalence.

Phase 2 of the openhands-coder-session design: at temperature 0 the stateless
(`openhands-completion`) and session (`openhands-coder-session --kv-verify`)
arms should produce near-identical trajectories — the session changes *where
prefill FLOPs happen*, not any token the model sees. This script checks that
claim instance by instance and summarizes the prefill savings the session arm
reported.

Usage:
    python -m benchmarks.compare_equivalence stateless.jsonl session.jsonl

Exit codes: 0 = all instances equivalent, 1 = at least one mismatch,
2 = a kv-verify (or other) error was recorded in either arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["instance_id"]] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path, help="stateless-arm predictions JSONL")
    p.add_argument("session", type=Path, help="session-arm predictions JSONL")
    args = p.parse_args(argv)

    a, b = load(args.baseline), load(args.session)
    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        print(f"only in {args.baseline.name}: {only_a}")
    if only_b:
        print(f"only in {args.session.name}: {only_b}")

    mismatches = 0
    errors = 0
    total_rendered = 0
    total_prefilled = 0

    for iid in shared:
        ra, rb = a[iid], b[iid]
        ma, mb = ra.get("_metadata", {}), rb.get("_metadata", {})

        for label, m in ((args.baseline.name, ma), (args.session.name, mb)):
            if m.get("error"):
                errors += 1
                print(f"{iid}: ERROR in {label}: {m['error']}")

        same_patch = ra.get("model_patch") == rb.get("model_patch")
        fields = {}
        for key in ("agent_iterations", "num_llm_calls", "completion_tokens"):
            va, vb = ma.get(key), mb.get(key)
            if va != vb:
                fields[key] = (va, vb)

        sess = mb.get("pie_session") or {}
        rendered = sess.get("prompt_tokens_rendered", 0)
        prefilled = sess.get("prompt_tokens_prefilled", 0)
        total_rendered += rendered
        total_prefilled += prefilled
        savings = (
            f"prefill {prefilled}/{rendered} tokens "
            f"({rendered / prefilled:.1f}x less)" if prefilled else "no session stats"
        )
        modes = sess.get("modes", {})

        if same_patch and not fields:
            print(f"{iid}: EQUIVALENT — {savings} modes={modes}")
        else:
            mismatches += 1
            print(f"{iid}: MISMATCH — patch_equal={same_patch} "
                  f"diverging_fields={fields} {savings} modes={modes}")

    print()
    print(f"{len(shared)} instances compared, "
          f"{len(shared) - mismatches} equivalent, {mismatches} mismatched, "
          f"{errors} errored")
    if total_prefilled:
        print(f"session arm total prompt prefill: {total_prefilled} of "
              f"{total_rendered} rendered tokens "
              f"({total_rendered / total_prefilled:.2f}x reduction)")

    if errors:
        return 2
    return 1 if mismatches or only_a or only_b else 0


if __name__ == "__main__":
    sys.exit(main())
