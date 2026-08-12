#!/usr/bin/env python3
"""Regression test for summarize.py, built from real wire captures.

The A/B's reporting layer is the one part that cannot be checked by running
the benchmark — if it is broken you discover it after the GPU time is spent
(and on the 2026-08-12 H200 run, after the pod is gone). So drive it here
against synthetic arms assembled from the checked-in qwen-code captures.

Covers: two-arm comparison, trajectory-equivalence detection in both
directions, and the single-arm mode.

Usage: python3 test_summarize.py     (exit 0 = pass)
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / "tests/inferlets/fixtures/rl_completions/wire/episode1"
SUMMARIZE = HERE / "summarize.py"


def build_arm(root: Path, name: str, tasks: dict, walls: dict, oks: dict) -> Path:
    arm = root / name
    for task, captures in tasks.items():
        logs = arm / task / "logs"
        logs.mkdir(parents=True)
        for c in captures:
            shutil.copy(c, logs / os.path.basename(c))
        (logs.parent / "wall.txt").write_text(f"{walls[task]}\n")
        (logs.parent / "check.txt").write_text(f"check_rc={0 if oks[task] else 1}\n")
    return arm


def run(*args) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(SUMMARIZE), *map(str, args)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout


def main() -> int:
    caps = sorted(glob.glob(str(FIXTURES / "openai-*.json")))
    if len(caps) < 9:
        print(f"need >=9 captures under {FIXTURES}, found {len(caps)}")
        return 2
    # A capture with real usage numbers, so token accounting is exercised.
    prompt_tokens = sum(
        (json.load(open(c)).get("response") or {}).get("usage", {}).get("prompt_tokens") or 0
        for c in caps[0:3])

    failures = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        full = {"task-a": caps[0:3], "task-b": caps[3:6], "task-c": caps[6:9]}
        short = dict(full, **{"task-c": caps[6:8]})  # one fewer call -> divergence
        walls_a = {"task-a": 12.5, "task-b": 8.0, "task-c": 20.0}
        walls_b = {"task-a": 10.0, "task-b": 7.5, "task-c": 15.0}
        all_ok = {t: True for t in full}

        a = build_arm(root, "pie-model", full, walls_a, dict(all_ok, **{"task-c": False}))
        b = build_arm(root, "vllm-model", short, walls_b, all_ok)
        same = build_arm(root, "pie-copy", full, walls_a, dict(all_ok, **{"task-c": False}))

        # 1. Divergent arms: mismatch reported, non-zero exit.
        rc, out = run(a, b)
        if rc != 1 or "trajectory MISMATCH" not in out or "task-c" not in out:
            failures.append(f"divergent arms: rc={rc}, out lacked mismatch report")
        if "wall-clock ratio" not in out or "1.246" not in out:
            failures.append("divergent arms: wall-clock ratio missing/incorrect")
        if str(prompt_tokens) not in out:
            failures.append(f"divergent arms: prompt tokens {prompt_tokens} not accounted")
        if "ok 2/3" not in out or "ok 3/3" not in out:
            failures.append("divergent arms: task success counts wrong")

        # 2. Identical arms: equivalence asserted, clean exit.
        rc, out = run(a, same)
        if rc != 0 or "trajectories equivalent" not in out:
            failures.append(f"identical arms: rc={rc}, equivalence not reported")

        # 3. Single arm: summarizes instead of refusing (the mode whose
        #    absence made the H200 pie-arm numbers unsalvageable).
        rc, out = run(a)
        if rc != 0 or "TOTAL pie-model" not in out or "wall 40.5s" not in out:
            failures.append(f"single arm: rc={rc}, summary missing")

    for f in failures:
        print(f"FAIL: {f}")
    print("PASS: summarize.py two-arm, equivalence, and single-arm modes"
          if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
