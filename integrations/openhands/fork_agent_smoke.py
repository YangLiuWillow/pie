"""GPU smoke for Step 2 — fork-based test-time scaling in the openhands-agent
inferlet.

Runs ONE SWE-bench instance through the Pattern A agent with num_branches>1:
the agent runs a trunk trajectory to ``branch_at_step``, then forks the KV
context AND the tool-server workspace K ways and finishes each branch
independently. We validate:

  1. the inferlet returns a ``branches`` array of length num_branches,
  2. each branch ran on its own ``workspace_id`` ("0", "1", ...),
  3. each branch's diff comes from its OWN workspace dir (isolation),
  4. (informational) whether the branches diverged (different diffs).

Requires a running ``pie serve`` with the openhands-agent inferlet installed
(the accompanying sbatch boots both). Assumes temperature > 0 so forked
contexts actually diverge (greedy forks generate identically).

Env: INSTANCE_ID, PIE_PORT, NUM_BRANCHES, BRANCH_AT_STEP, TEMPERATURE,
MAX_STEPS, SWEB_WS_ROOT (set by the sbatch for stable workspace paths).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

# Surface the driver's per-branch "[<wid>][step ...]" progress lines so the
# interleaved trajectories are visible for divergence inspection.
logging.basicConfig(level=logging.INFO, format="%(message)s")

from datasets import load_dataset

from benchmarks.swe_bench import (
    Problem,
    checked_out_repo,
    _run_agent_inferlet,
    _format_user_prompt,
    capture_patch,
)
from tool_server import start_tool_server

INSTANCE_ID = os.environ.get("INSTANCE_ID", "django__django-14373")
PIE_PORT = int(os.environ.get("PIE_PORT", "18098"))
NUM_BRANCHES = int(os.environ.get("NUM_BRANCHES", "2"))
BRANCH_AT_STEP = int(os.environ.get("BRANCH_AT_STEP", "3"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", "600"))
INSTANCE_TIMEOUT_S = float(os.environ.get("INSTANCE_TIMEOUT_S", "1500"))


def _load_problem(instance_id: str) -> Problem:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    for row in ds:
        if row["instance_id"] == instance_id:
            return Problem.from_row(row)
    raise SystemExit(f"instance {instance_id!r} not found in SWE-bench_Verified")


def main() -> int:
    print(f"=== Step-2 fork smoke: {INSTANCE_ID} ===")
    print(f"  num_branches={NUM_BRANCHES} branch_at_step={BRANCH_AT_STEP} "
          f"temperature={TEMPERATURE} max_steps={MAX_STEPS}")
    problem = _load_problem(INSTANCE_ID)

    with checked_out_repo(problem) as ws:
        server, port = start_tool_server(str(ws))
        try:
            result = asyncio.run(_run_agent_inferlet(
                f"ws://127.0.0.1:{PIE_PORT}",
                "openhands-agent@0.1.0",
                task=_format_user_prompt(problem, use_cwd=True),
                tool_server_url=f"http://127.0.0.1:{port}",
                max_steps=MAX_STEPS,
                num_branches=NUM_BRANCHES,
                branch_at_step=BRANCH_AT_STEP,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                idle_timeout_s=IDLE_TIMEOUT_S,
                instance_timeout_s=INSTANCE_TIMEOUT_S,
            ))

            # Capture per-branch diffs from each branch's own workspace BEFORE
            # the tool server drops the forked copies on shutdown.
            trunk_root = str(ws).rstrip("/")
            diffs: dict[str, str] = {}
            for i in range(NUM_BRANCHES):
                wid = str(i)
                root = Path(trunk_root if i == 0 else f"{trunk_root}__fork{wid}")
                diffs[wid] = capture_patch(root) if root.exists() else "<workspace missing>"
        finally:
            server.shutdown()

    # ── Validate ──────────────────────────────────────────────────────────
    if not isinstance(result, dict) or "branches" not in result:
        print("FAIL: inferlet did not return a `branches` array")
        print("  got:", str(result)[:500])
        return 1
    branches = result["branches"]
    print(f"\ninferlet returned {len(branches)} branch(es); "
          f"branch_step={result.get('branch_step')} "
          f"any_finished={result.get('any_finished')} "
          f"total_wall_s={result.get('total_wall_s')}")

    ok = True
    if len(branches) != NUM_BRANCHES:
        print(f"FAIL: expected {NUM_BRANCHES} branches, got {len(branches)}")
        ok = False

    seen_wids = set()
    for b in branches:
        wid = b.get("workspace_id")
        seen_wids.add(wid)
        d = diffs.get(wid, "")
        print(f"  branch[{wid}]: finished={b.get('finished')} steps={b.get('steps')} "
              f"diff_bytes={len(d)} wall_s={b.get('total_wall_s'):.1f} "
              f"msg={str(b.get('message'))[:60]!r}")

    if seen_wids != {str(i) for i in range(NUM_BRANCHES)}:
        print(f"FAIL: workspace_ids {sorted(seen_wids)} != expected 0..{NUM_BRANCHES-1}")
        ok = False

    # Isolation / divergence report. Divergence is what makes best-of-K
    # meaningful: identical diffs across branches ⇒ TTS yields one candidate.
    diff_vals = [diffs[str(i)] for i in range(NUM_BRANCHES)]
    non_empty = sum(1 for d in diff_vals if d and not d.startswith("<"))
    distinct = len(set(diff_vals))

    def _touched_files(diff: str) -> list[str]:
        return re.findall(r"^\+\+\+ b/(.+)$", diff, flags=re.MULTILINE)

    print("\nper-branch summary (files touched / diff hash):")
    for i in range(NUM_BRANCHES):
        d = diffs[str(i)]
        h = hashlib.sha1(d.encode()).hexdigest()[:10] if d else "-"
        files = _touched_files(d)
        print(f"  branch[{i}]: {len(d)}B  sha1={h}  files={files or '[]'}")

    print(f"\nper-branch diffs: {non_empty}/{NUM_BRANCHES} non-empty, "
          f"{distinct} distinct")
    if non_empty == 0:
        print("WARN: no branch produced a diff (agent didn't edit within MAX_STEPS "
              f"={MAX_STEPS}) — bump MAX_STEPS or pick another instance")
    elif distinct >= 2:
        print("PASS+DIVERGE: branches produced DISTINCT diffs — fork test-time "
              "scaling yields genuinely different candidates.")
    else:
        print("PASS-ISOLATED but NO DIVERGENCE: branches ran on isolated "
              "workspaces but produced IDENTICAL diffs. Either the instance is too "
              "easy (one obvious fix) or forked contexts sample identically — "
              "inspect the interleaved [<wid>][step] logs above to tell which.")

    for wid in sorted(diffs):
        d = diffs[wid]
        print(f"\n----- branch[{wid}] diff ({len(d)}B) -----")
        print(d[:1500] if d else "<empty>")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
