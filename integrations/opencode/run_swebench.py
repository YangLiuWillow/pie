#!/usr/bin/env python3
"""SWE-bench Verified against pie, driven by stock opencode.

## The two halves, and which one this is

SWE-bench has a **drive** half and a **score** half:

  1. **Drive** — give the agent a clean checkout at `base_commit` plus the
     issue text, let it work, capture the diff it produced.
  2. **Score** — apply that diff plus the held-out `test_patch` to a fresh
     checkout, run the tests in Docker, check the FAIL_TO_PASS /
     PASS_TO_PASS transitions.

This is the drive half. It emits the predictions JSONL the official grader
eats (`python -m swebench.harness.run_evaluation`). Scoring is deliberately
NOT reimplemented here: the transitions are the benchmark's definition of
correct, and a hand-rolled scorer would be a second opinion about what
"resolved" means. It needs Docker.

## Ported from the OpenHands harness, not reinvented

`integrations/openhands/benchmarks/swe_bench.py` on the OpenHands branch is
the same job for a different agent, and three details there are load-bearing
enough to copy rather than rediscover:

  * **Patch capture stages first.** `git diff HEAD` silently drops every file
    the agent *created*, because untracked files appear in no diff — and an
    agent that writes a reproduction script has created one. So: `git add -A`
    then `diff --cached HEAD`. Upstream's own `get_staged_git_patch`.
  * **The subset is seeded**, so "5 instances" means the same 5 next week.
  * **An empty patch is a result, not an error.** It scores as unresolved,
    which is the honest outcome for an agent that ran and changed nothing.

What is NOT ported is the fake-user nudge loop and the summarizing condenser.
Those are OpenHands-SDK constructs; opencode has its own agent loop and its
own compaction, and reproducing them around a black-box CLI would be
measuring my emulation rather than opencode.

Usage:
    python3 run_swebench.py --n 5 --model pie/qwen3-coder-30b \\
        --out /tmp/preds.jsonl [--timeout 900]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
SUBSET_SEED = 1234

# Instances a Qwen3-Coder-30B-A3B agent is KNOWN to solve.
#
# Why this list exists: a seeded random subset conflates two failures. If the
# agent produces no patch, that is either the serving stack misbehaving or the
# model being unable to do the task — and SWE-bench Verified's base rate for a
# 30B is low enough that "0 of 5" is an unremarkable outcome for a healthy
# stack. On a set the same model has already solved, a zero is attributable.
#
# Provenance: these are `baseline_t0_full_50.report.json`'s `resolved_ids` on
# the OpenHands branch — the 13 that `litellm+qwen3-coder-30b-a3b-t0` resolved
# out of a neutral 50, scored by the official Docker grader. Pie's own
# OpenHands arm reproduced 11 of the 13; the two it missed are listed
# separately because their failures are documented and model-side, not
# machinery: `xarray-4966` was a wrong fix (FAIL_TO_PASS 0/4) and
# `sklearn-10908` was a 0-byte patch after the model corrupted its own
# workspace path and thrashed.
#
# Same model FAMILY, different agent (OpenHands, not opencode) and different
# hardware (A100 CUDA, not Metal). So this bounds what the model can do; it
# does not promise opencode reproduces it. A miss here is worth reading; a
# clean sweep is not proof of parity.
KNOWN_SOLVABLE_BOTH = [
    "django__django-12276",
    "django__django-13028",
    "django__django-13089",
    "django__django-14373",
    "django__django-15569",
    "django__django-16485",
    "matplotlib__matplotlib-22719",
    "pydata__xarray-4075",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-13496",
    "sympy__sympy-19346",
]
KNOWN_SOLVABLE_BASELINE_ONLY = [
    "pydata__xarray-4966",
    "scikit-learn__scikit-learn-10908",
]

HERE = Path(__file__).resolve().parent


# ── Prompt ───────────────────────────────────────────────────────────────
#
# Deliberately close to the official evaluation template's shape: state the
# task, point at the repo root, and be explicit that the fix goes in the
# source rather than the tests. The elaborate 8-phase script the OpenHands
# harness uses is tuned for its own agent; opencode has its own system prompt
# and plan discipline, and layering a second workflow on top would be
# measuring the prompt rather than the serving stack.
PROMPT = """\
You are fixing a real issue in the {repo} repository. The repository is
already checked out in the current working directory at the relevant commit.

<issue>
{problem_statement}
</issue>

Fix the issue by editing the SOURCE files in this repository.

Rules:
- Do NOT modify any test files. The fix is graded by hidden tests.
- Make the smallest change that correctly fixes the issue.
- When you are done, stop. Do not ask questions; you are running unattended.
"""


def load_problems(n: int, seed: int = SUBSET_SEED, instances: list[str] | None = None):
    """`n` seeded-random rows, or exactly the rows named in `instances`."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    if instances:
        by_id = {r["instance_id"]: r for r in ds}
        missing = [i for i in instances if i not in by_id]
        if missing:
            raise SystemExit(f"not in {DATASET}: {missing}")
        return [by_id[i] for i in instances]
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    return [ds[i] for i in idx]


def prepare_workspace(row: dict, root: Path) -> Path:
    """A clean checkout of `repo` at `base_commit`.

    Fetches the single commit rather than cloning history: these are django,
    sympy and astropy, and a full clone of each is minutes of wall time that
    tells us nothing about the model.
    """
    ws = root / row["instance_id"]
    ws.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{row['repo']}.git"
    run = lambda *a: subprocess.run(a, cwd=ws, check=True, capture_output=True, text=True)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", url)
    run("git", "fetch", "-q", "--depth", "1", "origin", row["base_commit"])
    run("git", "checkout", "-q", "FETCH_HEAD")
    return ws


def capture_patch(ws: Path) -> str:
    """The agent's diff, staged first so created files are included.

    See the module docstring: a plain `git diff HEAD` loses every new file,
    which on this benchmark means losing reproduction scripts and sometimes
    the fix itself.

    The harness's own `opencode.json` is removed first. Staging everything is
    what makes created files visible, and it makes the config visible too —
    the first smoke run produced a 1,036-byte "patch" that was nothing but
    the provider config I had just copied in. A patch that grades a file the
    agent never wrote is worse than an empty one, because it looks like work.
    """
    (ws / "opencode.json").unlink(missing_ok=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True, text=True)
    res = subprocess.run(
        ["git", "--no-pager", "diff", "--no-color", "--cached", "HEAD"],
        cwd=ws, capture_output=True, text=True,
    )
    return res.stdout


def run_agent(ws: Path, prompt: str, model: str, opencode: str, timeout: int, tries: int = 3):
    """One unattended opencode run inside `ws`. Returns (ok, seconds, note).

    Retries a **startup** failure, and only that. opencode can come up with
    `{"name":"UnknownError","message":"Unexpected server error"}` before it
    writes a single log line — observed after an earlier run was left hung on
    an interactive permission prompt with no TTY to answer it. It clears on
    its own, so the cost of retrying is seconds and the cost of not retrying
    is a benchmark row scored 0 for a reason that has nothing to do with the
    model.

    The discriminator is time: a startup failure returns in about a second,
    where a genuine agent run does not. A slow failure is passed through
    untouched — it did work, and whatever it wrote is its answer.
    """
    t0 = time.time()
    for attempt in range(1, tries + 1):
        shutil.copy(HERE / "opencode.json", ws / "opencode.json")
        start = time.time()
        # Through the user's shell, not exec'd directly, and this is not
        # cosmetic: `subprocess.run([opencode, ...])` fails EVERY time with
        # `{"name":"UnknownError","message":"Unexpected server error"}` before
        # opencode writes a log line, while `zsh -c '<the same argv>'` in the
        # same cwd succeeds every time. Ruled out, by measurement: the
        # environment (`/usr/bin/env` output is byte-identical either way),
        # stdin (DEVNULL fails from Python and succeeds from zsh), stdout
        # capture, `close_fds`, `start_new_session`, PATH (adding either
        # ~/.cargo/bin or ~/.opencode/bin changes nothing), and the file
        # descriptor limit. The cause is unresolved; a shell is what works, and
        # it is also how a person runs this, so it is the faithful invocation
        # rather than merely the lucky one.
        cmd = " ".join(shlex.quote(a) for a in [opencode, "run", "-m", model, prompt])
        try:
            p = subprocess.run(
                ["/bin/zsh", "-c", cmd],
                cwd=ws, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # A timeout is a result: whatever the agent had written to disk by
            # then is still its answer, and is captured by the caller.
            return False, time.time() - t0, f"timeout after {timeout}s"
        if p.returncode == 0:
            return True, time.time() - t0, ""
        elapsed = time.time() - start
        note = f"exit {p.returncode}: {p.stderr.strip()[:200]}"
        if elapsed > 10 or attempt == tries:
            return False, time.time() - t0, note
        print(f"    startup failure in {elapsed:.0f}s, retrying ({attempt}/{tries - 1})")
        time.sleep(5)
    return False, time.time() - t0, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="pie/qwen3-coder-30b")
    ap.add_argument("--out", default="/tmp/preds.jsonl")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--label", default=None, help="model_name_or_path in the predictions")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--opencode", default=os.path.expanduser("~/.opencode/bin/opencode"))
    ap.add_argument("--known-solvable", action="store_true",
                    help="use instances this model family has already solved "
                         "(see KNOWN_SOLVABLE_BOTH) instead of a seeded subset")
    ap.add_argument("--instances", nargs="+", help="explicit instance ids")
    ap.add_argument("--restart-cmd",
                    help="shell command run BEFORE each instance, e.g. a fresh "
                         "`pie serve`. Isolates the server-wear defect so an "
                         "accuracy signal is not swamped by it — and hides that "
                         "defect, which is why it is opt-in and reported below.")
    args = ap.parse_args()

    label = args.label or args.model.replace("/", "-")
    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="swebench-"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"workspaces: {root}")

    chosen = args.instances or (KNOWN_SOLVABLE_BOTH[: args.n] if args.known_solvable else None)
    problems = load_problems(args.n, instances=chosen)
    if args.restart_cmd:
        print("restarting the server before each instance — the wear defect is "
              "being ISOLATED, not fixed\n")
    print(f"{len(problems)} instances: {[p['instance_id'] for p in problems]}\n")

    rows, summary = [], []
    for i, row in enumerate(problems, 1):
        iid = row["instance_id"]
        print(f"[{i}/{len(problems)}] {iid} ({row['repo']}) …", flush=True)
        if args.restart_cmd:
            r = subprocess.run(["/bin/zsh", "-c", args.restart_cmd],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    restart FAILED: {r.stderr.strip()[:200]}")
        try:
            ws = prepare_workspace(row, root)
        except subprocess.CalledProcessError as e:
            print(f"    checkout FAILED: {e.stderr.strip()[:200]}")
            summary.append((iid, "checkout-failed", 0.0, 0))
            rows.append({"instance_id": iid, "model_name_or_path": label, "model_patch": ""})
            continue

        prompt = PROMPT.format(repo=row["repo"], problem_statement=row["problem_statement"])
        ok, secs, note = run_agent(ws, prompt, args.model, args.opencode, args.timeout)
        patch = capture_patch(ws)
        # The patch is what counts, not the exit code: an agent that timed out
        # having already written a correct fix is still graded on the fix.
        state = "ok" if ok else ("timeout" if "timeout" in note else "agent-error")
        print(f"    {state} in {secs:.0f}s, patch {len(patch)} bytes"
              + (f" — {note}" if note and state != "ok" else ""))
        summary.append((iid, state, secs, len(patch)))
        rows.append({"instance_id": iid, "model_name_or_path": label, "model_patch": patch})

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nwrote {args.out}")
    print(f"{'instance':<34} {'state':<14} {'secs':>7} {'patch B':>8}")
    for iid, state, secs, n in summary:
        print(f"{iid:<34} {state:<14} {secs:>7.0f} {n:>8}")
    nonempty = sum(1 for _, _, _, n in summary if n > 0)
    print(f"\n{nonempty}/{len(summary)} produced a non-empty patch.")
    print("Scoring needs Docker:\n"
          f"  python -m swebench.harness.run_evaluation \\\n"
          f"      --dataset_name {DATASET} --predictions_path {args.out} \\\n"
          f"      --max_workers 4 --run_id pie-opencode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
