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
import hashlib
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

# Two copies of the same benchmark, and they are not interchangeable.
#
# DATASET drives. It stays on `princeton-nlp/…` because the seeded subset is
# defined by ROW INDEX into this copy (`load_problems`), so switching it would
# silently redefine what `--n 5` means and break the "same 5 next week"
# guarantee this harness is built on.
#
# SCORING_DATASET grades. SWE-bench 5.0's harness reads an `image` column that
# the princeton-nlp copy does not have, so passing DATASET to the grader fails
# outright. Driving needs only problem_statement/repo/base_commit, which both
# copies carry identically, so the split costs nothing.
DATASET = "princeton-nlp/SWE-bench_Verified"
SCORING_DATASET = "SWE-bench/SWE-bench_Verified"
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


def load_snapshot(path: Path, n: int | None = None):
    """Cases from a `test-time-bench/dataset-snapshot.v2` file.

    The snapshot is the source of truth for the PROMPT, not merely the case
    list: each case carries a fully rendered `input`, so changing the prompt
    means publishing a new dataset version. `PROMPT` in this module has no such
    property — it can be edited between two runs that then get compared.

    Returned rows are shaped like the HuggingFace rows the rest of this file
    consumes, plus `_prompt`, which `main` uses verbatim when present.
    """
    d = json.loads(path.read_text())
    if d.get("schema") != "test-time-bench/dataset-snapshot.v2":
        raise SystemExit(f"unexpected snapshot schema: {d.get('schema')}")
    rows = []
    for c in d["cases"][: n or len(d["cases"])]:
        g = c.get("grading") or {}
        rows.append({
            "instance_id": c["id"],
            "repo": g.get("repo"),
            "base_commit": g.get("base_commit"),
            "problem_statement": "",   # unused: `_prompt` is authoritative here
            "_prompt": c["input"],
        })
    missing = [r["instance_id"] for r in rows if not (r["repo"] and r["base_commit"])]
    if missing:
        raise SystemExit(f"snapshot cases lack repo/base_commit: {missing}")
    return rows


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


def capture_patch(ws: Path, base: str = "HEAD") -> str:
    """The agent's diff against the BASE COMMIT, staged first so created files
    are included.

    Against `base`, not `HEAD`, and that distinction silently discarded real
    work: opencode's agent finishes by *committing*. `git add -A` then
    `diff --cached HEAD` compares the index to HEAD, so once the agent commits,
    HEAD already contains the fix, index == HEAD, and the diff is EMPTY. The
    harness then records a 0-byte patch for a case the model actually solved,
    and it looks exactly like an agent that did nothing.

    Measured: two Qwen3-8B cases each ran `edit` → `git add` → `git commit`
    ("1 file changed, 4 insertions(+)") and both were captured as empty.
    Diffing against the base commit captures committed and uncommitted work
    alike. (The earlier Coder-30B runs were checked and have 0 commits past
    base, so their recorded patches are unaffected.)

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
        ["git", "--no-pager", "diff", "--no-color", "--cached", base],
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
        # Output goes to a FILE, not a pipe, and this is not a style choice.
        # `capture_output=True` hands every descendant an inheritable pipe, and
        # opencode starts a local server: when that server outlives the CLI,
        # `communicate()` blocks waiting for EOF that never comes — *including
        # after* the timeout fires and kills the child, because the pipe still
        # has a writer. Observed directly: opencode logged `init`, exited, and
        # the driver sat in `poll()` for 21 minutes with no child process at
        # all. A file has no such lifetime coupling, and it leaves a per-case
        # log worth reading afterwards.
        log_path = ws.parent / f"{ws.name}.opencode.log"
        try:
            with open(log_path, "ab") as logf:
                p = subprocess.run(
                    ["/bin/zsh", "-c", cmd],
                    cwd=ws, stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=False, timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            # A timeout is a result: whatever the agent had written to disk by
            # then is still its answer, and is captured by the caller.
            return False, time.time() - t0, f"timeout after {timeout}s"
        if p.returncode == 0:
            return True, time.time() - t0, ""
        elapsed = time.time() - start
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-200:].strip()
        except OSError:
            pass
        note = f"exit {p.returncode}: {tail}"
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
    ap.add_argument("--dataset-snapshot",
                    help="a test-time-bench dataset-snapshot.v2 file. Drives from ITS "
                         "cases and ITS rendered prompts instead of the HF dataset and "
                         "this module's PROMPT constant, so the prompt is pinned by a "
                         "dataset version rather than by an editable string here.")
    ap.add_argument("--restart-cmd",
                    help="shell command run BEFORE each instance, e.g. a fresh "
                         "`pie serve`. Isolates the server-wear defect so an "
                         "accuracy signal is not swamped by it — and hides that "
                         "defect, which is why it is opt-in and reported below.")
    args = ap.parse_args()

    label = args.label or args.model.replace("/", "-")
    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="swebench-"))
    root.mkdir(parents=True, exist_ok=True)
    # RESOLVED once, here, so every consumer agrees on one spelling of the path.
    # On macOS `/tmp` is a symlink to `/private/tmp`, and opencode's permission
    # gate compares the tool's target against the project directory TEXTUALLY:
    # a tool call naming `/tmp/...` inside a project opencode knows as
    # `/private/tmp/...` reads as outside the workspace, so it is auto-rejected
    # with no TTY to approve it. Measured across two cases of one run: the case
    # whose model happened to emit `/private/tmp/...` completed 19 tool calls,
    # while the case that copied the prompt's `/tmp/...` had its FIRST call
    # rejected and stopped. Same harness, same model — the difference was which
    # spelling the model echoed, which also makes the failure intermittent.
    root = root.resolve()
    print(f"workspaces: {root}")

    if args.dataset_snapshot:
        problems = load_snapshot(Path(args.dataset_snapshot), args.n if args.n else None)
        print(f"driving from snapshot {args.dataset_snapshot} "
              f"(sha256 {hashlib.sha256(Path(args.dataset_snapshot).read_bytes()).hexdigest()[:16]}…)")
    else:
        chosen = args.instances or (KNOWN_SOLVABLE_BOTH[: args.n] if args.known_solvable else None)
        problems = load_problems(args.n, instances=chosen)
    if args.restart_cmd:
        print("restarting the server before each instance — the wear defect is "
              "being ISOLATED, not fixed\n")
    print(f"{len(problems)} instances: {[p['instance_id'] for p in problems]}\n")

    rows, summary, substitutions = [], [], []
    for i, row in enumerate(problems, 1):
        iid = row["instance_id"]
        print(f"[{i}/{len(problems)}] {iid} ({row['repo']}) …", flush=True)
        if args.restart_cmd:
            r = subprocess.run(["/bin/zsh", "-c", args.restart_cmd],
                               capture_output=True, text=True)
            if r.returncode != 0:
                # FATAL. A failed restart leaves the agent talking to a dead or
                # stale server, and every downstream signal then lies: the agent
                # errors out in ~60 s, the patch is 0 bytes, and the row is
                # indistinguishable from a model that tried and failed. Observed
                # exactly once, and it cost a run: the reboot hit
                # "does not fit the memory this machine has left" while a 14 GB
                # grader VM was resident, and the case was recorded as
                # `agent-error` with an empty patch.
                raise SystemExit(
                    f"FATAL: restart-cmd failed before {iid} "
                    f"(exit {r.returncode}). Refusing to drive against a server "
                    f"this run did not start.\n"
                    f"{(r.stderr or r.stdout).strip()[-400:]}")
        try:
            ws = prepare_workspace(row, root)
        except subprocess.CalledProcessError as e:
            print(f"    checkout FAILED: {e.stderr.strip()[:200]}")
            summary.append((iid, "checkout-failed", 0.0, 0))
            rows.append({"instance_id": iid, "model_name_or_path": label, "model_patch": ""})
            continue

        # A snapshot's rendered prompt wins: re-wrapping it in this module's
        # template would measure our re-rendering, not the pinned benchmark.
        #
        # ONE substitution is applied, and it is not optional: TTB's snapshot
        # prompts hardcode `Worktree: /workspace/<repo>`, which is the layout
        # of TTB's own eval-worker CONTAINER. Outside that container the path
        # does not exist, and the observed failure is silent rather than loud —
        # the agent obediently globs `/workspace/...`, opencode's permission
        # gate fires because it is outside the workspace, the call is
        # auto-rejected with no TTY to approve it, and the agent stops after
        # one model call having done nothing. Measured: 1 model call, ~30
        # output tokens, 0-byte patch, `state: ok`.
        case_started_ms = int(time.time() * 1000)
        prompt = row.get("_prompt")
        if prompt:
            container = f"/workspace/{row['repo'].split('/')[-1]}"
            prompt, subs = prompt.replace(container, str(ws)), prompt.count(container)
            if subs == 0:
                # Loud, because a snapshot whose worktree convention changed
                # would otherwise reintroduce the silent failure above.
                raise SystemExit(
                    f"{iid}: snapshot prompt names no worktree matching {container!r}; "
                    "the container-path rebind found nothing to rebind")
            substitutions.append({"case_id": iid, "from": container,
                                  "to": str(ws), "count": subs})
        else:
            prompt = PROMPT.format(repo=row["repo"],
                                   problem_statement=row["problem_statement"])
        ok, secs, note = run_agent(ws, prompt, args.model, args.opencode, args.timeout)
        patch = capture_patch(ws, row.get("base_commit") or "HEAD")
        # The patch is what counts, not the exit code: an agent that timed out
        # having already written a correct fix is still graded on the fix.
        state = "ok" if ok else ("timeout" if "timeout" in note else "agent-error")
        print(f"    {state} in {secs:.0f}s, patch {len(patch)} bytes"
              + (f" — {note}" if note and state != "ok" else ""))
        summary.append((iid, state, secs, len(patch), case_started_ms))
        rows.append({"instance_id": iid, "model_name_or_path": label, "model_patch": patch})

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nwrote {args.out}")

    # Per-case record for `ttb_summary.py`, which joins it against opencode's
    # own token accounting to produce a `test-time-bench`-shaped run summary.
    # Emitted always, not behind a flag: it is three fields the loop already
    # has, and the runs that most needed it are the ones nobody thought to ask
    # for it on. `workspace` is the join key — opencode sessions are
    # directory-scoped, so it is what links a case to its telemetry.
    if substitutions:
        subs_path = Path(args.out).with_suffix(".prompt-rebind.json")
        subs_path.write_text(json.dumps(
            {"note": "TTB snapshot prompts hardcode the container worktree "
                     "`/workspace/<repo>`; each case's prompt had it rebound to the "
                     "real checkout. Without this the agent is permission-blocked "
                     "on a path that does not exist and stops after one call.",
             "substitutions": substitutions}, indent=2) + "\n")
        print(f"wrote {subs_path} ({len(substitutions)} prompts rebound)")

    cases_path = Path(args.out).with_suffix(".cases.jsonl")
    cases_path.write_text("".join(
        json.dumps({"case_id": iid, "state": state, "seconds": round(secs, 1),
                    # When this case started, so telemetry joins to THIS run.
                    # A re-run into the same TTB_OUT_DIR leaves the previous
                    # run's sessions at the identical workspace path, and a
                    # directory-only join silently sums both — observed: 6
                    # model calls and 490 s reported for a 5-call, 426 s run.
                    "started_ms": started_ms,
                    "patch_bytes": nbytes,
                    # RESOLVED: on macOS /tmp is a symlink to /private/tmp, and
                    # opencode stores the resolved path. Recording the symlink
                    # made every telemetry join miss silently — 0 tokens, 0
                    # model calls, on cases that had really run.
                    "workspace": str((root / iid).resolve())}) + "\n"
        for iid, state, secs, nbytes, started_ms in summary))
    print(f"wrote {cases_path}")
    print(f"{'instance':<34} {'state':<14} {'secs':>7} {'patch B':>8}")
    for iid, state, secs, n, _ in summary:
        print(f"{iid:<34} {state:<14} {secs:>7.0f} {n:>8}")
    nonempty = sum(1 for _, _, _, n, _ in summary if n > 0)
    print(f"\n{nonempty}/{len(summary)} produced a non-empty patch.")
    print("Scoring needs Docker:\n"
          f"  python -m swebench.harness.run_evaluation \\\n"
          f"      --dataset_name {SCORING_DATASET} --predictions_path {args.out} \\\n"
          f"      --max_workers 4 --run_id pie-opencode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
