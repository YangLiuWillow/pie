"""Phase-1 SWE-Bench Verified harness — driving half.

The two halves of SWE-Bench:

  1. **Drive** — for each problem, hand an OpenHands agent a fresh checkout
     of the target repo at ``base_commit`` plus the ``problem_statement``,
     let it run, capture ``git diff HEAD`` as the predicted patch.
  2. **Score** — apply the predicted patch + the held-out ``test_patch`` to
     a clean checkout, run the test suite in a Docker sandbox, check the
     ``FAIL_TO_PASS``/``PASS_TO_PASS`` transitions. This requires Docker and
     is delegated to ``python -m swebench.harness.run_evaluation``.

This module implements (1) and produces a predictions JSONL compatible with
the grader.

Predictions JSONL schema (one row per problem):

    {
      "instance_id":         "<dataset row's instance_id>",
      "model_name_or_path":  "<a label, e.g. 'pie+qwen3-coder-32b'>",
      "model_patch":         "<unified diff against HEAD@base_commit>"
    }

Three "backends" are supported for development:

  * ``"pie"``     — real PieLLM. Requires a running ``pie serve`` and the
                    ``openhands-completion`` inferlet installed.
  * ``"litellm"`` — vanilla openhands.sdk.LLM. Requires real API creds
                    (OpenAI / Anthropic / local OpenAI-compat endpoint).
  * ``"test"``    — TestLLM scripted to return a no-op finish. Useful for
                    smoke-testing the harness plumbing without a model.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import tempfile
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ─── Dataset access ─────────────────────────────────────────────────────


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
SUBSET_SEED = 20251201    # frozen; do not change between runs
DEFAULT_SUBSET_N = 50


def load_verified(dataset: str = DEFAULT_DATASET, split: str = DEFAULT_SPLIT):
    """Return the HF Dataset object. Caches under ``~/.cache/huggingface``."""
    from datasets import load_dataset
    return load_dataset(dataset, split=split)


def deterministic_subset_indices(n_total: int, n_subset: int, seed: int = SUBSET_SEED) -> list[int]:
    """Deterministic ``n_subset`` indices into a ``n_total``-element dataset."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_subset))


# ─── Problem & prediction types ─────────────────────────────────────────


@dataclass
class Problem:
    """A single SWE-Bench Verified row, narrowed to what we need to drive."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Problem":
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            hints_text=row.get("hints_text") or "",
        )


@dataclass
class Prediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    # Non-standard fields useful for debugging — the grader will ignore them.
    wall_clock_s: float = 0.0
    agent_iterations: int = 0
    error: str = ""
    stuck_retries: int = 0

    def to_jsonl(self) -> str:
        return json.dumps({
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
            "_metadata": {
                "wall_clock_s": round(self.wall_clock_s, 3),
                "agent_iterations": self.agent_iterations,
                "error": self.error,
                "stuck_retries": self.stuck_retries,
            },
        })


# ─── Workspace setup ─────────────────────────────────────────────────────


@contextmanager
def checked_out_repo(problem: Problem, *, cache_dir: Path | None = None) -> Iterator[Path]:
    """Clone ``problem.repo`` at ``problem.base_commit`` into a temp dir.

    If ``cache_dir`` is given, we maintain a bare clone there and only copy
    the working tree into the temp dir — much faster across many problems
    that share a repo.
    """
    with tempfile.TemporaryDirectory(prefix=f"sweb-{problem.instance_id}-") as tmp:
        ws = Path(tmp) / "repo"
        url = f"https://github.com/{problem.repo}.git"
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            bare = cache_dir / (problem.repo.replace("/", "__") + ".git")
            if not bare.exists():
                _run(["git", "clone", "--bare", url, str(bare)])
            _run(["git", "clone", str(bare), str(ws)])
        else:
            _run(["git", "clone", "--depth", "200", url, str(ws)])

        _run(["git", "-C", str(ws), "fetch", "origin", problem.base_commit])
        _run(["git", "-C", str(ws), "checkout", problem.base_commit])
        yield ws


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stderr: {res.stderr}"
        )


def capture_patch(workspace: Path) -> str:
    """Return ``git diff HEAD`` — the patch the agent produced."""
    res = subprocess.run(
        ["git", "-C", str(workspace), "diff", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return res.stdout


# ─── Backend factory ─────────────────────────────────────────────────────


def build_llm(backend: str, **kwargs):
    """Return a configured ``openhands.sdk.LLM`` for the chosen backend.

    Args:
        backend: ``"pie"`` | ``"litellm"`` | ``"test"``.
        **kwargs: backend-specific overrides (model, base_url, api_key, ...).
    """
    # Patches shared SDK machinery (both backends run through it), so install
    # once regardless of which backend is selected — see editor_repair's
    # module docstring for why this isn't Pie-specific.
    from pie_openhands.editor_repair import install as _install_editor_repair
    _install_editor_repair()

    if backend == "pie":
        from pie_openhands import PieLLM
        return PieLLM(
            model=kwargs.pop("model", "default"),
            pie_uri=kwargs.pop("pie_uri", "ws://127.0.0.1:8080"),
            pie_username=kwargs.pop("pie_username", "local-dev"),
            pie_inferlet=kwargs.pop("pie_inferlet", "openhands-completion@0.1.0"),
            num_retries=kwargs.pop("num_retries", 3),
            **kwargs,
        )
    if backend == "litellm":
        from openhands.sdk import LLM
        return LLM(**kwargs)
    if backend == "test":
        from openhands.sdk import Message, TextContent
        from openhands.sdk.testing import TestLLM
        # Scripted: agent says it's done, no edits. Useful for plumbing tests.
        reply = Message(
            role="assistant",
            content=[TextContent(text="I have no edits to make — nothing to fix here.")],
        )
        return TestLLM.from_messages([reply] * 100)
    raise ValueError(f"unknown backend: {backend!r}")


# ─── Agent construction ──────────────────────────────────────────────────


SWE_BENCH_SYSTEM_SUFFIX = textwrap.dedent("""\
    You are solving a software-engineering problem from SWE-Bench Verified.
    You have direct file access to a checked-out repository in your workspace
    and a bash terminal. Read the bug report, locate the relevant code, make
    the minimum edit that fixes the bug, and stop.

    Constraints:
      * Do not run the test suite — the grader will do that.
      * Do not create new files unless the bug requires it.
      * Make no commits, no branches, no git operations at all.
      * Keep changes scoped to the reported bug. Do not refactor.
""")


def build_agent(llm):
    """Return an ``Agent`` configured with the default file/terminal tools."""
    from openhands.sdk import Agent
    from openhands.tools.preset.default import get_default_tools

    tools = get_default_tools(enable_browser=False)
    return Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": False, "extra_instructions": SWE_BENCH_SYSTEM_SUFFIX},
    )


# ─── Driver ─────────────────────────────────────────────────────────────


# Sent when the SDK's StuckDetector fires (repeating/alternating action-
# observation loops — see docs/openhands-integration.md investigation notes).
# Confirmed via live traces that this pattern often comes from the model
# fixating on a scratch file it created itself rather than the real bug
# location, so the nudge calls that out explicitly rather than just saying
# "try something different".
STUCK_NUDGE_MESSAGE = (
    "You appear to be stuck: repeating the same action without making "
    "progress. Stop repeating that exact tool call. Re-read the problem "
    "statement and double-check you are editing the actual source file "
    "where the reported bug lives — not a scratch/test file you created to "
    "explore the issue. If you already understand the fix, make the edit "
    "directly."
)


def run_with_stuck_retries(
    conv,
    *,
    max_stuck_retries: int = 2,
    instance_id: str = "?",
) -> int:
    """Run ``conv`` to completion, nudging past StuckDetector triggers.

    ``conv.run()`` already runs until FINISHED/STUCK/erroring/iteration-cap.
    If it comes back STUCK, send ``STUCK_NUDGE_MESSAGE`` (which the SDK
    treats as a normal user message, clearing the STUCK status — see
    ``LocalConversation.send_message``) and call ``run()`` again, up to
    ``max_stuck_retries`` times. Returns the number of nudges actually sent.

    Takes a duck-typed ``conv`` (only ``.run()``, ``.send_message()``, and
    ``.state.execution_status`` are used) so this is unit-testable without a
    real Agent/tool/workspace.
    """
    from openhands.sdk import ConversationExecutionStatus

    conv.run()

    stuck_retries = 0
    while (
        conv.state.execution_status == ConversationExecutionStatus.STUCK
        and stuck_retries < max_stuck_retries
    ):
        stuck_retries += 1
        logger.warning(
            "%s: stuck pattern detected, sending nudge (retry %d/%d)",
            instance_id, stuck_retries, max_stuck_retries,
        )
        conv.send_message(STUCK_NUDGE_MESSAGE)
        conv.run()

    return stuck_retries


def solve_one(
    problem: Problem,
    *,
    backend: str,
    backend_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 50,
    max_stuck_retries: int = 2,
    cache_dir: Path | None = None,
    label: str | None = None,
) -> Prediction:
    """Run one OpenHands agent against one problem, return a Prediction.

    If the SDK's StuckDetector fires, sends ``STUCK_NUDGE_MESSAGE`` and
    resumes the run (up to ``max_stuck_retries`` times) instead of giving up
    immediately — see the stuck-loop investigation notes for why this is
    needed even with the escape-bug repair layer installed.
    """
    from openhands.sdk import Conversation

    label = label or f"{backend}+default"
    backend_kwargs = backend_kwargs or {}
    t0 = time.monotonic()

    try:
        with checked_out_repo(problem, cache_dir=cache_dir) as ws:
            llm = build_llm(backend, **backend_kwargs)
            agent = build_agent(llm)
            conv = Conversation(
                agent=agent,
                workspace=str(ws),
                max_iteration_per_run=max_iterations,
                visualizer=None,
            )
            conv.send_message(_format_user_prompt(problem))
            stuck_retries = run_with_stuck_retries(
                conv,
                max_stuck_retries=max_stuck_retries,
                instance_id=problem.instance_id,
            )

            patch = capture_patch(ws)
            return Prediction(
                instance_id=problem.instance_id,
                model_name_or_path=label,
                model_patch=patch,
                wall_clock_s=time.monotonic() - t0,
                agent_iterations=_count_iterations(conv),
                stuck_retries=stuck_retries,
            )
    except Exception as e:
        logger.exception("solve_one failed for %s", problem.instance_id)
        return Prediction(
            instance_id=problem.instance_id,
            model_name_or_path=label,
            model_patch="",
            wall_clock_s=time.monotonic() - t0,
            agent_iterations=0,
            error=f"{type(e).__name__}: {e}",
        )


def _format_user_prompt(problem: Problem) -> str:
    hints = (
        f"\n\nMaintainer hints:\n{problem.hints_text}"
        if problem.hints_text.strip() else ""
    )
    return (
        f"Repository: {problem.repo}\n"
        f"Base commit: {problem.base_commit}\n\n"
        f"## Problem statement\n\n{problem.problem_statement}{hints}\n\n"
        "Please make the minimum edit to fix this bug. Do not run tests."
    )


def _count_iterations(conv) -> int:
    """Best-effort count of agent steps from the conversation event log."""
    try:
        return sum(1 for _ in conv.state.events)
    except Exception:
        return 0


# ─── Top-level run loop ─────────────────────────────────────────────────


@dataclass
class RunOptions:
    backend: str = "test"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    subset_size: int = DEFAULT_SUBSET_N
    subset_indices: list[int] | None = None
    instance_ids: list[str] | None = None  # if set, takes precedence over subset
    max_iterations: int = 50
    max_stuck_retries: int = 2
    cache_dir: Path | None = None
    output_path: Path = Path("predictions.jsonl")
    label: str | None = None


def run(options: RunOptions) -> Path:
    """Drive the full subset, writing one JSONL row per problem.

    Returns the output path.
    """
    ds = load_verified()
    rows = list(ds)

    if options.instance_ids:
        wanted = set(options.instance_ids)
        selected = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in selected}
        if missing:
            raise ValueError(f"unknown instance_ids: {sorted(missing)}")
    else:
        indices = options.subset_indices or deterministic_subset_indices(
            len(rows), options.subset_size
        )
        selected = [rows[i] for i in indices]

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    with options.output_path.open("w") as f:
        for row in selected:
            problem = Problem.from_row(row)
            logger.info("solving %s (%s)", problem.instance_id, problem.repo)
            pred = solve_one(
                problem,
                backend=options.backend,
                backend_kwargs=options.backend_kwargs,
                max_iterations=options.max_iterations,
                max_stuck_retries=options.max_stuck_retries,
                cache_dir=options.cache_dir,
                label=options.label,
            )
            f.write(pred.to_jsonl() + "\n")
            f.flush()
            logger.info(
                "  → %s (%.1fs, %d iters, %d-byte patch%s%s)",
                "ok" if not pred.error else "ERROR",
                pred.wall_clock_s, pred.agent_iterations,
                len(pred.model_patch),
                f", {pred.stuck_retries} stuck-retries" if pred.stuck_retries else "",
                f", error={pred.error!r}" if pred.error else "",
            )

    return options.output_path
