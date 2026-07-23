"""SWE-Bench Verified harness — driving half.

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

Key features ported from the official OpenHands benchmarks repo
(All-Hands-AI/benchmarks):

  * **Fake user response loop** — when the agent responds with a plain-text
    message instead of a tool call, send a nudge and re-run instead of
    letting FINISHED fire after a single turn.
  * **Condenser** — ``LLMSummarizingCondenser`` truncates long conversation
    history to prevent context degradation on multi-turn agent loops.
  * **Structured 8-phase prompt** — guides the model through a systematic
    read → run → explore → test → analyze → fix → verify → review workflow,
    matching the official evaluation template.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_llm_calls: int = 0
    response_latencies: list[float] = field(default_factory=list)
    # Per-step breakdown (populated by Pattern A agent inferlet).
    generate_s: float = 0.0
    tool_s: float = 0.0
    per_step: list[dict[str, Any]] = field(default_factory=list)
    # KV-session telemetry (populated by the pie backend with --pie-session).
    pie_session: dict[str, Any] = field(default_factory=dict)
    # Fork test-time scaling: K candidate patches (one per forked branch). The
    # top-level model_patch is branch 0; best-of-K is computed downstream by
    # bestofk_split.py + scoring each candidate file.
    candidates: list[dict[str, Any]] = field(default_factory=list)

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
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "num_llm_calls": self.num_llm_calls,
                "response_latencies": [round(l, 4) for l in self.response_latencies],
                "generate_s": round(self.generate_s, 3),
                "tool_s": round(self.tool_s, 3),
                "per_step": self.per_step,
                "pie_session": self.pie_session,
                "candidates": self.candidates,
            },
        })


# ─── Workspace setup ─────────────────────────────────────────────────────


@contextmanager
def checked_out_repo(problem: Problem, *, cache_dir: Path | None = None) -> Iterator[Path]:
    """Clone ``problem.repo`` at ``problem.base_commit`` into a temp dir.

    If ``cache_dir`` is given, we maintain a bare clone there and only copy
    the working tree into the temp dir — much faster across many problems
    that share a repo.

    If ``$SWEB_WS_ROOT`` is set, the working tree lives at the deterministic
    path ``$SWEB_WS_ROOT/sweb-<instance_id>/repo`` instead of a randomized
    tempdir. Equivalence runs need this: the workspace path leaks into the
    prompts via tool output (FileEditor cwd, terminal output, tracebacks), so
    two arms only see identical prompts — a precondition for identical t=0
    trajectories — if the path is identical too.
    """
    ws_root = os.environ.get("SWEB_WS_ROOT")
    if ws_root:
        base = Path(ws_root) / f"sweb-{problem.instance_id}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
        try:
            yield _clone_into(base, problem, cache_dir)
        finally:
            shutil.rmtree(base, ignore_errors=True)
    else:
        with tempfile.TemporaryDirectory(prefix=f"sweb-{problem.instance_id}-") as tmp:
            yield _clone_into(Path(tmp), problem, cache_dir)


class _Aborted(Exception):
    """Raised by a worker that declines to start because the run is aborting."""


# Per-bare-path locks guarding the shared cache_dir clones (see _clone_into).
_bare_clone_locks: dict[str, threading.Lock] = {}
_bare_clone_locks_guard = threading.Lock()


def _bare_clone_lock(key: str) -> threading.Lock:
    with _bare_clone_locks_guard:
        return _bare_clone_locks.setdefault(key, threading.Lock())


def _clone_into(base: Path, problem: Problem, cache_dir: Path | None) -> Path:
    ws = base / "repo"
    url = f"https://github.com/{problem.repo}.git"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        bare = cache_dir / (problem.repo.replace("/", "__") + ".git")
        # The bare clone is shared across all instances of the same repo, so
        # under --concurrency two threads solving two instances of that repo
        # would otherwise race on the check-then-clone below. Serialize the
        # create per bare path, and publish atomically via a tmp-then-rename so
        # a concurrent reader never sees a half-written bare repo.
        with _bare_clone_lock(str(bare)):
            if not bare.exists():
                tmp_bare = bare.with_name(bare.name + ".tmp")
                shutil.rmtree(tmp_bare, ignore_errors=True)
                _run(["git", "clone", "--bare", url, str(tmp_bare)])
                tmp_bare.rename(bare)
        _run(["git", "clone", str(bare), str(ws)])
    else:
        _run(["git", "clone", "--depth", "200", url, str(ws)])

    _run(["git", "-C", str(ws), "fetch", "origin", problem.base_commit])
    _run(["git", "-C", str(ws), "checkout", problem.base_commit])
    return ws


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
    I've already taken care of all changes to any of the test files described \
    in the issue. This means you DON'T have to modify the testing logic or \
    any of the tests in any way!
    Also the development Python environment is already set up for you \
    (i.e., all dependencies already installed), so you don't need to install \
    other packages.
    Your task is to make the minimal changes to non-test files in the \
    repository directory to ensure the issue is satisfied.

    Follow these phases to resolve the issue:

    Phase 1. READING: read the problem and reword it in clearer terms
       1.1 If there are code or config snippets, express in words any best practices or conventions in them.
       1.2 Highlight message errors, method names, variables, file names, stack traces, and technical details.
       1.3 Explain the problem in clear terms.
       1.4 Enumerate the steps to reproduce the problem.
       1.5 Highlight any best practices to take into account when testing and fixing the issue.

    Phase 2. RUNNING: install and run the tests on the repository
       2.1 Follow the readme.
       2.2 Install the environment and anything needed.
       2.3 Iterate and figure out how to run the tests.

    Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
       3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
       3.2 Identify all files related to the problem statement.
       3.3 Propose the methods and files to fix the issue and explain why.
       3.4 From the possible file locations, select the most likely location to fix the issue.

    Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue
       4.1 Look at existing test files in the repository to understand the test format/structure.
       4.2 Create a minimal reproduction script that reproduces the located issue.
       4.3 Run the reproduction script to confirm you are reproducing the issue.
       4.4 Adjust the reproduction script as necessary.

    Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
       5.1 State clearly what the problem is.
       5.2 State clearly where the problem is located.
       5.3 State clearly how the test reproduces the issue.
       5.4 State clearly the best practices to take into account in the fix.
       5.5 State clearly how to fix the problem.

    Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution
       6.1 Make minimal, focused changes to fix the issue.

    Phase 7. VERIFICATION: Test your implementation thoroughly
       7.1 Run your reproduction script to verify the fix works.
       7.2 Add edge cases to your test script to ensure comprehensive coverage.
       7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

    Phase 8. FINAL REVIEW: Carefully re-read the problem description and compare your changes.
       8.1 Ensure you've fully addressed all requirements.
       8.2 Run any tests in the repository related to:
         8.2.1 The issue you are fixing
         8.2.2 The files you modified
         8.2.3 The functions you changed
       8.3 If any tests fail, revise your implementation until all tests pass.

    Be thorough in your exploration, testing, and reasoning. It's fine if \
    your thinking process is lengthy - quality and completeness are more \
    important than brevity.
""")


def build_agent(llm, *, enable_condenser: bool = True):
    """Return an ``Agent`` configured with the default file/terminal tools.

    When ``enable_condenser`` is True (the default), wraps the agent with an
    ``LLMSummarizingCondenser`` that truncates conversation history after 240
    events — matching the official OpenHands evaluation defaults.
    """
    from openhands.sdk import Agent
    from openhands.tools.preset.default import get_default_tools

    tools = get_default_tools(enable_browser=False)

    condenser = None
    if enable_condenser:
        try:
            from openhands.sdk.context.condenser import LLMSummarizingCondenser
            condenser = LLMSummarizingCondenser(
                llm=llm,
                max_size=240,
                keep_first=2,
            )
        except Exception:
            logger.warning("Failed to create condenser, proceeding without one")

    return Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={
            "cli_mode": True,
            "extra_instructions": SWE_BENCH_SYSTEM_SUFFIX,
        },
        condenser=condenser,
    )


# ─── Driver ─────────────────────────────────────────────────────────────


# Sent when the SDK's StuckDetector fires (repeating/alternating action-
# observation loops — see docs/openhands-integration.md investigation notes).
STUCK_NUDGE_MESSAGE = (
    "You appear to be stuck: repeating the same action without making "
    "progress. Stop repeating that exact tool call. Re-read the problem "
    "statement and double-check you are editing the actual source file "
    "where the reported bug lives — not a scratch/test file you created to "
    "explore the issue. If you already understand the fix, make the edit "
    "directly."
)

# Sent when the agent replies with a plain-text message (no tool calls)
# instead of using its tools. Without this, the SDK sets FINISHED after
# a single content-only turn, ending the run before the agent ever
# explored the codebase.
FAKE_USER_RESPONSE = (
    "Please continue working on the task on whatever approach you think is "
    "suitable.\n"
    "When you think you have solved the question, please use the finish tool "
    "and include your final answer in the message parameter of the finish "
    "tool.\n"
    "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
)


def _agent_finished_with_finish_action(events) -> bool:
    """Check if the agent's last action was a FinishAction."""
    try:
        from openhands.sdk.event import ActionEvent
        from openhands.sdk.tool.builtins.finish import FinishAction
    except ImportError:
        return False
    for event in reversed(list(events)):
        if isinstance(event, ActionEvent):
            if event.action is not None and isinstance(event.action, FinishAction):
                return True
            return False
    return False


def _agent_sent_message(events) -> bool:
    """Check if the agent's last event was a message (not a tool call)."""
    try:
        from openhands.sdk.event import ActionEvent, MessageEvent
    except ImportError:
        return False
    for event in reversed(list(events)):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            return False
    return False


def run_with_stuck_retries(
    conv,
    *,
    max_stuck_retries: int = 2,
    max_fake_responses: int = 10,
    instance_id: str = "?",
) -> int:
    """Run ``conv`` to completion with two recovery mechanisms.

    1. **Fake user responses** — when the agent sends a content-only message
       (no tool calls), the SDK sets FINISHED. This function detects that
       case and sends a nudge to keep the agent working (matching the official
       OpenHands evaluation harness behavior).

    2. **Stuck retries** — when the SDK's StuckDetector fires (repeating
       action-observation loops), sends ``STUCK_NUDGE_MESSAGE`` to break out.

    Returns the total number of nudges sent (both types combined).
    """
    from openhands.sdk import ConversationExecutionStatus

    total_nudges = 0
    fake_responses_sent = 0

    while True:
        conv.run()
        status = conv.state.execution_status

        # Handle STUCK status
        if status == ConversationExecutionStatus.STUCK:
            if total_nudges < max_stuck_retries:
                total_nudges += 1
                logger.warning(
                    "%s: stuck pattern detected, sending nudge (%d)",
                    instance_id, total_nudges,
                )
                conv.send_message(STUCK_NUDGE_MESSAGE)
                continue
            else:
                logger.warning("%s: stuck, max retries exhausted", instance_id)
                break

        # Handle FINISHED status — check if agent actually called finish
        if status == ConversationExecutionStatus.FINISHED:
            try:
                events = list(conv.state.events)
            except Exception:
                break

            if _agent_finished_with_finish_action(events):
                break

            if not _agent_sent_message(events):
                break

            if fake_responses_sent >= max_fake_responses:
                logger.warning(
                    "%s: max fake responses (%d) reached, stopping",
                    instance_id, max_fake_responses,
                )
                break

            fake_responses_sent += 1
            total_nudges += 1
            logger.info(
                "%s: agent sent message without tool calls, sending fake "
                "user response (%d/%d)",
                instance_id, fake_responses_sent, max_fake_responses,
            )
            msg = FAKE_USER_RESPONSE
            if fake_responses_sent >= 2:
                msg += (
                    'If you want to give up, use the "finish" tool to '
                    "finish the interaction.\n"
                )
            conv.send_message(msg)
            continue

        # Any other status (ERROR, PAUSED, etc.) — stop
        break

    return total_nudges


def solve_one(
    problem: Problem,
    *,
    backend: str,
    backend_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 100,
    max_stuck_retries: int = 2,
    max_fake_responses: int = 10,
    enable_condenser: bool = True,
    cache_dir: Path | None = None,
    label: str | None = None,
) -> Prediction:
    """Run one OpenHands agent against one problem, return a Prediction."""
    from openhands.sdk import Conversation

    label = label or f"{backend}+default"
    backend_kwargs = backend_kwargs or {}
    t0 = time.monotonic()

    llm = None
    try:
        with checked_out_repo(problem, cache_dir=cache_dir) as ws:
            llm = build_llm(backend, **backend_kwargs)
            agent = build_agent(llm, enable_condenser=enable_condenser)
            conv = Conversation(
                agent=agent,
                workspace=str(ws),
                max_iteration_per_run=max_iterations,
                visualizer=None,
            )
            conv.send_message(_format_user_prompt(problem, ws=ws))
            nudges = run_with_stuck_retries(
                conv,
                max_stuck_retries=max_stuck_retries,
                max_fake_responses=max_fake_responses,
                instance_id=problem.instance_id,
            )

            patch = capture_patch(ws)
            metrics = _extract_metrics(conv)
            session_stats = _pie_session_stats(llm)
            if session_stats:
                logger.info(
                    "%s: pie session — %d calls, %d/%d prompt tokens prefilled, modes=%s",
                    problem.instance_id,
                    session_stats.get("num_calls", 0),
                    session_stats.get("prompt_tokens_prefilled", 0),
                    session_stats.get("prompt_tokens_rendered", 0),
                    session_stats.get("modes", {}),
                )
            return Prediction(
                instance_id=problem.instance_id,
                model_name_or_path=label,
                model_patch=patch,
                wall_clock_s=time.monotonic() - t0,
                agent_iterations=_count_iterations(conv),
                stuck_retries=nudges,
                pie_session=session_stats,
                **metrics,
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
            pie_session=_pie_session_stats(llm),
        )
    finally:
        # A leaked session pins its KV snapshot server-side; deletion is
        # idempotent and must run on the error path too.
        if llm is not None and hasattr(llm, "close_pie_session"):
            llm.close_pie_session()


def _pie_session_stats(llm) -> dict[str, Any]:
    """Session telemetry from a PieLLM in session mode, else empty."""
    if llm is not None and getattr(llm, "pie_session", False):
        try:
            return llm.pie_session_summary()
        except Exception:
            return {}
    return {}


def _format_user_prompt(
    problem: Problem, *, use_cwd: bool = False, ws: Path | None = None
) -> str:
    repo_dir = problem.repo.split("/")[-1]
    if ws is not None:
        # The real checkout path. The `/workspace/<repo>` fallback is a
        # leftover from the dockerized official harness and does not exist
        # here; pointing the model at a nonexistent directory makes small
        # models loop on `ls /workspace/...` failures at t=0 until the
        # iteration budget runs out (job 18783763, 0-byte patches).
        location = str(ws)
    elif use_cwd:
        location = (
            "the current working directory (use `pwd` to see the absolute "
            "path, and relative paths like `./django/` to explore)"
        )
    else:
        location = f"/workspace/{repo_dir}"
    return (
        f"I have access to a python code repository in {location} . "
        f"You can explore and modify files using "
        f"the available tools. Consider the following issue description:\n\n"
        f"<issue_description>\n{problem.problem_statement}\n</issue_description>\n\n"
        f"Can you help me implement the necessary changes to the repository "
        f"so that the requirements specified in the <issue_description> are met?"
    )


def _count_iterations(conv) -> int:
    """Best-effort count of agent steps from the conversation event log."""
    try:
        return sum(1 for _ in conv.state.events)
    except Exception:
        return 0


def _extract_metrics(conv) -> dict[str, Any]:
    """Pull token counts and per-call latencies from a finished conversation."""
    try:
        m = conv.state.stats.get_combined_metrics()
        usage = m.accumulated_token_usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        latencies = [r.latency for r in m.response_latencies]
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "num_llm_calls": len(m.token_usages),
            "response_latencies": latencies,
        }
    except Exception:
        return {}


# ─── Top-level run loop ─────────────────────────────────────────────────


@dataclass
class RunOptions:
    backend: str = "test"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    subset_size: int = DEFAULT_SUBSET_N
    subset_indices: list[int] | None = None
    instance_ids: list[str] | None = None  # if set, takes precedence over subset
    max_iterations: int = 100
    max_stuck_retries: int = 2
    max_fake_responses: int = 10
    enable_condenser: bool = True
    cache_dir: Path | None = None
    output_path: Path = Path("predictions.jsonl")
    label: str | None = None
    resume: bool = False
    concurrency: int = 1  # instances solved in parallel; 1 == serial (unchanged)


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

    done_ids: set[str] = set()
    if options.resume and options.output_path.exists():
        with options.output_path.open() as existing:
            for line in existing:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["instance_id"])
        if done_ids:
            logger.info("resume: skipping %d already-completed instances", len(done_ids))

    todo: list[Problem] = []
    for row in selected:
        problem = Problem.from_row(row)
        if problem.instance_id in done_ids:
            logger.info("skipping %s (already completed)", problem.instance_id)
            continue
        todo.append(problem)

    write_lock = threading.Lock()
    abort = threading.Event()  # tripped when the Pie server goes down

    def _solve(problem: Problem) -> Prediction:
        # Don't begin new work once the server has been declared down.
        if abort.is_set():
            raise _Aborted(problem.instance_id)
        logger.info("solving %s (%s)", problem.instance_id, problem.repo)
        return solve_one(
            problem,
            backend=options.backend,
            backend_kwargs=options.backend_kwargs,
            max_iterations=options.max_iterations,
            max_stuck_retries=options.max_stuck_retries,
            max_fake_responses=options.max_fake_responses,
            enable_condenser=options.enable_condenser,
            cache_dir=options.cache_dir,
            label=options.label,
        )

    # concurrency == 1 keeps the original strictly-serial behavior (a pool of
    # one worker drains futures in submission order); the fan-out only matters
    # for N > 1, where the runtime batches the concurrent decode requests.
    workers = max(1, options.concurrency)
    with options.output_path.open("a" if options.resume else "w") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_solve, p): p for p in todo}
            for fut in as_completed(futures):
                problem = futures[fut]
                try:
                    pred = fut.result()
                except _Aborted:
                    continue
                except Exception as e:  # a worker crashed — log and keep going
                    logger.error(
                        "instance %s crashed: %s",
                        problem.instance_id, e, exc_info=True,
                    )
                    continue

                if "ConnectionRefusedError" in pred.error:
                    logger.error(
                        "Pie server is down (ConnectionRefusedError for %s). "
                        "Aborting run — use --resume to continue after restart.",
                        problem.instance_id,
                    )
                    abort.set()  # stop *new* work; in-flight workers still finish
                    continue

                with write_lock:  # the only place the output file is touched
                    f.write(pred.to_jsonl() + "\n")
                    f.flush()
                logger.info(
                    "  → %s %s (%.1fs, %d iters, %d-byte patch%s%s)",
                    problem.instance_id,
                    "ok" if not pred.error else "ERROR",
                    pred.wall_clock_s, pred.agent_iterations,
                    len(pred.model_patch),
                    f", {pred.stuck_retries} stuck-retries" if pred.stuck_retries else "",
                    f", error={pred.error!r}" if pred.error else "",
                )

    if abort.is_set():
        raise SystemExit(42)  # preserve the existing abort contract for --resume
    return options.output_path
