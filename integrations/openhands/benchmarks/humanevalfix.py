"""HumanEvalFix harness — a lighter smoke-test benchmark than SWE-Bench.

SWE-Bench exercises the full stack (real repo clone, multi-file navigation,
Docker-based scoring) but that makes it slow to iterate on and, per the
Phase-1 investigation, its failures are dominated by model-capability limits
(wrong-file fixation, long-context degradation) rather than the tool-calling
plumbing this integration actually needs to validate.

HumanEvalFix (from ``bigcode/humanevalpack``) is a single-file "this function
has one injected bug, fix it" task: 164 problems, no repo clone, and scoring
is a local subprocess running the held-out asserts — no Docker. That makes it
a fast, self-contained way to check "is the agent's tool-calling loop through
PieLLM actually working" without waiting on SWE-Bench's per-problem latency
or its Docker-gated scoring step.

Unlike ``benchmarks.swe_bench``, drive and score happen in the same pass:
``solve_one`` returns a ``Result`` with ``passed`` already computed.

Security note: scoring executes the model's generated code (plus the
dataset's held-out test) in a subprocess with a timeout. This is standard
practice for code-execution benchmarks (HumanEval's own reference harness
does the same) and is fine for locally-run, non-adversarial smoke tests, but
this module should not be pointed at untrusted input.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ─── Dataset access ─────────────────────────────────────────────────────


DEFAULT_DATASET = "bigcode/humanevalpack"
DEFAULT_CONFIG = "python"
DEFAULT_SPLIT = "test"
SUBSET_SEED = 20260705    # frozen; do not change between runs
DEFAULT_SUBSET_N = 20     # full set is only 164; keep the default run short

SOLUTION_FILENAME = "solution.py"


def load_humanevalfix(dataset: str = DEFAULT_DATASET, config: str = DEFAULT_CONFIG,
                       split: str = DEFAULT_SPLIT):
    """Return the HF Dataset object. Caches under ``~/.cache/huggingface``."""
    from datasets import load_dataset
    return load_dataset(dataset, config, split=split)


def deterministic_subset_indices(n_total: int, n_subset: int, seed: int = SUBSET_SEED) -> list[int]:
    """Deterministic ``n_subset`` indices into a ``n_total``-element dataset."""
    n_subset = min(n_subset, n_total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_subset))


# ─── Problem & result types ──────────────────────────────────────────────


@dataclass
class Problem:
    """A single HumanEvalPack row, narrowed to what the "fix" task needs."""
    task_id: str
    entry_point: str
    docstring: str
    import_stmt: str
    test_setup: str
    declaration: str
    buggy_solution: str
    test: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Problem":
        return cls(
            task_id=row["task_id"],
            entry_point=row["entry_point"],
            docstring=row["docstring"],
            import_stmt=row.get("import") or "",
            test_setup=row.get("test_setup") or "",
            declaration=row["declaration"],
            buggy_solution=row["buggy_solution"],
            test=row["test"],
        )

    @property
    def starter_code(self) -> str:
        """The buggy program to seed ``solution.py`` with."""
        body = self.declaration + self.buggy_solution
        if self.import_stmt.strip():
            return f"{self.import_stmt}\n\n{body}"
        return body


@dataclass
class Result:
    task_id: str
    model_name_or_path: str
    passed: bool
    fixed_solution: str
    wall_clock_s: float = 0.0
    agent_iterations: int = 0
    stuck_retries: int = 0
    error: str = ""
    score_detail: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_llm_calls: int = 0
    response_latencies: list[float] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "model_name_or_path": self.model_name_or_path,
            "passed": self.passed,
            "fixed_solution": self.fixed_solution,
            "_metadata": {
                "wall_clock_s": round(self.wall_clock_s, 3),
                "agent_iterations": self.agent_iterations,
                "stuck_retries": self.stuck_retries,
                "error": self.error,
                "score_detail": self.score_detail,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "num_llm_calls": self.num_llm_calls,
                "response_latencies": [round(l, 4) for l in self.response_latencies],
            },
        })


# ─── Workspace setup ─────────────────────────────────────────────────────


@contextmanager
def solo_file_workspace(problem: Problem) -> Iterator[Path]:
    """A plain temp dir containing just ``solution.py`` — no git, no clone."""
    with tempfile.TemporaryDirectory(prefix=f"heval-{problem.task_id.replace('/', '_')}-") as tmp:
        ws = Path(tmp)
        (ws / SOLUTION_FILENAME).write_text(problem.starter_code)
        yield ws


# ─── Scoring (local subprocess, no Docker) ───────────────────────────────


def score_fix(problem: Problem, file_text: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    """Run ``file_text`` plus the held-out asserts in a subprocess.

    Returns ``(passed, detail)`` where ``detail`` is empty on success and a
    truncated stderr/stdout tail on failure (including on timeout).
    """
    parts = [file_text]
    if problem.test_setup.strip():
        parts.append(problem.test_setup)
    parts.append(problem.test)
    program = "\n\n".join(parts)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name

    try:
        res = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if res.returncode == 0:
            return True, ""
        detail = (res.stderr or res.stdout)[-2000:]
        return False, detail
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    finally:
        Path(path).unlink(missing_ok=True)


# ─── Agent construction ──────────────────────────────────────────────────


HUMANEVALFIX_SYSTEM_SUFFIX = textwrap.dedent(f"""\
    You are given a single Python file, `{SOLUTION_FILENAME}`, containing one
    function with an injected bug. Read the docstring to understand the
    intended behavior, find the bug, and fix it with the minimal edit
    necessary.

    Constraints:
      * Do not change the function's name or signature.
      * Do not add a `__main__` block or try to run/test the file yourself —
        a held-out test suite will be run against it after you finish.
      * Keep the fix inside the one file already present; do not create new
        files.
""")


def build_agent(llm):
    """Return an ``Agent`` configured with the default file/terminal tools."""
    from openhands.sdk import Agent
    from openhands.tools.preset.default import get_default_tools

    tools = get_default_tools(enable_browser=False)
    return Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": False, "extra_instructions": HUMANEVALFIX_SYSTEM_SUFFIX},
    )


def _format_user_prompt(problem: Problem) -> str:
    return (
        f"The file {SOLUTION_FILENAME} contains a function called "
        f"`{problem.entry_point}` with an intentional bug.\n\n"
        f"Intended behavior (from its docstring):\n{problem.docstring}\n\n"
        "Find the bug and fix it with the minimal edit necessary."
    )


# ─── Driver ─────────────────────────────────────────────────────────────


def solve_one(
    problem: Problem,
    *,
    backend: str,
    backend_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 15,
    max_stuck_retries: int = 2,
    max_fake_responses: int = 10,
    score_timeout_s: float = 10.0,
    label: str | None = None,
) -> Result:
    """Run one OpenHands agent against one problem, drive + score in one pass."""
    from openhands.sdk import Conversation
    from benchmarks.swe_bench import build_llm, run_with_stuck_retries

    label = label or f"{backend}+default"
    backend_kwargs = backend_kwargs or {}
    t0 = time.monotonic()

    try:
        with solo_file_workspace(problem) as ws:
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
                max_fake_responses=max_fake_responses,
                instance_id=problem.task_id,
            )

            fixed_solution = (ws / SOLUTION_FILENAME).read_text()
            passed, detail = score_fix(problem, fixed_solution, timeout_s=score_timeout_s)
            from benchmarks.swe_bench import _extract_metrics
            metrics = _extract_metrics(conv)
            return Result(
                task_id=problem.task_id,
                model_name_or_path=label,
                passed=passed,
                fixed_solution=fixed_solution,
                wall_clock_s=time.monotonic() - t0,
                agent_iterations=_count_iterations(conv),
                stuck_retries=stuck_retries,
                score_detail=detail,
                **metrics,
            )
    except Exception as e:
        logger.exception("solve_one failed for %s", problem.task_id)
        return Result(
            task_id=problem.task_id,
            model_name_or_path=label,
            passed=False,
            fixed_solution="",
            wall_clock_s=time.monotonic() - t0,
            agent_iterations=0,
            error=f"{type(e).__name__}: {e}",
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
    task_ids: list[str] | None = None  # if set, takes precedence over subset
    max_iterations: int = 15
    max_stuck_retries: int = 2
    score_timeout_s: float = 10.0
    output_path: Path = Path("humanevalfix_results.jsonl")
    label: str | None = None


def run(options: RunOptions) -> Path:
    """Drive + score the full subset, writing one JSONL row per problem.

    Returns the output path.
    """
    ds = load_humanevalfix()
    rows = list(ds)

    if options.task_ids:
        wanted = set(options.task_ids)
        selected = [r for r in rows if r["task_id"] in wanted]
        missing = wanted - {r["task_id"] for r in selected}
        if missing:
            raise ValueError(f"unknown task_ids: {sorted(missing)}")
    else:
        indices = options.subset_indices or deterministic_subset_indices(
            len(rows), options.subset_size
        )
        selected = [rows[i] for i in indices]

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    n_passed = 0
    with options.output_path.open("w") as f:
        for row in selected:
            problem = Problem.from_row(row)
            logger.info("solving %s (%s)", problem.task_id, problem.entry_point)
            result = solve_one(
                problem,
                backend=options.backend,
                backend_kwargs=options.backend_kwargs,
                max_iterations=options.max_iterations,
                max_stuck_retries=options.max_stuck_retries,
                score_timeout_s=options.score_timeout_s,
                label=options.label,
            )
            f.write(result.to_jsonl() + "\n")
            f.flush()
            n_passed += int(result.passed)
            logger.info(
                "  → %s (%.1fs, %d iters%s%s)",
                "PASS" if result.passed else ("ERROR" if result.error else "FAIL"),
                result.wall_clock_s, result.agent_iterations,
                f", {result.stuck_retries} stuck-retries" if result.stuck_retries else "",
                f", error={result.error!r}" if result.error else "",
            )

    logger.info(
        "Resolved-rate: %d/%d (%.1f%%)",
        n_passed, len(selected), 100 * n_passed / len(selected) if selected else 0.0,
    )
    return options.output_path
