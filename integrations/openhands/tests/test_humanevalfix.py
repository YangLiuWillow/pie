"""Smoke test for the HumanEvalFix driver plumbing.

Three sub-tests:

  * **harness_unit** — deterministic subset selection, ``Problem.from_row``,
    and the scoring subprocess (against both the dataset's own canonical fix
    and its buggy version). Hermetic; no model, no network needed beyond the
    one-time HF dataset download (gated by HUMANEVALFIX_NETWORK, same
    convention as test_swe_bench_smoke.py).

  * **drive_one_problem** — runs the real driver against one problem with
    TestLLM (no real model, so the file is never fixed). Verifies the
    plumbing end to end: workspace setup → agent.run → file read-back →
    scoring → JSONL output. Gated the same way.
"""

from __future__ import annotations

import json
import os

import pytest

from benchmarks import humanevalfix


# ─── Unit-ish: deterministic subset + Problem parsing ────────────────────


def test_deterministic_subset_is_stable():
    a = humanevalfix.deterministic_subset_indices(164, 20)
    b = humanevalfix.deterministic_subset_indices(164, 20)
    assert a == b
    assert len(a) == 20
    assert len(set(a)) == 20
    assert all(0 <= i < 164 for i in a)


def test_deterministic_subset_seed_changes_selection():
    a = humanevalfix.deterministic_subset_indices(164, 20, seed=1)
    b = humanevalfix.deterministic_subset_indices(164, 20, seed=2)
    assert a != b


def test_problem_from_row_extracts_needed_fields():
    row = {
        "task_id": "Python/0",
        "entry_point": "foo",
        "docstring": "does foo things",
        "import": "",
        "test_setup": "",
        "declaration": "def foo(x):\n",
        "buggy_solution": "    return x - 1\n",
        "test": "def check(foo):\n    assert foo(1) == 1\ncheck(foo)\n",
        "canonical_solution": "irrelevant",  # ground truth — driver must not use
    }
    p = humanevalfix.Problem.from_row(row)
    assert p.task_id == "Python/0"
    assert p.entry_point == "foo"
    assert p.starter_code == "def foo(x):\n    return x - 1\n"
    assert not hasattr(p, "canonical_solution")


def test_problem_starter_code_includes_import_when_present():
    row = {
        "task_id": "Python/1",
        "entry_point": "bar",
        "docstring": "does bar things",
        "import": "from typing import List",
        "test_setup": "",
        "declaration": "def bar(xs: List[int]) -> int:\n",
        "buggy_solution": "    return sum(xs) - 1\n",
        "test": "",
    }
    p = humanevalfix.Problem.from_row(row)
    assert p.starter_code == (
        "from typing import List\n\n"
        "def bar(xs: List[int]) -> int:\n"
        "    return sum(xs) - 1\n"
    )


# ─── Scoring ─────────────────────────────────────────────────────────────


def _sample_problem() -> humanevalfix.Problem:
    return humanevalfix.Problem(
        task_id="Python/0",
        entry_point="add_one",
        docstring="Return x + 1.",
        import_stmt="",
        test_setup="",
        declaration="def add_one(x):\n",
        buggy_solution="    return x - 1\n",  # bug: should be +1
        test="def check(add_one):\n    assert add_one(1) == 2\n    assert add_one(0) == 1\ncheck(add_one)\n",
    )


def test_score_fix_fails_on_buggy_solution():
    problem = _sample_problem()
    passed, detail = humanevalfix.score_fix(problem, problem.starter_code)
    assert passed is False
    assert detail  # some assertion/traceback text


def test_score_fix_passes_on_correct_solution():
    problem = _sample_problem()
    fixed = "def add_one(x):\n    return x + 1\n"
    passed, detail = humanevalfix.score_fix(problem, fixed)
    assert passed is True
    assert detail == ""


def test_score_fix_handles_syntax_error_gracefully():
    problem = _sample_problem()
    passed, detail = humanevalfix.score_fix(problem, "def add_one(x)\n    return x + 1\n")
    assert passed is False
    assert detail


def test_score_fix_times_out_on_infinite_loop():
    problem = humanevalfix.Problem(
        task_id="Python/hang",
        entry_point="spin",
        docstring="never returns",
        import_stmt="",
        test_setup="",
        declaration="def spin():\n",
        buggy_solution="    while True:\n        pass\n",
        test="spin()\n",
    )
    passed, detail = humanevalfix.score_fix(problem, problem.starter_code, timeout_s=1.0)
    assert passed is False
    assert "timeout" in detail


def test_result_jsonl_roundtrips_expected_fields():
    result = humanevalfix.Result(
        task_id="Python/0",
        model_name_or_path="test+nothing",
        passed=True,
        fixed_solution="def add_one(x):\n    return x + 1\n",
        wall_clock_s=1.234567,
        agent_iterations=3,
        stuck_retries=1,
    )
    parsed = json.loads(result.to_jsonl())
    assert parsed["task_id"] == "Python/0"
    assert parsed["passed"] is True
    assert parsed["_metadata"]["agent_iterations"] == 3
    assert parsed["_metadata"]["stuck_retries"] == 1


# ─── Network/IO smoke ────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("HUMANEVALFIX_NETWORK") != "1",
    reason="HUMANEVALFIX_NETWORK=1 not set (this test downloads the HF dataset)",
)
def test_drive_one_problem_end_to_end(tmp_path):
    """End-to-end: real dataset row, TestLLM, file read-back, scoring, JSONL.

    TestLLM is scripted to never make edits, so the file stays buggy and the
    score is False. The point is that workspace setup → agent.run → file
    read-back → scoring → JSONL output all run without crashing.
    """
    ds = humanevalfix.load_humanevalfix()
    task_id = ds[0]["task_id"]

    out = tmp_path / "results.jsonl"
    humanevalfix.run(humanevalfix.RunOptions(
        backend="test",
        task_ids=[task_id],
        output_path=out,
        max_iterations=3,
        label="test+noop",
    ))
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["task_id"] == task_id
    assert result["model_name_or_path"] == "test+noop"
    # TestLLM didn't edit anything, so the buggy solution is still buggy.
    assert result["passed"] is False
    assert result["_metadata"]["error"] == ""
