"""Smoke test for the SWE-Bench driver plumbing.

Two sub-tests:

  * **harness_unit** — verifies dataset loading, subset selection, and the
    backend factory. Hermetic; no network needed if HF dataset is cached.

  * **drive_one_problem** — actually clones a small repo and runs the driver
    against one problem with TestLLM (no real model). Verifies the plumbing
    end to end: workspace setup → agent.run → git diff capture → JSONL
    output. Gated by SWE_BENCH_NETWORK=1 because it does ~30s of git clone
    plus an HF dataset download on first run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks import swe_bench


# ─── Unit-ish: deterministic subset + backend factory ───────────────────


def test_deterministic_subset_is_stable():
    a = swe_bench.deterministic_subset_indices(500, 50)
    b = swe_bench.deterministic_subset_indices(500, 50)
    assert a == b
    assert len(a) == 50
    assert len(set(a)) == 50
    assert all(0 <= i < 500 for i in a)


def test_deterministic_subset_seed_changes_selection():
    a = swe_bench.deterministic_subset_indices(500, 50, seed=1)
    b = swe_bench.deterministic_subset_indices(500, 50, seed=2)
    assert a != b


def test_build_llm_test_backend():
    llm = swe_bench.build_llm("test")
    from openhands.sdk.testing import TestLLM
    assert isinstance(llm, TestLLM)


def test_build_llm_pie_backend_returns_piellm():
    from pie_openhands import PieLLM
    llm = swe_bench.build_llm("pie", model="default", pie_render_strategy="raw_concat")
    assert isinstance(llm, PieLLM)
    assert llm.pie_inferlet == "openhands-completion@0.1.0"
    assert llm.pie_render_strategy == "raw_concat"


def test_problem_from_row_extracts_needed_fields():
    row = {
        "instance_id": "x__y-1",
        "repo": "x/y",
        "base_commit": "deadbeef",
        "problem_statement": "fix it",
        "hints_text": "look at foo.py",
        "patch": "irrelevant",          # ground truth — driver must not use
    }
    p = swe_bench.Problem.from_row(row)
    assert p.instance_id == "x__y-1"
    assert p.repo == "x/y"
    assert p.base_commit == "deadbeef"
    # Ground-truth fields must not leak onto the Problem.
    assert not hasattr(p, "patch")


def test_prediction_jsonl_matches_swebench_grader_schema():
    pred = swe_bench.Prediction(
        instance_id="x__y-1",
        model_name_or_path="test+nothing",
        model_patch="diff --git a/foo b/foo\n",
        wall_clock_s=1.234567,
        agent_iterations=3,
    )
    parsed = json.loads(pred.to_jsonl())
    # These three fields are what swebench.harness.run_evaluation reads.
    assert set(parsed) >= {"instance_id", "model_name_or_path", "model_patch"}
    # Metadata is under a sentinel key, so the grader ignores it.
    assert "_metadata" in parsed
    assert parsed["_metadata"]["agent_iterations"] == 3


# ─── Network/IO smoke ────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("SWE_BENCH_NETWORK") != "1",
    reason="SWE_BENCH_NETWORK=1 not set (this test clones a real repo)",
)
def test_drive_one_problem_end_to_end(tmp_path):
    """End-to-end: clone a real SWE-Bench repo, run TestLLM, capture patch.

    TestLLM is scripted to never make edits, so the captured patch is empty.
    The point is that the plumbing — clone, checkout, agent.run, git diff,
    JSONL serialization — all runs without crashing.

    Picks a small repo (psf/requests) for speed; if it ever drops out of
    SWE-Bench Verified, swap to another small one.
    """
    # Find a problem from a small repo for speed.
    ds = swe_bench.load_verified()
    rows = [r for r in ds if r["repo"] == "psf/requests"]
    assert rows, "psf/requests not in SWE-Bench Verified — pick another repo"

    out = tmp_path / "predictions.jsonl"
    swe_bench.run(swe_bench.RunOptions(
        backend="test",
        instance_ids=[rows[0]["instance_id"]],
        output_path=out,
        cache_dir=tmp_path / "cache",
        max_iterations=3,
        label="test+noop",
    ))
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    pred = json.loads(lines[0])
    assert pred["instance_id"] == rows[0]["instance_id"]
    assert pred["model_name_or_path"] == "test+noop"
    # TestLLM didn't edit anything, so the patch is empty.
    assert pred["model_patch"] == ""
