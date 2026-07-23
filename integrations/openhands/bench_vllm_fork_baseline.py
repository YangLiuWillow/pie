"""The fair vLLM baseline for fork test-time scaling.

Replicates the SAME mid-trajectory K-way branch structure that the Pie
`openhands-agent` inferlet forks at ~0 (see `_solve_one_agent_fork` in
`benchmarks/swe_bench.py`), but on the LiteLLM -> vLLM path with prefix caching
ON — so we can measure what the incumbent must pay for the same branching.

Per instance:
  1. TRUNK — run a real OpenHands agent (litellm backend) to `branch_at_step`
     steps against a checked-out repo. This builds the shared, GENERATED
     branch-point context (the reuse the Pie fork exploits).
  2. BRANCH — for k in 0..K-1, build a fresh Conversation on its OWN workspace
     copy (`cp -a` of the trunk tree) seeded with a deep-copy of the trunk's
     event history (mirrors `LocalConversation.fork()`, but with a chosen
     workspace + fresh LLM so per-branch edits AND per-branch token metrics are
     isolated). Run each branch to completion, concurrently (threads; vLLM
     batches them — the fair analogue of Pie's concurrent forks).
  3. MEASURE — each branch's prompt_tokens (re-prefill proxy), first-call
     latency (TTFT proxy), wall, and captured patch. Aggregate the re-prefill
     tax across branches and emit a Prediction whose `_metadata.candidates`
     schema matches the Pie side, so `bestofk.py split/combine` scores it
     unchanged.

Design rationale + the honest accounting (prefix cache hits the shared trunk on
branches 2..K, so the win is re-transmission + eviction-resistance +
compounding continuation, NOT the initial branch-point prefill):
  integrations/openhands/docs/VLLM_FORK_BASELINE_DESIGN.md

Env / CLI:
  INSTANCE_IDS  comma-separated SWE-bench instance ids
  BACKEND       "litellm" (default) or "test" (CPU plumbing check, no vLLM)
  MODEL         litellm model, e.g. openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
  BASE_URL      vLLM OpenAI endpoint, e.g. http://localhost:18000/v1
  BRANCH_AT_STEP, NUM_BRANCHES, TEMPERATURE, TOP_P, MAX_ITERS
  OUTPUT        predictions jsonl path
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.swe_bench import (
    Problem, checked_out_repo, build_llm, build_agent, capture_patch,
    run_with_stuck_retries, _extract_metrics, _count_iterations,
    _format_user_prompt, Prediction,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vllm_fork_baseline")

BACKEND = os.environ.get("BACKEND", "litellm")
MODEL = os.environ.get("MODEL", "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:18000/v1")
API_KEY = os.environ.get("LITELLM_API_KEY", "dummy")
BRANCH_AT_STEP = int(os.environ.get("BRANCH_AT_STEP", "5"))
NUM_BRANCHES = int(os.environ.get("NUM_BRANCHES", "3"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "60"))
OUTPUT = Path(os.environ.get("OUTPUT", "predictions/vllm_fork_baseline.jsonl"))
LABEL = os.environ.get("LABEL", f"vllm-fork-baseline+{MODEL.split('/')[-1]}")


def _load(instance_id: str) -> Problem:
    from datasets import load_dataset
    for row in load_dataset("princeton-nlp/SWE-bench_Verified", split="test"):
        if row["instance_id"] == instance_id:
            return Problem.from_row(row)
    raise SystemExit(f"instance {instance_id!r} not found")


def _make_llm():
    """A fresh LLM per branch — avoids the shared-`_prompt_cache_key` clobber
    that `LocalConversation.fork()` guards against, and keeps per-branch token
    metrics independent."""
    if BACKEND == "test":
        return build_llm("test")
    from pydantic import SecretStr
    kwargs: dict[str, Any] = {
        "model": MODEL, "base_url": BASE_URL, "api_key": SecretStr(API_KEY),
        "temperature": TEMPERATURE, "top_p": TOP_P,
    }
    return build_llm("litellm", **kwargs)


def _new_conversation(llm, workspace: str, max_iters: int):
    from openhands.sdk import Conversation
    agent = build_agent(llm, enable_condenser=True)
    return Conversation(
        agent=agent, workspace=workspace,
        max_iteration_per_run=max_iters, visualizer=None,
    )


def _seed_from_trunk(branch_conv, trunk_conv) -> None:
    """Copy the trunk's event history + runtime state into a fresh branch
    conversation — the same deep-copy `LocalConversation.fork()` does, but into
    a conversation we built on the branch's own workspace. On `run()` the branch
    resumes with full memory of the trunk (its first LLM call re-sends the whole
    branch-point context -> the vLLM re-prefill/cache-hit we measure)."""
    for event in trunk_conv.state.events:
        branch_conv._state.events.append(event.model_copy(deep=True))
    try:
        branch_conv._state.activated_knowledge_skills = list(
            trunk_conv._state.activated_knowledge_skills
        )
        branch_conv._state.agent_state = copy.deepcopy(trunk_conv._state.agent_state)
    except Exception:
        logger.warning("could not copy runtime state (non-fatal)")


def _fork_workspace(trunk_ws: Path, k: int) -> Path:
    """`cp -a --reflink=auto` the trunk tree (incl. uncommitted edits) into an
    isolated per-branch workspace. Branch 0 reuses the trunk tree itself."""
    if k == 0:
        return trunk_ws
    dst = Path(f"{str(trunk_ws).rstrip('/')}__vfork{k}")
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(trunk_ws), str(dst)],
        check=True,
    )
    return dst


def _run_branch(k: int, trunk_conv, trunk_ws: Path) -> dict[str, Any]:
    ws = _fork_workspace(trunk_ws, k)
    llm = _make_llm()
    conv = _new_conversation(llm, str(ws), MAX_ITERS)
    _seed_from_trunk(conv, trunk_conv)
    t0 = time.monotonic()
    nudges = run_with_stuck_retries(conv, instance_id=f"branch{k}")
    wall = time.monotonic() - t0
    m = _extract_metrics(conv)
    patch = capture_patch(ws)
    lat = m.get("response_latencies", []) or []
    return {
        "workspace_id": str(k),
        "model_patch": patch,
        "finished": True,
        "steps": _count_iterations(conv),
        "total_wall_s": round(wall, 3),
        "prompt_tokens": m.get("prompt_tokens", 0),
        "completion_tokens": m.get("completion_tokens", 0),
        "num_llm_calls": m.get("num_llm_calls", 0),
        "first_latency_s": round(lat[0], 4) if lat else 0.0,
        "nudges": nudges,
    }


def solve_instance(instance_id: str) -> Prediction:
    problem = _load(instance_id)
    t_start = time.monotonic()
    with checked_out_repo(problem) as ws:
        # --- TRUNK: run to the branch point ---
        trunk_llm = _make_llm()
        trunk_conv = _new_conversation(trunk_llm, str(ws), BRANCH_AT_STEP)
        trunk_conv.send_message(_format_user_prompt(problem, ws=ws))
        t_trunk = time.monotonic()
        trunk_conv.run()  # capped at BRANCH_AT_STEP iterations
        trunk_wall = time.monotonic() - t_trunk
        trunk_m = _extract_metrics(trunk_conv)
        logger.info(
            "%s: trunk ran to branch point in %d events (%.1fs, %d prompt_tok)",
            instance_id, len(list(trunk_conv.state.events)),
            trunk_wall, trunk_m.get("prompt_tokens", 0),
        )

        # --- BRANCH: K concurrent continuations ---
        with ThreadPoolExecutor(max_workers=NUM_BRANCHES) as ex:
            futures = [ex.submit(_run_branch, k, trunk_conv, ws)
                       for k in range(NUM_BRANCHES)]
            candidates = [f.result() for f in futures]
    candidates.sort(key=lambda c: int(c["workspace_id"]))

    branch_prefill = sum(c["prompt_tokens"] for c in candidates)
    branch_wall_concurrent = max((c["total_wall_s"] for c in candidates), default=0.0)
    branch_wall_serial = sum(c["total_wall_s"] for c in candidates)
    n_nonempty = sum(1 for c in candidates if c["model_patch"])
    n_distinct = len({c["model_patch"] for c in candidates})
    logger.info(
        "%s: %d branches, %d non-empty, %d distinct; branch prefill=%d tok, "
        "wall concurrent=%.1fs serial=%.1fs",
        instance_id, len(candidates), n_nonempty, n_distinct,
        branch_prefill, branch_wall_concurrent, branch_wall_serial,
    )
    primary = candidates[0] if candidates else {"model_patch": "", "steps": 0}
    pred = Prediction(
        instance_id=instance_id,
        model_name_or_path=LABEL,
        model_patch=primary["model_patch"],
        wall_clock_s=time.monotonic() - t_start,
        agent_iterations=primary.get("steps", 0),
        prompt_tokens=trunk_m.get("prompt_tokens", 0) + branch_prefill,
        candidates=candidates,
    )
    return pred


def main() -> int:
    ids = [s for s in os.environ.get("INSTANCE_IDS", "").split(",") if s.strip()]
    if not ids:
        ids = ["django__django-14373"]
    print(f"=== vLLM fork baseline: {len(ids)} instance(s), K={NUM_BRANCHES}, "
          f"branch_at_step={BRANCH_AT_STEP}, temp={TEMPERATURE}, backend={BACKEND} ===")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as fh:
        for iid in ids:
            print(f"\n----- {iid} -----")
            try:
                pred = solve_instance(iid.strip())
            except Exception as e:
                logger.exception("instance %s failed", iid)
                pred = Prediction(instance_id=iid.strip(), model_name_or_path=LABEL,
                                  model_patch="", error=f"{type(e).__name__}: {e}")
            fh.write(pred.to_jsonl() + "\n")
            fh.flush()
    print(f"\nPredictions: {OUTPUT}  (split with bestofk.py, score with the SWE-bench grader)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
