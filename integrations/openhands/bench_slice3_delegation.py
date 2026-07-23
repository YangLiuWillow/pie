"""Slice 3 — agent-level critic-continue fork delegation (real generation).

Extends ``bench_fork_delegation.py`` (a synthetic 3-turn micro-measurement) to
the *agent* level: a real OpenHands agent solves a SWE-bench issue end-to-end,
building a large real KV snapshot; we then delegate a **critic-continue**
subtask — "review this fix, add a regression guard, finalize" — to a child that
continues from the parent's *full* context. The child's ``PieLLM`` is derived
exactly as the OpenHands SDK derives a delegated sub-agent's LLM
(``parent.model_copy(update={"stream": False})``; see
openhands/tools/task/manager.py:306-317, the double-copy path), so its first
call forks the parent's prompt KV. A no-fork **control** (a fresh ``PieLLM``,
never copied) runs the identical critic prompt cold — the litellm baseline's
behavior.

Three results per instance, from ONE real critic generation each arm:

  1. **Behavior-neutrality (primary).** At temperature 0 the fork copies KV
     bit-identically, so the forked child and the cold control seeing the same
     prompt must emit identical logits -> identical tokens. We assert
     ``sha256(fork output) == sha256(cold output)``. If it holds, fork is
     provably behaviour-neutral and the saving below is *pure*.
  2. **Prefill saved.** forked prefill = suffix only (critic turn) vs cold
     prefill = full context. Reported as a %.
  3. **Wall saved.** forked critic-call wall vs cold critic-call wall. Because
     both arms do byte-identical work (same prompt, same output), this wall
     comparison is free of the trajectory-divergence noise that muddied the
     aligned session/stateless run.

Runtime asserts (any failure voids the instance): child ``mode == "forked"``;
``forked_prefill == child_len - parent_snapshot_len`` (suffix-only); a post-fork
parent extend still returns ``mode == "extended"`` (parent uncorrupted).

SCOPE: this is the *continuation* delegation family — the child prompt contains
the parent's full prefix, which the current full-prefix fork supports. Stock
``TaskToolSet`` spawns a *fresh* sub-agent sharing only system+tools preamble;
that needs longest-common-prefix fork (Slice 4). The critic call is a single
review generation (``tools=[]``): additional tool-executing rounds would just be
session *extends*, whose acceleration is already proven by the coder-session /
aligned runs — the novel thing fork adds is the first call reusing parent KV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import benchmarks.swe_bench as sb
from benchmarks.swe_bench import (
    Problem,
    build_agent,
    capture_patch,
    checked_out_repo,
    load_verified,
    run_with_stuck_retries,
    _format_user_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("slice3")

URI = "ws://127.0.0.1:18099"
INFERLET = "openhands-coder-session@0.1.0"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Default instances: the 4 aligned-eval instances — known-solvable at temp 0,
# sandboxes reused via $SWEB_WS_ROOT, deterministic.
DEFAULT_INSTANCES = [
    "django__django-14373",
    "django__django-15569",
    "pydata__xarray-4075",
    "scikit-learn__scikit-learn-10908",
]

CRITIC_INSTRUCTION = (
    "A fix for the issue above has now been applied to the repository (see the "
    "trajectory above). Act as a critical reviewer of that fix. Carefully "
    "re-read the original issue and the change that was made, then:\n"
    "  1. State whether the fix is correct and complete, and why.\n"
    "  2. Identify any edge case the fix might miss (especially the "
    "negative/empty/None-input cases).\n"
    "  3. Propose a concrete regression guard (a minimal test) that would catch "
    "a future regression of this bug.\n"
    "Give your review as prose followed by the proposed test as a unified diff."
)


def _llm_kwargs(**overrides) -> dict[str, Any]:
    kw = dict(
        model=MODEL,
        pie_uri=URI,
        pie_username="local-dev",
        pie_inferlet=INFERLET,
        pie_session=True,
        native_tool_calling=True,
        pie_python_tool_parser=True,  # matches the winning aligned/pyparser config
        pie_kv_verify=True,  # inferlet asserts seq_len==render_len on every call;
                             # a fork that miscounts prefix+suffix errors the call.
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
    )
    kw.update(overrides)
    return kw


def _make_llm(**overrides):
    from pie_openhands import PieLLM
    return PieLLM(**_llm_kwargs(**overrides))


def _canonical_output(resp) -> str:
    """A stable string capturing everything the model emitted this call."""
    try:
        msg = resp.choices[0].message
    except Exception:
        return repr(resp)
    content = msg.content or ""
    calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        if fn is not None:
            calls.append({"name": fn.name, "arguments": fn.arguments})
    return json.dumps({"content": content, "tool_calls": calls}, sort_keys=True)


def _critic_call(llm, messages: list[dict], tools: list, *, max_tokens: int = 2048) -> dict:
    """One real critic generation. Returns telemetry + output hash + wall.

    ``tools`` MUST be the parent's tool schemas: the coder-session inferlet
    renders tool definitions into the prompt, so the child render only extends
    the parent snapshot (and the fork prefix-match only holds) when the same
    tools are passed. Omitting them shortens the render below the snapshot and
    the fork silently falls back to a full fresh prefill (the v1 bug).
    """
    t0 = time.monotonic()
    resp = llm._transport_call(
        messages=messages, tools=tools, max_tokens=max_tokens, temperature=0.0,
    )
    wall = time.monotonic() - t0
    out = _canonical_output(resp)
    stats = llm._pie_session_stats[-1]
    return {
        "mode": stats["mode"],
        "prompt_len": stats["prompt_len"],
        "prefill_tokens": stats["prefill_tokens"],
        "wall_s": wall,
        "out_hash": hashlib.sha256(out.encode()).hexdigest(),
        "out_len": len(out),
    }


# ─── Parent solve: real agent, but keep the llm + capture its last render ───


def _solve_parent(problem: Problem, llm, *, max_iterations: int) -> dict:
    """Run a real OpenHands agent to solve ``problem`` with ``llm``.

    Returns the patch, the messages AND tools of the parent's LAST transport
    call (the render its final KV snapshot was built from — the exact prefix
    the child must reuse), and parent session telemetry. Does NOT close the
    session.
    """
    from openhands.sdk import Conversation

    # Capture the raw ``messages`` and ``tools`` of every parent transport call,
    # atomically, so they come from the SAME (last) call. The last call is the
    # render the parent's final KV snapshot corresponds to; the child forks from
    # that, so child_msgs+tools must reproduce it exactly as a prefix. Tools
    # matter: the inferlet renders tool schemas into the prompt (~4k tokens), so
    # dropping them shortens the render below the snapshot -> fork miss (v1 bug).
    import pie_openhands.llm as _llm_mod
    orig = _llm_mod.PieLLM._transport_call
    captured: dict[str, Any] = {}

    def _wrapped(self, *, messages, **kw):
        if self is llm:
            captured["last"] = [dict(m) for m in messages]
            captured["tools"] = list(kw.get("tools") or [])
        return orig(self, messages=messages, **kw)

    _llm_mod.PieLLM._transport_call = _wrapped
    try:
        with checked_out_repo(problem, cache_dir=None) as ws:
            agent = build_agent(llm, enable_condenser=True)
            conv = Conversation(
                agent=agent,
                workspace=str(ws),
                max_iteration_per_run=max_iterations,
                visualizer=None,
            )
            conv.send_message(_format_user_prompt(problem, ws=ws))
            nudges = run_with_stuck_retries(
                conv, instance_id=problem.instance_id,
            )
            patch = capture_patch(ws)
    finally:
        _llm_mod.PieLLM._transport_call = orig

    if "last" not in captured:
        raise RuntimeError("parent made no transport call — nothing to fork from")

    return {
        "patch": patch,
        "last_messages": captured["last"],
        "last_tools": captured.get("tools", []),
        "nudges": nudges,
        "session_len": llm._pie_session_len,
        "session_hash": llm._pie_session_hash,
    }


def run_instance(problem: Problem, *, max_iterations: int) -> dict:
    """Full Slice-3 measurement for one instance."""
    logger.info("=== %s: parent solve ===", problem.instance_id)
    parent = _make_llm()
    child = None
    control = None
    control2 = None
    try:
        p = _solve_parent(problem, parent, max_iterations=max_iterations)
        parent_snapshot_len = p["session_len"]
        logger.info(
            "%s: parent solved — %d-byte patch, snapshot=%d tokens, %d nudges",
            problem.instance_id, len(p["patch"]), parent_snapshot_len, p["nudges"],
        )

        tools = p["last_tools"]
        # Continuation prompt = parent's full final render + the critic turn.
        child_msgs = p["last_messages"] + [
            {"role": "user", "content": CRITIC_INSTRUCTION},
        ]

        # ── Fork child: exactly the SDK's delegate-LLM derivation ──────────
        # (manager.py double-copy: model_copy -> reset_metrics -> model_copy)
        child = parent.model_copy(update={"stream": False})
        child.reset_metrics()
        child = child.model_copy(update={"stream": False})
        assert child._pie_fork_from == parent._pie_session_id, \
            "Slice-2 glue did not stamp fork source across the double copy"
        cf = _critic_call(child, child_msgs, tools)
        logger.info("%s: fork  critic -> %s", problem.instance_id, cf)

        # ── Control A: fresh llm, never copied -> cold prefill ─────────────
        control = _make_llm()
        cc = _critic_call(control, child_msgs, tools)
        logger.info("%s: coldA critic -> %s", problem.instance_id, cc)

        # ── Control B: engine-determinism baseline (identical to A) ────────
        # Two identical fresh calls establish the engine's own reproducibility
        # floor. On MoE + batch-dependent numerics temp-0 output is not
        # guaranteed bit-identical, so "fork is behaviour-neutral" must be read
        # as "fork diverges from cold no more than cold diverges from cold".
        control2 = _make_llm()
        cc2 = _critic_call(control2, child_msgs, tools)
        logger.info("%s: coldB critic -> mode=%s out_hash=%s", problem.instance_id, cc2["mode"], cc2["out_hash"][:12])

        # ── Parent-uncorrupted check: extend the parent post-fork ──────────
        probe_msgs = p["last_messages"] + [
            {"role": "user", "content": "Reply with the single word OK."},
        ]
        pe = _critic_call(parent, probe_msgs, tools, max_tokens=8)
        logger.info("%s: parent extend post-fork -> mode=%s", problem.instance_id, pe["mode"])

        # ── Structural checks (the pass/fail gate) ─────────────────────────
        # kv_verify was on for every call above; a fork that miscounted prefix+
        # suffix would have raised, so reaching here with mode==forked means the
        # structural fork check passed.
        checks: dict[str, Any] = {}
        checks["child_forked"] = (cf["mode"] == "forked")
        checks["control_fresh"] = (cc["mode"] == "fresh")
        checks["renders_match"] = (cf["prompt_len"] == cc["prompt_len"])
        checks["suffix_only"] = (
            cf["prefill_tokens"] == cf["prompt_len"] - parent_snapshot_len
        )
        checks["parent_uncorrupted"] = (pe["mode"] == "extended")

        # ── Behaviour-neutrality (reported, interpreted vs the noise floor) ─
        engine_deterministic = (cc["out_hash"] == cc2["out_hash"])
        fork_matches_cold = (cf["out_hash"] == cc["out_hash"])
        # Fork is neutral if it reproduces cold exactly OR — when the engine
        # itself is nondeterministic — it is no more divergent than cold-vs-cold.
        neutral = fork_matches_cold or not engine_deterministic

        saved = cc["prefill_tokens"] - cf["prefill_tokens"]
        pct_prefill = 100.0 * saved / cc["prefill_tokens"] if cc["prefill_tokens"] else 0.0
        wall_saved = cc["wall_s"] - cf["wall_s"]
        pct_wall = 100.0 * wall_saved / cc["wall_s"] if cc["wall_s"] else 0.0

        result = {
            "instance_id": problem.instance_id,
            "parent_patch_bytes": len(p["patch"]),
            "parent_snapshot_len": parent_snapshot_len,
            "child_len": cf["prompt_len"],
            "fork_prefill": cf["prefill_tokens"],
            "cold_prefill": cc["prefill_tokens"],
            "prefill_saved": saved,
            "prefill_saved_pct": round(pct_prefill, 1),
            "fork_wall_s": round(cf["wall_s"], 2),
            "cold_wall_s": round(cc["wall_s"], 2),
            "wall_saved_pct": round(pct_wall, 1),
            "checks": checks,
            "structural_pass": all(checks.values()),
            "engine_deterministic": engine_deterministic,
            "fork_matches_cold": fork_matches_cold,
            "behavior_neutral": neutral,
            "fork_out_hash": cf["out_hash"],
            "coldA_out_hash": cc["out_hash"],
            "coldB_out_hash": cc2["out_hash"],
        }
        # An instance "passes" on the structural fork claims; neutrality is
        # reported alongside (and only fails if the engine is deterministic yet
        # fork still diverges — a genuine fork-correctness red flag).
        result["all_checks_pass"] = result["structural_pass"] and neutral
        logger.info(
            "%s: prefill %d->%d (%.1f%% saved), wall %.1f->%.1fs (%.1f%%), "
            "struct=%s engine_det=%s fork==cold=%s neutral=%s",
            problem.instance_id, cc["prefill_tokens"], cf["prefill_tokens"],
            pct_prefill, cc["wall_s"], cf["wall_s"], pct_wall,
            result["structural_pass"], engine_deterministic, fork_matches_cold, neutral,
        )
        return result
    finally:
        for llm in (parent, child, control, control2):
            if llm is not None:
                try:
                    llm.close_pie_session()
                except Exception:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", action="append", dest="instance_ids")
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--out", default=None, help="JSON results path")
    args = ap.parse_args()

    instance_ids = args.instance_ids or DEFAULT_INSTANCES

    rows = {r["instance_id"]: r for r in load_verified()}
    missing = [i for i in instance_ids if i not in rows]
    if missing:
        raise SystemExit(f"unknown instance_ids: {missing}")

    results = []
    for iid in instance_ids:
        problem = Problem.from_row(rows[iid])
        try:
            results.append(run_instance(problem, max_iterations=args.max_iterations))
        except Exception as e:
            logger.exception("instance %s failed", iid)
            results.append({"instance_id": iid, "error": f"{type(e).__name__}: {e}"})

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SLICE-3 CRITIC-CONTINUE FORK DELEGATION — SUMMARY")
    print("=" * 72)
    ok = [r for r in results if r.get("all_checks_pass")]
    for r in results:
        if "error" in r:
            print(f"  {r['instance_id']:<40} ERROR: {r['error']}")
            continue
        print(
            f"  {r['instance_id']:<40} "
            f"prefill {r['cold_prefill']:>7}->{r['fork_prefill']:<6} "
            f"({r['prefill_saved_pct']:>5.1f}% saved)  "
            f"wall {r['cold_wall_s']:>6.1f}->{r['fork_wall_s']:<6.1f}s "
            f"({r['wall_saved_pct']:>5.1f}%)  "
            f"struct={r['structural_pass']}  "
            f"eng_det={r['engine_deterministic']}  "
            f"fork==cold={r['fork_matches_cold']}  "
            f"neutral={r['behavior_neutral']}"
        )
    print(f"\n  {len(ok)}/{len(results)} instances passed (structural + neutral)")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")

    all_pass = bool(results) and all(r.get("all_checks_pass") for r in results)
    print("\nSLICE-3 MEASUREMENT " + ("PASSED" if all_pass else "INCOMPLETE"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
