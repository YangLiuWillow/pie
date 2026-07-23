"""Stage-2 A/B: mask-condense vs rebuild-condense on a REAL agent trajectory.

Runs ONE SWE-bench instance twice through the openhands-agent (Pattern A) with a
deliberately LOW context_token_limit so condensation fires several times:
  - condense_mode="rebuild": summarize dropped turns + re-prefill a fresh context
    (what vLLM+prefix-caching must do — and it also pays an LLM summary call).
  - condense_mode="mask": mask the stale middle turns' KV out of attention
    (0 re-prefill, no summary call — the B2-mask win).

Compares wall time, prefill tokens, steps, and patch (quality parity). The mask
run should be faster (skips re-prefill AND the summary generation each condense),
with a comparable patch.

Env: INSTANCE_ID, PIE_PORT, CONTEXT_LIMIT, MAX_STEPS, KEEP_RECENT.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from datasets import load_dataset

from benchmarks.swe_bench import (
    Problem, checked_out_repo, _run_agent_inferlet, _format_user_prompt, capture_patch,
)
from tool_server import start_tool_server

logging.basicConfig(level=logging.INFO, format="%(message)s")

INSTANCE_ID = os.environ.get("INSTANCE_ID", "sympy__sympy-19346")
PIE_PORT = int(os.environ.get("PIE_PORT", "18098"))
CONTEXT_LIMIT = int(os.environ.get("CONTEXT_LIMIT", "6000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
KEEP_RECENT = int(os.environ.get("KEEP_RECENT", "10"))


def _load(instance_id):
    for row in load_dataset("princeton-nlp/SWE-bench_Verified", split="test"):
        if row["instance_id"] == instance_id:
            return Problem.from_row(row)
    raise SystemExit(f"instance {instance_id!r} not found")


def run_mode(problem, mode):
    with checked_out_repo(problem) as ws:
        server, port = start_tool_server(str(ws))
        try:
            result = asyncio.run(_run_agent_inferlet(
                f"ws://127.0.0.1:{PIE_PORT}", "openhands-agent@0.1.0",
                task=_format_user_prompt(problem, use_cwd=True),
                tool_server_url=f"http://127.0.0.1:{port}",
                max_steps=MAX_STEPS,
                context_token_limit=CONTEXT_LIMIT,
                condense_mode=mode,
                condense_keep_recent=KEEP_RECENT,
                idle_timeout_s=600.0,
                instance_timeout_s=2400.0,
            ))
            patch = capture_patch(ws)
        finally:
            server.shutdown()
    m = result.get("metrics", {}) if isinstance(result, dict) else {}
    return {
        "mode": mode,
        "finished": result.get("finished") if isinstance(result, dict) else None,
        "steps": result.get("steps", 0) if isinstance(result, dict) else 0,
        "wall_s": m.get("total_wall_s", 0.0),
        "generate_s": m.get("total_generate_s", 0.0),
        "prompt_tokens": m.get("total_prompt_tokens", 0),
        "completion_tokens": m.get("total_completion_tokens", 0),
        "patch_bytes": len(patch),
    }


def main() -> int:
    print(f"=== mask-condense A/B: {INSTANCE_ID} "
          f"(context_limit={CONTEXT_LIMIT}, max_steps={MAX_STEPS}, keep_recent={KEEP_RECENT}) ===")
    problem = _load(INSTANCE_ID)
    results = []
    for mode in ("rebuild", "mask"):
        print(f"\n----- running condense_mode={mode} -----")
        results.append(run_mode(problem, mode))

    print("\n=== A/B summary ===")
    hdr = f"{'mode':8} {'finished':8} {'steps':>5} {'wall_s':>9} {'gen_s':>9} {'prompt_tok':>11} {'patchB':>7}"
    print(hdr)
    for r in results:
        print(f"{r['mode']:8} {str(r['finished']):8} {r['steps']:>5} "
              f"{r['wall_s']:>9.1f} {r['generate_s']:>9.1f} {r['prompt_tokens']:>11} {r['patch_bytes']:>7}")
    rb, mk = results[0], results[1]
    if rb["wall_s"] > 0:
        print(f"\nmask vs rebuild: wall {mk['wall_s']:.1f}s vs {rb['wall_s']:.1f}s "
              f"({100*(rb['wall_s']-mk['wall_s'])/rb['wall_s']:+.1f}% faster), "
              f"prompt_tokens {mk['prompt_tokens']} vs {rb['prompt_tokens']}")
        print("NOTE: mask skips BOTH the kept-context re-prefill AND the summary "
              "LLM call each condensation. Grep the log for [condense] vs "
              "[condense-mask] to count events. Check patch bytes for quality parity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
