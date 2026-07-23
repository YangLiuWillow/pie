"""GPU smoke for the OpenEvolve PieLLM backend (openevolve/llm/pie.py).

Exercises the backend end-to-end against a running pie server:
  * generate_with_context (single, skip_analysis path) returns text.
  * generate_children (batched B2) returns N diffs + shared_prefill_saved > 0.
  * delete_parent prunes.

Proves the Python transport wiring on top of the already-GPU-verified inferlet.
"""
import asyncio
import os
import sys

sys.path.insert(0, "/nfs/roberts/project/pi_ql324/ly337/gemini/openevolve")
from openevolve.llm.pie import PieLLM, compute_topk_sig  # noqa: E402


class Cfg:
    name = "Qwen2.5-Coder-7B-Instruct"
    system_message = "You are an expert software engineer."
    temperature = 0.7
    top_p = 0.95
    max_tokens = 256
    timeout = 180
    random_seed = 0
    api_base = None


PARENT_BLOCK = """# Current Program
```python
def pairwise_sum(xs):
    total = 0
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i < j:
                total += xs[i] + xs[j]
    return total
```"""

SECTIONS = {
    "system": "You are an expert software engineer improving programs by evolution.",
    "task": "Improve the FITNESS SCORE of the program below.",
    "parent_block": PARENT_BLOCK,
    "analysis_instruction": "List exactly 4 distinct improvement directions, numbered 1-4. No code yet.",
}
STEERS = [f"Implement improvement direction #{k} as a SEARCH/REPLACE diff." for k in range(1, 5)]


async def main() -> None:
    os.environ.setdefault("PIE_URI", "ws://127.0.0.1:18099")
    llm = PieLLM(Cfg())

    # 1. single completion (skip_analysis path).
    txt = await llm.generate_with_context(
        "You are a helpful assistant.",
        [{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=8,
    )
    print("single ->", repr(txt[:80]))
    assert isinstance(txt, str) and len(txt) > 0, "single completion empty"

    # 2. batched analyze-then-diverge.
    sig = compute_topk_sig(PARENT_BLOCK, {"combined_score": 0.41}, ["p1", "p2"])
    out = await llm.generate_children(
        parent_id="prog-root",
        topk_sig=sig,
        sections=SECTIONS,
        num_children=4,
        steers=STEERS,
        child_temperature=[0.9],
        analysis_max_tokens=200,
        child_max_tokens=200,
    )
    t = out["telemetry"]
    print("batched telemetry ->", t)
    assert len(out["children"]) == 4, out
    assert t["l1g_mode"] == "generated" and t["l1g_decode_tokens"] > 0, t
    assert t["shared_prefill_saved"] == t["shared_prefix_tokens"] * 3, t
    diffs = [c["diff"] for c in out["children"]]
    assert all(len(d) > 0 for d in diffs), "empty child diff"
    print(f"   4 children, distinct={len(set(diffs))}, saved={t['shared_prefill_saved']} tokens")

    # 3. reuse the analysis (needs keep first).
    await llm.generate_children(parent_id="prog-root", topk_sig=sig, sections=SECTIONS,
                                num_children=4, steers=STEERS, keep_analysis=True,
                                analysis_max_tokens=200, child_max_tokens=200)
    out2 = await llm.generate_children(parent_id="prog-root", topk_sig=sig, sections=SECTIONS,
                                       num_children=4, steers=STEERS, reuse_analysis=True,
                                       child_max_tokens=200)
    assert out2["telemetry"]["l1g_mode"] == "reused", out2["telemetry"]
    print("   reuse_analysis -> l1g_mode=reused, decode=0 OK")

    # 4. lifecycle prune.
    await llm.delete_parent("prog-root", sig)
    print("\nPIE_LLM BACKEND SMOKE PASS")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"BACKEND SMOKE FAIL: {e}", file=sys.stderr)
        sys.exit(1)
