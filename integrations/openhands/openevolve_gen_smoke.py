"""GPU smoke for the openevolve-generation inferlet (Path B, B-batched).

Drives the inferlet directly through PieClient (no OpenEvolve). Proves the
load-bearing KV-sharing mechanics of the analyze-then-diverge design:

  1. fresh batch     -> L1p built, L1g generated, N children forked off P+A;
                        shared_prefill_saved == (N-1)*shared_prefix_tokens > 0.
  2. same key again  -> L1p OPENED (cross-call KV reuse, prefill 0); L1g
                        regenerated (deleted at end-of-batch by default).
  3. keep + reuse    -> with keep_analysis the L1g snapshot survives; a
                        reuse_analysis call then reuses it (no re-decode).
  4. delete          -> action="delete" prunes both snapshots.

The telemetry assertions are the real test; child-text distinctness is a soft
check (model-dependent).
"""
import asyncio
import json
import sys

from pie_client import Event, PieClient

URI = "ws://127.0.0.1:18099"
INFERLET = "openevolve-generation@0.1.0"
ROOT = "/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openevolve-generation"
WASM = f"{ROOT}/target/wasm32-wasip2/release/openevolve_generation.wasm"
MANIFEST = f"{ROOT}/Pie.toml"

PARENT_BLOCK = """# Current Program Information
- Fitness: 0.4120
- Focus areas: runtime, correctness

# Current Program
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
    "system": "You are an expert software engineer improving programs through evolutionary search.",
    "task": "Below is a program and its measured performance. You will improve its FITNESS SCORE.",
    "parent_block": PARENT_BLOCK,
    "analysis_instruction": (
        "First, analyze the program's weaknesses and list exactly 4 distinct, "
        "concrete improvement directions, one per line, numbered 1-4. "
        "Do not write code yet."
    ),
}

STEERS = [
    "Now implement improvement direction #1 as a SEARCH/REPLACE diff.",
    "Now implement improvement direction #2 as a SEARCH/REPLACE diff.",
    "Now implement improvement direction #3 as a SEARCH/REPLACE diff.",
    "Now implement improvement direction #4 as a SEARCH/REPLACE diff.",
]

N = 4
KEY = dict(run_id="smoke-run", parent_id="prog-root", topk_sig="sig0")


async def call(client, payload):
    proc = await client.launch_process(INFERLET, input=payload)
    while True:
        event, value = await asyncio.wait_for(proc.recv(), timeout=180)
        if event == Event.Return:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode()
            if isinstance(value, str):
                value = json.loads(value)
            return value
        if event == Event.Error:
            raise RuntimeError(f"inferlet error: {value!r}")


def base(**extra):
    return {
        **KEY,
        "sections": SECTIONS,
        "num_children": N,
        "steers": STEERS,
        "analysis_max_tokens": 256,
        "child_max_tokens": 256,
        "analysis_temperature": 0.7,
        "child_temperature": [0.9],
        **extra,
    }


async def main():
    async with PieClient(URI) as client:
        await client.authenticate("local-dev")
        await client.install_program(WASM, MANIFEST, force_overwrite=True)

        # 1. Fresh batch: build L1p, generate L1g, fork N leaves.
        r1 = await call(client, base())
        t1 = r1["telemetry"]
        print("1 fresh:", json.dumps(t1))
        assert t1["l1p_mode"] == "built", t1
        assert t1["l1p_prefill_tokens"] > 0, t1
        assert t1["l1g_mode"] == "generated" and t1["l1g_decode_tokens"] > 0, t1
        assert t1["shared_prefix_tokens"] > 0, t1
        assert len(r1["children"]) == N, r1
        assert t1["shared_prefill_saved"] == t1["shared_prefix_tokens"] * (N - 1), t1
        diffs = [c["diff"] for c in r1["children"]]
        assert all(len(d) > 0 for d in diffs), "some child produced empty output"
        distinct = len(set(diffs))
        print(f"   children: {N}, distinct={distinct}, analysis_len={len(r1['analysis']['text'])}")

        # 2. Same key again: L1p must be OPENED (cross-call KV reuse, prefill 0).
        r2 = await call(client, base())
        t2 = r2["telemetry"]
        print("2 reopen:", json.dumps(t2))
        assert t2["l1p_mode"] == "opened", t2
        assert t2["l1p_prefill_tokens"] == 0, "L1p KV was not reused across calls!"
        assert t2["shared_prefill_saved"] == t2["shared_prefix_tokens"] * (N - 1), t2

        # 3. keep_analysis: publish L1g, then reuse it.
        r3 = await call(client, base(keep_analysis=True))
        assert r3["telemetry"]["l1g_mode"] == "generated", r3["telemetry"]
        r4 = await call(client, base(reuse_analysis=True))
        t4 = r4["telemetry"]
        print("4 reuse-A:", json.dumps(t4))
        assert t4["l1g_mode"] == "reused", t4
        assert t4["l1g_decode_tokens"] == 0, "reused analysis should not re-decode"
        assert len(r4["children"]) == N, r4

        # 4. delete prunes both snapshots.
        rd = await call(client, base(action="delete"))
        assert rd["telemetry"]["l1p_mode"] == "deleted", rd
        # after delete, a fresh key rebuilds L1p.
        r5 = await call(client, base())
        assert r5["telemetry"]["l1p_mode"] == "built", r5["telemetry"]

        print("\nSMOKE PASS: KV sharing + naming lifecycle verified.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"SMOKE FAIL: {e}", file=sys.stderr)
        sys.exit(1)
