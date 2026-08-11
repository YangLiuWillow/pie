#!/usr/bin/env python3
"""KV snapshot save/resume test for rl-rollout on pie 0.5.

Runs against a live `pie serve` (one engine lifetime — snapshots are
engine-global but do not survive restart):

  1. warm:   prompt P, greedy          -> saves boundary L1 = len(P)+k-1
  2. cold:   P2 = P + gen + extra, greedy, no resume hints -> reference
  3. resume: same P2, saved_lens=[L1]  -> must report cached_tokens == L1 and
             match the cold run's tokens exactly (greedy determinism through a
             resumed snapshot is the correctness proof)
  4. chain:  P3 = P2 + gen2 + extra2, saved_lens=[L3(from run 3), L1] ->
             must resume at the LONGER boundary
"""
import asyncio
import json
import sys

from pie_client import PieClient, Event

WASM = "/root/cargo-target/wasm32-wasip2/release/rl_rollout.wasm"
MANIFEST = "/root/pie/tests/inferlets/rl-rollout/Pie.toml"
URI = "ws://127.0.0.1:8080"

# "The capital of France is" + filler to cross a page boundary (page=32).
P = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9625, 374, 1083,
     264, 3283, 429, 374, 3881, 369, 1181, 6722, 13, 576, 6722, 315, 9625,
     374, 12095, 13, 576, 6722, 315, 9625, 374, 1083, 264, 3283, 13]
EXTRA = [576, 1196, 4588, 911, 419, 13]   # arbitrary "next turn" tokens
EXTRA2 = [3838, 911, 279, 3146, 30]


async def run(client, iid, **kw):
    inp = {"max_tokens": 12, "temperature": 0.0, "save_kv": True, **kw}
    proc = await client.launch_process(iid, input=inp)
    out = []
    while True:
        event, msg = await proc.recv()
        if event == Event.Return:
            out.append(msg)
            break
        if event == Event.Error:
            raise RuntimeError(f"inferlet error: {msg}")
        out.append(msg)
    body = out[-1]
    return json.loads(body if isinstance(body, str) else body.decode())


async def main():
    client = PieClient(URI)
    await client.connect()
    with open(MANIFEST) as f:
        name = ver = None
        for line in f:
            if line.startswith("name"):
                name = line.split('"')[1]
            if line.startswith("version"):
                ver = line.split('"')[1]
                break
    iid = f"{name}@{ver}"
    await client.install_program(WASM, MANIFEST, force_overwrite=True)

    r1 = await run(client, iid, prompt_tokens=P)
    L1 = r1["saved_len"]
    print(f"run1 warm:   n={r1['num_prompt_tokens']} out={r1['num_output_tokens']} "
          f"cached={r1['cached_tokens']} saved_len={L1}")
    assert r1["cached_tokens"] == 0
    assert L1 == len(P) + r1["num_output_tokens"] - 1, (L1, len(P), r1["num_output_tokens"])

    p2 = P + r1["token_ids"] + EXTRA
    r2 = await run(client, iid, prompt_tokens=p2, save_kv=False)
    print(f"run2 cold:   n={r2['num_prompt_tokens']} out={r2['num_output_tokens']} "
          f"cached={r2['cached_tokens']}")
    assert r2["cached_tokens"] == 0

    r3 = await run(client, iid, prompt_tokens=p2, saved_lens=[L1])
    L3 = r3["saved_len"]
    print(f"run3 resume: n={r3['num_prompt_tokens']} out={r3['num_output_tokens']} "
          f"cached={r3['cached_tokens']} saved_len={L3}")
    assert r3["cached_tokens"] == L1, f"expected resume at {L1}, got {r3['cached_tokens']}"

    same_tokens = r2["token_ids"] == r3["token_ids"]
    lp_diff = max((abs(a - b) for a, b in zip(r2["logprobs"], r3["logprobs"])), default=0.0)
    print(f"  cold-vs-resumed: tokens_equal={same_tokens} max_logprob_diff={lp_diff:.3e}")
    assert same_tokens, f"token divergence:\n  cold={r2['token_ids']}\n  warm={r3['token_ids']}"

    p3 = p2 + r3["token_ids"] + EXTRA2
    r4 = await run(client, iid, prompt_tokens=p3, saved_lens=[L3, L1])
    print(f"run4 chain:  n={r4['num_prompt_tokens']} out={r4['num_output_tokens']} "
          f"cached={r4['cached_tokens']} saved_len={r4['saved_len']}")
    assert r4["cached_tokens"] == L3, f"expected resume at longer {L3}, got {r4['cached_tokens']}"

    print("KV_SNAPSHOT_TEST_PASSED")
    await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
