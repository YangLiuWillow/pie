"""E2E test for rl-completions KV snapshot reuse (cumulative-prefix cache).

Three turns against one daemon:

  1. prompt P                  -> full prefill, completion C.
                                  Asserts resumed_tokens == 0.
  2. prompt P + C + suffix     -> the cumulative-turn shape. Asserts the
                                  advertised prefix was reused
                                  (resumed_tokens covers at least P, at most
                                  P+C) and the response is well-formed.
  3. unrelated prompt          -> pointer hash mismatch. Asserts
                                  resumed_tokens == 0 (miss is safe).

Wall-clock per turn is printed for eyeballing; the assertion is on the
deterministic `pie_reuse.resumed_tokens` debug field, not timing.

    python3 tests/inferlets/test_rl_completions_reuse.py \\
        --driver portable --model /root/qwen17b-f16 --device auto
"""
from __future__ import annotations

import time

import httpx

from conftest import run_tests
from test_rl_completions import _launch


async def _turn(http, base, prompt, max_tokens=16):
    t0 = time.monotonic()
    resp = await http.post(
        f"{base}/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
    )
    dt = time.monotonic() - t0
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    reused = (body.get("pie_reuse") or {}).get("resumed_tokens")
    assert reused is not None, f"pie_reuse missing: {list(body.keys())}"
    ids = body["prompt_token_ids"]
    completion = body["choices"][0]["token_ids"]
    print(f"turn: prompt={len(ids)} reused={reused} completion={len(completion)} {dt:.2f}s")
    return ids, completion, reused


async def test_rl_completions_reuse(client, args):
    base = await _launch(client)
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        # Long-ish prompt so prefill dominates and reuse is visible.
        seed_text = "The quick brown fox jumps over the lazy dog. " * 120

        p1, c1, r1 = await _turn(http, base, seed_text)
        assert r1 == 0, f"turn 1 must be a cold start, got resumed={r1}"
        assert c1, "turn 1 generated nothing"

        # Cumulative turn: previous prompt + completion + a small suffix.
        p2 = p1 + c1 + p1[:16]
        _, c2, r2 = await _turn(http, base, p2)
        assert c2, "turn 2 generated nothing"
        assert r2 >= len(p1), (
            f"turn 2 reused only {r2} tokens; expected at least the previous "
            f"prompt ({len(p1)})"
        )
        assert r2 <= len(p1) + len(c1), (
            f"turn 2 claims {r2} reused tokens > previous prompt+completion "
            f"({len(p1) + len(c1)})"
        )

        # Unrelated prompt: the advertised pointer must not match.
        p3, _, r3 = await _turn(http, base, "Completely unrelated text about volcanoes. " * 100)
        assert r3 == 0, f"unrelated prompt must miss, got resumed={r3}"
        assert p3[:8] != p1[:8], "test bug: unrelated prompt tokenized identically?"


def tests():
    return [test_rl_completions_reuse]


if __name__ == "__main__":
    run_tests(tests())
