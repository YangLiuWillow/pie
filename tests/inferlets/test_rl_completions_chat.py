"""Contract test for rl-completions /v1/chat/completions (turn-0, §3.2).

Asserts the response is chat-shaped AND carries the token-id fields the
gateway's cumulative accumulator seeds from:
  - root prompt_token_ids present and non-empty
  - choices[0].token_ids present and non-empty
  - choices[0].message.{role,content}
  - finish_reason in stop|length|tool_calls
  - usage.completion_tokens == len(token_ids)

Then feeds the turn-0 prompt_ids + completion back through /v1/completions
as a cumulative turn (the real turn-0 → turn-1 handoff) and checks the id
echo — the append-only invariant across the endpoint boundary.

    python3 tests/inferlets/test_rl_completions_chat.py \\
        --driver portable --model /root/qwen17b-f16 --device auto
"""
from __future__ import annotations

import httpx

from conftest import run_tests
from test_rl_completions import _launch


async def test_rl_completions_chat(client, args):
    base = await _launch(client)
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        resp = await http.post(
            f"{base}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a terse assistant."},
                    {"role": "user", "content": "Name one primary color."},
                ],
                "max_tokens": 16,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:200]}"
        b = resp.json()

        assert isinstance(b.get("prompt_token_ids"), list) and b["prompt_token_ids"], (
            f"root prompt_token_ids missing/empty: {list(b.keys())}"
        )
        choice = b["choices"][0]
        assert choice.get("finish_reason") in ("stop", "length", "tool_calls")
        msg = choice["message"]
        assert msg["role"] == "assistant" and "content" in msg, f"bad message: {msg}"
        tok = choice.get("token_ids")
        assert isinstance(tok, list) and tok, "choices[0].token_ids missing/empty"
        assert (b.get("usage") or {}).get("completion_tokens") == len(tok)
        print(f"chat turn-0: prompt={len(b['prompt_token_ids'])} completion={len(tok)} "
              f"finish={choice['finish_reason']} text={msg['content']!r}")

        # Turn-0 -> turn-1 handoff: the cumulative /v1/completions prompt is
        # turn-0's prompt_ids + its completion + a suffix. Must be echoed exact.
        cumulative = b["prompt_token_ids"] + tok + b["prompt_token_ids"][:8]
        resp2 = await http.post(
            f"{base}/v1/completions",
            json={"prompt": cumulative, "max_tokens": 8, "temperature": 0.0},
        )
        assert resp2.status_code == 200, f"turn-1: {resp2.status_code} {resp2.text[:200]}"
        b2 = resp2.json()
        assert b2["prompt_token_ids"] == cumulative, "turn-1 prompt echo not byte-exact"
        assert b2["choices"][0]["token_ids"], "turn-1 produced no tokens"
        print(f"turn-1 handoff: cumulative prompt={len(cumulative)} echoed exact, "
              f"reused={(b2.get('pie_reuse') or {}).get('resumed_tokens')}")


def tests():
    return [test_rl_completions_chat]


if __name__ == "__main__":
    run_tests(tests())
