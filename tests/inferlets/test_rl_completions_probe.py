"""Scratch probe: find the context-size wall (scheduler endowment hypothesis).

Sends progressively longer string prompts to rl-completions and reports where
generation stops returning tokens / starts erroring. Not part of any suite.

    .venv/bin/python tests/inferlets/test_rl_completions_probe.py \\
        --driver portable --model Qwen/Qwen3-0.6B --device cpu --timeout 1200
"""
from __future__ import annotations

import httpx

from conftest import run_tests
from test_rl_completions import _launch


async def test_rl_completions_probe(client, args):
    base = await _launch(client)
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        for words in (5000, 6200, 7500, 9000):
            prompt = "alpha bravo charlie delta echo foxtrot golf hotel " * (words // 8)
            resp = await http.post(
                f"{base}/v1/completions",
                json={"prompt": prompt, "max_tokens": 8, "temperature": 0.0},
            )
            if resp.status_code != 200:
                print(f"~{words} words -> HTTP {resp.status_code}: {resp.text[:120]}")
                continue
            body = resp.json()
            n_prompt = len(body.get("prompt_token_ids") or [])
            choice = body["choices"][0]
            n_out = len(choice.get("token_ids") or [])
            print(f"prompt {n_prompt} tokens -> {n_out} completion tokens, finish={choice.get('finish_reason')}")
            assert n_out > 0, f"WALL at ~{n_prompt} prompt tokens: no completion tokens"


def tests():
    return [test_rl_completions_probe]


if __name__ == "__main__":
    run_tests(tests())
