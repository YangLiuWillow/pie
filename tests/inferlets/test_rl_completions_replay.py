"""Golden-fixture replay test for the rl-completions inferlet (SLOW, gated).

Replays real cumulative-token-mode requests recorded from the Phase 1a
qwen-code runs through stock vLLM (fixtures/rl_completions_1.7b/ — gateway
trace records with per-turn ``prompt_token_ids``), against the inferlet on
this machine. This is the Phase 1b promise made executable: Pie serves the
*recorded* contract, not a reading of it.

Per replayed turn (the shortest turn of each episode/session, output capped):
  - the EXACT recorded request flags are sent (``logprobs: true`` bool,
    ``return_token_ids``, ``stream`` + ``stream_options`` as recorded)
  - prompt = the recorded ``prompt_token_ids``, byte-exact
  - response must satisfy everything rllm's gateway parses: first-chunk
    ``prompt_token_ids`` echo, per-chunk ``token_ids`` deltas, terminal
    ``finish_reason`` + ``usage``, ``[DONE]``

Token-level agreement with vLLM's recorded completions is NOT asserted —
different engine, different sampling; the contract is the envelope and the
id accounting, not the model's choices.

This test involves ~9-15k-token CPU/Metal prefills and a 1.7B model
download on first run. It is NOT part of run_all; invoke explicitly:

    VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH \\
      .venv/bin/python tests/inferlets/test_rl_completions_replay.py \\
      --driver portable --model Qwen/Qwen3-1.7B --device cpu \\
      --portable-n-gpu-layers 99 --timeout 900
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import httpx

from conftest import ROOT, run_tests
from test_rl_completions import _launch

FIXTURES = ROOT / "tests" / "inferlets" / "fixtures" / "rl_completions_1.7b" / "raw_envelopes"

# Cap generation per replayed turn: the contract under test is the envelope
# and id accounting, and 9-15k-token prefills already dominate runtime.
REPLAY_MAX_TOKENS = 24


def _select_turns() -> list[dict]:
    """Shortest-prompt record per recorded session (one per episode)."""
    by_session: dict[str, dict] = {}
    for f in sorted(FIXTURES.glob("*.json")):
        rec = json.loads(f.read_text())
        ids = rec.get("prompt_token_ids") or []
        if not ids:
            continue
        sid = rec.get("session_id", "?")
        if sid not in by_session or len(ids) < len(by_session[sid]["prompt_token_ids"]):
            by_session[sid] = rec
    return list(by_session.values())


async def test_rl_completions_replay(client, args):
    if not FIXTURES.is_dir():
        raise FileNotFoundError(
            f"{FIXTURES} missing — copy the Phase 1a golden fixtures in first"
        )
    turns = _select_turns()
    assert turns, "no replayable records with prompt_token_ids"

    base = await _launch(client)

    async with httpx.AsyncClient(timeout=args.timeout) as http:
        for rec in turns:
            prompt_ids = rec["prompt_token_ids"]
            raw = rec.get("raw_request") or {}
            body = {
                "prompt": prompt_ids,
                "add_special_tokens": False,
                # Recorded flags, verbatim shapes:
                "logprobs": raw.get("logprobs", True),
                "return_token_ids": raw.get("return_token_ids", True),
                "stream": True,
                "stream_options": raw.get("stream_options", {"include_usage": True}),
                "model": raw.get("model", ""),
                "max_tokens": REPLAY_MAX_TOKENS,
                "temperature": 0.0,
            }

            chunks: list[dict] = []
            done = False
            async with http.stream("POST", f"{base}/v1/completions", json=body) as resp:
                assert resp.status_code == 200, f"status {resp.status_code}"
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise AssertionError(f"server error mid-stream: {chunk['error']}")
                    chunks.append(chunk)

            n = len(prompt_ids)
            assert done, f"[{n}tok] no [DONE]"
            assert chunks, f"[{n}tok] no chunks"
            assert chunks[0].get("prompt_token_ids") == prompt_ids, (
                f"[{n}tok] first-chunk prompt_token_ids is not a byte-exact echo"
            )
            streamed = [t for c in chunks for t in c["choices"][0].get("token_ids", [])]
            assert streamed, f"[{n}tok] no completion token_ids streamed"
            finals = [c for c in chunks if c["choices"][0].get("finish_reason")]
            assert finals and finals[-1]["choices"][0]["finish_reason"] in ("stop", "length"), (
                f"[{n}tok] missing/invalid finish_reason"
            )
            usage = finals[-1].get("usage") or {}
            assert usage.get("prompt_tokens") == n, f"[{n}tok] usage.prompt_tokens {usage}"
            assert usage.get("completion_tokens") == len(streamed), (
                f"[{n}tok] usage.completion_tokens {usage} != {len(streamed)}"
            )
            print(f"replayed {n}-token turn: {len(streamed)} tokens generated, "
                  f"finish={finals[-1]['choices'][0]['finish_reason']}")


def tests():
    return [test_rl_completions_replay]


if __name__ == "__main__":
    run_tests(tests())
