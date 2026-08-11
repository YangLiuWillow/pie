#!/usr/bin/env python3
"""Golden-fixture replay through the pie-openai-bridge (pie 0.5).

The 0.5 port of `test_rl_completions_replay` from the 0.4 branch: replays the
Phase-1a qwen-code cumulative-token-mode turns (gateway trace records with
per-turn ``prompt_token_ids``) against the bridge, asserting everything
rllm's gateway parses. Token-level agreement with the recorded vLLM
completions is NOT asserted (different engine, different sampling) — the
contract is the envelope and the id accounting.

Beyond the 0.4 test (which replayed one turn per session to bound CPU
prefill time): replays EVERY recorded turn in session order on the GPU,
which drives the real cumulative snapshot chain and reports the aggregate
KV reuse ratio.

Usage: python3 replay_fixtures.py --fixtures /root/fixtures/raw_envelopes \
           [--base http://127.0.0.1:8123] [--max-tokens 24] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


def select_turns(fixtures: Path) -> list[list[dict]]:
    """All records grouped by session, ordered by prompt length (== turn
    order under the append-only invariant)."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(fixtures.glob("*.json")):
        if f.name.startswith("."):
            continue  # macOS AppleDouble files in transferred tarballs
        rec = json.loads(f.read_text())
        if rec.get("prompt_token_ids"):
            by_session[rec.get("session_id", "?")].append(rec)
    return [
        sorted(recs, key=lambda r: len(r["prompt_token_ids"]))
        for _, recs in sorted(by_session.items())
    ]


def stream_completion(base: str, body: dict, timeout: float) -> tuple[list[dict], bool]:
    req = urllib.request.Request(
        f"{base}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    chunks: list[dict] = []
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        assert resp.status == 200, f"status {resp.status}"
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise AssertionError(f"server error mid-stream: {chunk['error']}")
            chunks.append(chunk)
    return chunks, done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="cap turns per session (0 = all)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    sessions = select_turns(args.fixtures)
    assert sessions, "no replayable records with prompt_token_ids"
    total_prompt = total_cached = total_generated = 0
    replayed = 0
    t0 = time.time()

    for si, recs in enumerate(sessions):
        if args.limit:
            recs = recs[: args.limit]
        for rec in recs:
            prompt_ids = rec["prompt_token_ids"]
            raw = rec.get("raw_request") or {}
            body = {
                "prompt": prompt_ids,
                "add_special_tokens": False,
                # Recorded flag shapes, verbatim (bool logprobs etc.):
                "logprobs": raw.get("logprobs", True),
                "return_token_ids": raw.get("return_token_ids", True),
                "stream": True,
                "stream_options": raw.get("stream_options", {"include_usage": True}),
                "model": raw.get("model", ""),
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            }
            n = len(prompt_ids)
            chunks, done = stream_completion(args.base, body, args.timeout)

            assert done, f"[{n}tok] no [DONE]"
            assert chunks, f"[{n}tok] no chunks"
            assert chunks[0].get("prompt_token_ids") == prompt_ids, (
                f"[{n}tok] first-chunk prompt_token_ids is not a byte-exact echo"
            )
            streamed, lps = [], []
            finish = None
            usage = None
            for c in chunks:
                for ch in c.get("choices", []):
                    streamed.extend(ch.get("token_ids", []))
                    lps.extend((ch.get("logprobs") or {}).get("token_logprobs", []))
                    finish = ch.get("finish_reason") or finish
                if c.get("usage"):
                    usage = c["usage"]
            assert streamed, f"[{n}tok] no completion token_ids streamed"
            assert finish in ("stop", "length"), f"[{n}tok] finish_reason {finish!r}"
            assert len(lps) == len(streamed), f"[{n}tok] logprobs misaligned"
            assert usage and usage.get("prompt_tokens") == n, f"[{n}tok] usage {usage}"
            assert usage.get("completion_tokens") == len(streamed), (
                f"[{n}tok] usage.completion_tokens {usage} != {len(streamed)}"
            )
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            total_prompt += n
            total_cached += cached
            total_generated += len(streamed)
            replayed += 1
            print(f"s{si} [{n:>6}tok] gen={len(streamed):<3} cached={cached:>6} "
                  f"({100.0 * cached / n:5.1f}%) finish={finish}")

    dt = time.time() - t0
    ratio = 100.0 * total_cached / max(total_prompt, 1)
    print(f"\nreplayed {replayed} turns in {dt:.1f}s: "
          f"{total_prompt} prompt tokens, {total_cached} cached ({ratio:.1f}% KV reuse), "
          f"{total_generated} generated")
    print("FIXTURE_REPLAY_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
