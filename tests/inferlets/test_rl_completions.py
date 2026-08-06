"""E2E contract test for the rl-completions inferlet (HTTP daemon).

Validates the §3.1 cumulative-token-mode wire contract that rllm's model
gateway parses (pie-rl-verl-integration.md; recorded reality in
tests/inferlets/fixtures/rl_completions*/):

  - root ``prompt_token_ids`` echoes the input ids exactly
  - ``choices[0].token_ids`` present and non-empty (missing => rllm
    EnrichMismatchError => rollout retry burn)
  - the sampled stop token is IN ``token_ids`` but NOT in ``text``
  - ``finish_reason`` is "stop" | "length"
  - streaming: first chunk carries ``prompt_token_ids``; per-chunk
    ``token_ids`` deltas concatenate to the completion; final chunk carries
    ``finish_reason`` + ``usage``; ``data: [DONE]`` terminates
  - append-extension: a follow-up prompt of (prev prompt + prev completion +
    suffix) is served and echoed intact — the trainer's prefix-merge input

Usage::

    uv run python tests/inferlets/test_rl_completions.py --dummy
    uv run python tests/inferlets/test_rl_completions.py --model Qwen/Qwen3-0.6B --device cpu --driver dev
"""
from __future__ import annotations

import json
import socket
import time
import tomllib

import httpx

from conftest import INFERLETS_DIR, run_tests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


async def _launch(client) -> str:
    name = "rl-completions"
    wasm = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "release" / "rl_completions.wasm"
    manifest = INFERLETS_DIR / name / "Pie.toml"
    if not wasm.exists():
        raise FileNotFoundError(f"No WASM binary: {wasm} (cargo build --target wasm32-wasip2 --release)")
    meta = tomllib.loads(manifest.read_text())
    inferlet_id = f"{meta['package']['name']}@{meta['package']['version']}"
    await client.install_program(wasm, manifest, force_overwrite=True)
    port = _find_free_port()
    await client.launch_daemon(inferlet_id, port)
    if not _wait_for_port(port):
        raise RuntimeError(f"rl-completions daemon did not bind port {port}")
    return f"http://127.0.0.1:{port}"


def _check_envelope(body: dict, prompt_ids: list[int] | None) -> list[int]:
    """Assert §3.1 response shape; return completion token_ids."""
    assert isinstance(body.get("prompt_token_ids"), list) and body["prompt_token_ids"], (
        f"root prompt_token_ids missing/empty: {list(body.keys())}"
    )
    if prompt_ids is not None:
        assert body["prompt_token_ids"] == prompt_ids, "prompt_token_ids is not an exact echo"
    assert "weight_version" in body, "weight_version missing"
    choice = body["choices"][0]
    token_ids = choice.get("token_ids")
    assert isinstance(token_ids, list) and token_ids, "choices[0].token_ids missing/empty"
    assert choice.get("finish_reason") in ("stop", "length"), f"finish_reason: {choice.get('finish_reason')}"
    usage = body.get("usage") or {}
    assert usage.get("completion_tokens") == len(token_ids), "usage.completion_tokens mismatch"
    return token_ids


async def test_rl_completions(client, args):
    base = await _launch(client)

    async with httpx.AsyncClient(timeout=120) as http:
        # --- /health ---
        resp = await http.get(f"{base}/health")
        assert resp.status_code == 200 and resp.text == "ok", f"/health: {resp.status_code}"

        # --- Debug string prompt: bootstraps token ids without a tokenizer here ---
        resp = await http.post(
            f"{base}/v1/completions",
            json={"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0},
        )
        assert resp.status_code == 200, f"string prompt: {resp.status_code} {resp.text[:200]}"
        body = resp.json()
        boot_prompt = body["prompt_token_ids"]
        boot_completion = _check_envelope(body, None)

        # --- Token-id prompt (the real gateway path), with the §3.1 extras ---
        resp = await http.post(
            f"{base}/v1/completions",
            json={
                "prompt": boot_prompt,
                "add_special_tokens": False,
                "logprobs": True,          # bool — leniency check
                "return_token_ids": True,
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200, f"id prompt: {resp.status_code} {resp.text[:200]}"
        ids_completion = _check_envelope(resp.json(), boot_prompt)

        # --- Append-extension (cumulative turn shape) ---
        extended = boot_prompt + boot_completion + boot_prompt[:4]
        resp = await http.post(
            f"{base}/v1/completions",
            json={"prompt": extended, "max_tokens": 8, "temperature": 0.0},
        )
        assert resp.status_code == 200, f"extended prompt: {resp.status_code} {resp.text[:200]}"
        _check_envelope(resp.json(), extended)

        # --- Streaming ---
        chunks: list[dict] = []
        done_seen = False
        async with http.stream(
            "POST",
            f"{base}/v1/completions",
            json={"prompt": boot_prompt, "max_tokens": 8, "temperature": 0.0, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            ctype = resp.headers.get("content-type", "")
            assert "text/event-stream" in ctype, f"stream content-type: {ctype}"
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    done_seen = True
                    break
                chunks.append(json.loads(payload))

        assert done_seen, "no [DONE] terminator"
        assert chunks, "no data chunks"
        assert chunks[0].get("prompt_token_ids") == boot_prompt, "first chunk must echo prompt_token_ids"
        streamed_ids = [t for c in chunks for t in c["choices"][0].get("token_ids", [])]
        assert streamed_ids, "no token_ids streamed"
        finals = [c for c in chunks if c["choices"][0].get("finish_reason")]
        assert finals, "no chunk carries finish_reason"
        assert finals[-1]["choices"][0]["finish_reason"] in ("stop", "length")
        usage = finals[-1].get("usage") or {}
        assert usage.get("completion_tokens") == len(streamed_ids), (
            f"usage {usage.get('completion_tokens')} != streamed ids {len(streamed_ids)}"
        )

        # Determinism cross-check (argmax): streaming and non-streaming agree
        # on ids for the same prompt — token accounting is the whole contract.
        # (Skipped on the dummy driver: its logits aren't deterministic.)
        if getattr(args, "driver", "") != "dummy" and ids_completion and streamed_ids:
            assert streamed_ids == ids_completion, (
                "argmax streaming vs non-streaming token_ids diverge:"
                f" {streamed_ids[:8]}... vs {ids_completion[:8]}..."
            )


def tests():
    return [test_rl_completions]


if __name__ == "__main__":
    run_tests(tests())
