#!/usr/bin/env python3
"""pie-openai-bridge — OpenAI-compatible worker front for pie 0.5 RL rollouts.

Pie 0.5 removed guest-side HTTP servers, so the OpenAI wire envelope the rllm
gateway expects (pie-rl-verl-integration.md §3) lives here instead: a small
host-side sidecar that translates each HTTP request into one
`launch_process(rl-rollout)` call over the pie client WebSocket.

Serves (the training-relevant subset):
  POST /v1/completions   §3.1 cumulative token mode — `prompt` as list[int]
                         (string accepted for debugging), exact token-id echo,
                         per-token logprobs, weight_version, usage with
                         prompt_tokens_details.cached_tokens. `stream` gives
                         the SSE form (buffered server-side: first chunk
                         prompt_token_ids, one delta chunk, usage, [DONE]).
  GET  /health           200 "ok" (gateway router polls every 10 s).

KV reuse: the bridge remembers each response's saved boundary length, keyed by
a fingerprint of the prompt head (all turns of a lineage share it, cumulative
mode guarantees prefix extension), and passes candidates as `saved_lens`.
The inferlet verifies candidates content-addressed, so stale hints are misses.

Run:  python3 bridge.py --port 8123 --pie ws://127.0.0.1:8080 \
          [--wasm .../rl_rollout.wasm --manifest .../Pie.toml]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from aiohttp import web

from pie_client import Event, PieClient

log = logging.getLogger("pie-openai-bridge")

FINGERPRINT_TOKENS = 16
MAX_HINTS = 8
MAX_LINEAGES = 4096


class PrefixHints:
    """saved_len bookkeeping per conversation lineage.

    Lineage key = the first FINGERPRINT_TOKENS token ids: cumulative token
    mode extends prompts append-only, so every turn of a lineage shares its
    head. Correctness never depends on this map — the inferlet hash-verifies
    every candidate — it only decides how many candidates are worth sending.
    """

    def __init__(self) -> None:
        self._lens: dict[tuple, list[int]] = {}

    def key(self, prompt: list[int]) -> tuple:
        return tuple(prompt[:FINGERPRINT_TOKENS])

    def candidates(self, prompt: list[int]) -> list[int]:
        lens = self._lens.get(self.key(prompt), [])
        n = len(prompt)
        return [l for l in sorted(lens, reverse=True) if l <= n][:MAX_HINTS]

    def record(self, prompt: list[int], saved_len: int) -> None:
        if saved_len <= 0:
            return
        if len(self._lens) >= MAX_LINEAGES:
            self._lens.clear()  # crude bound; hints are best-effort
        lens = self._lens.setdefault(self.key(prompt), [])
        if saved_len not in lens:
            lens.append(saved_len)
            del lens[:-MAX_HINTS]


class Bridge:
    def __init__(self, pie_uri: str, inferlet: str) -> None:
        self.pie_uri = pie_uri
        self.inferlet = inferlet
        self.client: PieClient | None = None
        self.hints = PrefixHints()
        self.weight_version = 0

    async def connect(self, wasm: str | None, manifest: str | None) -> None:
        self.client = PieClient(self.pie_uri)
        await self.client.connect()
        if wasm and manifest:
            await self.client.install_program(wasm, manifest, force_overwrite=True)
            log.info("installed %s from %s", self.inferlet, wasm)

    async def rollout(self, inp: dict) -> dict:
        proc = await self.client.launch_process(self.inferlet, input=inp)
        last = None
        while True:
            event, msg = await proc.recv()
            if event == Event.Return:
                last = msg
                break
            if event == Event.Error:
                raise RuntimeError(str(msg))
        return json.loads(last if isinstance(last, str) else last.decode())


def _error(status: int, message: str, etype: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": etype, "param": None, "code": None}},
        status=status,
    )


async def handle_completions(request: web.Request) -> web.StreamResponse:
    bridge: Bridge = request.app["bridge"]
    try:
        body = await request.json()
    except Exception:
        return _error(400, "request body is not valid JSON")

    prompt = body.get("prompt")
    prompt_tokens: list[int] | None = None
    prompt_text: str | None = None
    if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
        prompt_tokens = prompt
    elif isinstance(prompt, str):
        prompt_text = prompt
    else:
        return _error(400, "prompt must be a list of token ids or a string")

    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 1.0))
    top_p = float(body.get("top_p", 1.0))
    seed = int(body.get("seed", 0))
    want_logprobs = bool(body.get("logprobs"))  # bool OR int, both truthy forms
    stream = bool(body.get("stream", False))

    inp: dict = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "return_text": True,
    }
    if prompt_tokens is not None:
        inp["prompt_tokens"] = prompt_tokens
        inp["saved_lens"] = bridge.hints.candidates(prompt_tokens)
    else:
        inp["prompt"] = prompt_text

    try:
        out = await bridge.rollout(inp)
    except Exception as e:
        return _error(500, f"rollout failed: {e}", "server_error")

    if prompt_tokens is not None:
        bridge.hints.record(prompt_tokens, out.get("saved_len", 0))
        echo_prompt_ids = prompt_tokens
    else:
        # String-prompt debug path: the inferlet templated it; echo what it
        # counted (ids unavailable — cumulative training mode never sends
        # strings, so this stays a debugging convenience).
        echo_prompt_ids = []

    token_ids = out["token_ids"]
    completion = {
        "text": out.get("text", ""),
        "index": 0,
        "token_ids": token_ids,
        "finish_reason": out["finish_reason"],
    }
    if want_logprobs:
        completion["logprobs"] = {"token_logprobs": out["logprobs"]}
    usage = {
        "prompt_tokens": out["num_prompt_tokens"],
        "completion_tokens": out["num_output_tokens"],
        "total_tokens": out["num_prompt_tokens"] + out["num_output_tokens"],
        "prompt_tokens_details": {"cached_tokens": out.get("cached_tokens", 0)},
    }
    base = {
        "id": f"cmpl-{int(time.time() * 1000):x}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.get("model", "pie"),
        "weight_version": bridge.weight_version,
        "prompt_token_ids": echo_prompt_ids,
    }

    if not stream:
        return web.json_response({**base, "choices": [completion], "usage": usage})

    # SSE form, buffered: shape-per-§3.1 (first chunk prompt_token_ids, then
    # one full delta, then usage, then [DONE]).
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await resp.prepare(request)

    async def emit(obj: dict) -> None:
        await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

    head = dict(completion)
    head_lp = head.pop("logprobs", None)
    await emit({**base, "choices": [{"text": "", "index": 0, "token_ids": [],
                                     "finish_reason": None}]})
    delta = {"text": completion["text"], "index": 0, "token_ids": token_ids,
             "finish_reason": completion["finish_reason"]}
    if head_lp is not None:
        delta["logprobs"] = head_lp
    await emit({**base, "choices": [delta]})
    await emit({**base, "choices": [], "usage": usage})
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def handle_health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pie", default="ws://127.0.0.1:8080")
    ap.add_argument("--inferlet", default="rl-rollout@0.1.0")
    ap.add_argument("--wasm")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    bridge = Bridge(args.pie, args.inferlet)
    app = web.Application()
    app["bridge"] = bridge
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/completions", handle_completions)
    app.router.add_get("/health", handle_health)

    async def on_startup(_: web.Application) -> None:
        await bridge.connect(args.wasm, args.manifest)
        log.info("bridge up: %s -> %s (%s)", args.port, args.pie, args.inferlet)

    app.on_startup.append(on_startup)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
