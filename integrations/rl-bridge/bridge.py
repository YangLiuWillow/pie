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
  GET  /health           200 {"status": "ok", "weight_version": n}
                         (gateway router polls every 10 s).
  GET/POST /admin/weight_version
                         read / advance the version stamped on responses,
                         after the trainer swaps pie's weights (R-G2/R-G3).

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

    def clear(self) -> None:
        self._lens.clear()


class Bridge:
    def __init__(self, pie_uri: str, inferlet: str, weight_version: int = 0) -> None:
        self.pie_uri = pie_uri
        self.inferlet = inferlet
        self.client: PieClient | None = None
        self.connected = False
        self._wasm: str | None = None
        self._manifest: str | None = None
        self.hints = PrefixHints()
        # Stamped onto every response. The gateway's own weight_version is
        # OVERRIDDEN by whatever the response body carries
        # (rllm_model_gateway.data_process.extract_weight_version), so this
        # number — not the trainer's — is what reaches the traces. It has to be
        # advanced whenever the served weights change.
        self.weight_version = weight_version

    def set_weight_version(self, weight_version: int) -> None:
        """Adopt new weights' version and drop every KV hint taken under the old
        ones. The hints name snapshot boundaries in a pie engine that a weight
        swap has restarted, so keeping them would send the inferlet looking for
        indices that no longer exist — and, worse, would be silent if a key
        happened to survive."""
        self.weight_version = weight_version
        self.hints.clear()

    async def connect(self, wasm: str | None, manifest: str | None) -> None:
        self._wasm, self._manifest = wasm, manifest
        await self._dial()

    async def _dial(self) -> None:
        self.client = PieClient(self.pie_uri)
        await self.client.connect()
        if self._wasm and self._manifest:
            await self.client.install_program(self._wasm, self._manifest, force_overwrite=True)
            log.info("installed %s from %s", self.inferlet, self._wasm)
        self.connected = True

    async def _redial(self) -> None:
        """Re-establish the WebSocket to pie after it drops.

        One connection was opened at startup and never renewed, so a single
        dropped socket bricked the bridge for its whole lifetime: every rollout
        failed instantly with "no close frame received or sent" while /health
        went on returning 200. A training run logged 784 such failures and a
        solve-rate probe measured three task sets as unsolvable — all of it
        against a connection that had been dead since the ninth request.
        """
        self.connected = False
        self.hints.clear()   # snapshots do not survive a pie restart
        try:
            if self.client is not None:
                await self.client.close()
        except Exception:
            pass
        await self._dial()
        log.warning("reconnected to pie at %s", self.pie_uri)

    async def rollout(self, inp: dict) -> dict:
        try:
            return await self._rollout_once(inp)
        except Exception as first:
            # Distinguishing a dropped socket from an inferlet error is not
            # worth guessing at: redial and retry once. A genuine inferlet
            # failure simply fails again, with its own message.
            log.warning("rollout failed (%s); redialing pie and retrying once", first)
            await self._redial()
            return await self._rollout_once(inp)

    async def _rollout_once(self, inp: dict) -> dict:
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


_QWEN3_TOOLS_HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
    "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_QWEN3_TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def render_tools_into_system(system: str, tools: list[dict]) -> str:
    """Replicate the Qwen3 chat template's hermes-style tools block (the
    rendering vLLM's template applies; chat.wit has no tools parameter, so the
    block rides inside the system text)."""
    lines = [_QWEN3_TOOLS_HEADER]
    for t in tools:
        fn = t.get("function", t)
        lines.append("\n" + json.dumps(fn, separators=(", ", ": "), ensure_ascii=False))
    lines.append(_QWEN3_TOOLS_FOOTER)
    return system + "".join(lines)


def parse_completion_text(text: str) -> tuple[str, str | None, list[dict]]:
    """Split raw completion text into (content, reasoning_content, tool_calls)
    — the vLLM reasoning-parser + hermes tool-parser shape qwen-code expects."""
    reasoning = None
    if "<think>" in text:
        head, _, rest = text.partition("<think>")
        thought, _, tail = rest.partition("</think>")
        reasoning = thought.strip() or None
        text = (head + tail).lstrip("\n")
    calls: list[dict] = []
    content_parts: list[str] = []
    rest = text
    while "<tool_call>" in rest:
        before, _, after = rest.partition("<tool_call>")
        content_parts.append(before)
        block, closed, rest = after.partition("</tool_call>")
        if not closed:
            rest = ""
            block = block.strip()
        try:
            obj = json.loads(block.strip())
            name = obj.get("name")
            args = obj.get("arguments", {})
            if name:
                calls.append({
                    "id": f"call_{int(time.time() * 1000):x}_{len(calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False)
                        if not isinstance(args, str) else args,
                    },
                })
        except json.JSONDecodeError:
            content_parts.append(block)  # malformed call: surface as text
    content_parts.append(rest)
    return "".join(content_parts).strip(), reasoning, calls


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


async def handle_chat(request: web.Request) -> web.StreamResponse:
    """§3.2 turn-0 path: messages in, chat shape out — but still carrying root
    prompt_token_ids and choices[0].token_ids for the training gateway."""
    bridge: Bridge = request.app["bridge"]
    try:
        body = await request.json()
    except Exception:
        return _error(400, "request body is not valid JSON")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "messages must be a non-empty list")

    tools = body.get("tools") or []
    if tools:
        messages = [dict(m) for m in messages]
        if messages and messages[0].get("role") == "system":
            sys_text = messages[0].get("content") or ""
            if isinstance(sys_text, list):
                sys_text = "".join(p.get("text", "") for p in sys_text)
            messages[0]["content"] = render_tools_into_system(sys_text, tools)
        else:
            messages.insert(0, {"role": "system",
                                "content": render_tools_into_system("", tools)})

    inp = {
        "messages": messages,
        "max_tokens": int(body.get("max_tokens", 256)),
        "temperature": float(body.get("temperature", 1.0)),
        "top_p": float(body.get("top_p", 1.0)),
        "seed": int(body.get("seed", 0)),
        "return_text": True,
    }
    try:
        out = await bridge.rollout(inp)
    except Exception as e:
        return _error(500, f"rollout failed: {e}", "server_error")

    prompt_ids = out.get("prompt_token_ids", [])
    if prompt_ids:
        bridge.hints.record(prompt_ids, out.get("saved_len", 0))

    content, reasoning, tool_calls = parse_completion_text(out.get("text", ""))
    finish_reason = "tool_calls" if tool_calls else out["finish_reason"]

    logprobs_obj = None
    if body.get("logprobs"):
        # Chat logprob shape: the gateway reads logprobs.content[].logprob.
        logprobs_obj = {
            "content": [{"token": "", "logprob": lp} for lp in out["logprobs"]]
        }
    usage = {
        "prompt_tokens": out["num_prompt_tokens"],
        "completion_tokens": out["num_output_tokens"],
        "total_tokens": out["num_prompt_tokens"] + out["num_output_tokens"],
        "prompt_tokens_details": {"cached_tokens": out.get("cached_tokens", 0)},
    }
    base = {
        "id": f"chatcmpl-{int(time.time() * 1000):x}",
        "created": int(time.time()),
        "model": body.get("model", "pie"),
        "weight_version": bridge.weight_version,
    }

    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    if not body.get("stream", False):
        choice = {
            "index": 0,
            "message": message,
            "token_ids": out["token_ids"],
            "finish_reason": finish_reason,
        }
        if logprobs_obj is not None:
            choice["logprobs"] = logprobs_obj
        return web.json_response({
            **base,
            "object": "chat.completion",
            "prompt_token_ids": prompt_ids,
            "choices": [choice],
            "usage": usage,
        })

    # SSE form (qwen-code always streams; a JSON body reads to it as a dead
    # stream — "Model stream ended without a finish reason" and 4 retries).
    # Buffered server-side: role chunk (carrying root prompt_token_ids for the
    # gateway's turn-0 trace), one content delta with token_ids + logprobs, a
    # finish chunk, usage, [DONE].
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await resp.prepare(request)

    async def emit(obj: dict) -> None:
        await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

    chunk_base = {**base, "object": "chat.completion.chunk"}
    await emit({**chunk_base, "prompt_token_ids": prompt_ids,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]})
    delta_body: dict = {"content": content}
    if reasoning is not None:
        delta_body["reasoning_content"] = reasoning
    if tool_calls:
        delta_body["tool_calls"] = [
            {"index": i, **tc} for i, tc in enumerate(tool_calls)
        ]
    delta = {"index": 0, "delta": delta_body,
             "token_ids": out["token_ids"], "finish_reason": None}
    if logprobs_obj is not None:
        delta["logprobs"] = logprobs_obj
    await emit({**chunk_base, "choices": [delta]})
    await emit({**chunk_base, "choices": [{"index": 0, "delta": {},
                                           "finish_reason": finish_reason}]})
    await emit({**chunk_base, "choices": [], "usage": usage})
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def handle_health(request: web.Request) -> web.Response:
    """Health means "I can serve", not "my process is running".

    Reporting ok while the link to pie was dead is what let three probe runs
    and part of a training run burn against a bridge that could not serve a
    single token — the router kept it in the fleet because /health said 200.
    """
    bridge: Bridge = request.app["bridge"]
    body = {
        "status": "ok" if bridge.connected else "degraded",
        "weight_version": bridge.weight_version,
        "pie": bridge.pie_uri,
        "connected": bridge.connected,
    }
    return web.json_response(body, status=200 if bridge.connected else 503)


async def handle_weight_version(request: web.Request) -> web.Response:
    """GET reports the served version; POST adopts a new one (R-G2/R-G3).

    The trainer calls POST after a weight swap. Restarting the bridge with
    ``--weight-version`` does the same thing; this endpoint exists so the swap
    can also be done without dropping the connection to pie."""
    bridge: Bridge = request.app["bridge"]
    if request.method == "POST":
        body = await request.json()
        try:
            version = int(body["weight_version"])
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(reason="weight_version must be an int")
        bridge.set_weight_version(version)
        log.info("weight_version -> %d (KV hints flushed)", version)
    return web.json_response({"weight_version": bridge.weight_version})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pie", default="ws://127.0.0.1:8080")
    ap.add_argument("--inferlet", default="rl-rollout@0.1.0")
    ap.add_argument("--wasm")
    ap.add_argument("--manifest")
    ap.add_argument("--weight-version", type=int, default=0,
                    help="version of the weights pie is serving; stamped onto "
                         "every response and onto the trainer's traces")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    bridge = Bridge(args.pie, args.inferlet, weight_version=args.weight_version)
    app = web.Application()
    app["bridge"] = bridge
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/completions", handle_completions)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/chat/completions", handle_chat)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/admin/weight_version", handle_weight_version)
    app.router.add_post("/admin/weight_version", handle_weight_version)

    async def on_startup(_: web.Application) -> None:
        await bridge.connect(args.wasm, args.manifest)
        log.info("bridge up: %s -> %s (%s)", args.port, args.pie, args.inferlet)

    app.on_startup.append(on_startup)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
