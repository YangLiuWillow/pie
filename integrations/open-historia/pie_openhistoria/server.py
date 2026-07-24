"""aiohttp server exposing an OpenAI-compatible surface backed by Pie.

Endpoints (both the `/v1`-prefixed and bare forms, since Open Historia lets the
user configure either):

    POST /v1/chat/completions   (and /chat/completions)   — the workhorse
    GET  /v1/models             (and /models)             — discovery
    GET  /health

CORS is wide open on every response. The self-hosted Open Historia desktop
build actually reaches this through its own same-origin `/api/ai/relay`
(server.js:565) which defeats CORS server-side, but the hosted web build calling
a local adapter needs these headers — mirroring the `OLLAMA_ORIGINS` guidance in
`main.jsx:321`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

from aiohttp import web

from . import backend, translate


@dataclass
class AdapterConfig:
    pie_uri: str = "ws://127.0.0.1:8080"
    pie_username: str = "local-dev"
    inferlet: str = "openhands-completion@0.1.0"
    # Advertised model id. Informational to the inferlet (the runtime picks its
    # first loaded model), but Open Historia shows/stores it and sends it back.
    model_id: str = "pie"
    request_timeout_s: float = 600.0
    host: str = "127.0.0.1"
    port: int = 8000


def _cors(resp: web.StreamResponse) -> web.StreamResponse:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = (
        "authorization, content-type, x-api-key, anthropic-version"
    )
    return resp


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        return _cors(web.Response(status=204))
    try:
        resp = await handler(request)
    except web.HTTPException as exc:
        return _cors(exc)
    return _cors(resp)


def _json_error(status: int, message: str, err_type: str = "adapter_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type}}, status=status
    )


async def handle_models(request: web.Request) -> web.Response:
    cfg: AdapterConfig = request.app["config"]
    return web.json_response({
        "object": "list",
        "data": [{
            "id": cfg.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "pie",
        }],
    })


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _run_inferlet(cfg: AdapterConfig, body: dict) -> dict:
    input_payload = translate.chat_request_to_inferlet_input(body)
    return await backend.complete(
        pie_uri=cfg.pie_uri,
        pie_username=cfg.pie_username,
        inferlet=cfg.inferlet,
        input_payload=input_payload,
        timeout_s=cfg.request_timeout_s,
    )


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    cfg: AdapterConfig = request.app["config"]
    try:
        body = await request.json()
    except Exception:
        return _json_error(400, "Request body must be valid JSON.", "invalid_request_error")

    model = body.get("model") or cfg.model_id
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    stream = bool(body.get("stream"))

    try:
        out = await _run_inferlet(cfg, body)
    except Exception as exc:  # surface Pie/connection failures as a 502
        return _json_error(502, f"Pie backend error: {exc}", "upstream_error")

    if not stream:
        return web.json_response(translate.inferlet_output_to_chat_completion(
            out, model=model, request_id=request_id, created=created,
        ))

    # SSE. NOTE: buffered-then-replayed (see translate.inferlet_output_to_stream_chunks).
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    _cors(resp)
    await resp.prepare(request)
    for chunk in translate.inferlet_output_to_stream_chunks(
        out, model=model, request_id=request_id, created=created,
    ):
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def build_app(config: AdapterConfig) -> web.Application:
    app = web.Application(middlewares=[cors_middleware], client_max_size=32 * 1024 * 1024)
    app["config"] = config
    for prefix in ("", "/v1"):
        app.router.add_post(f"{prefix}/chat/completions", handle_chat_completions)
        app.router.add_get(f"{prefix}/models", handle_models)
    app.router.add_get("/health", handle_health)
    # Catch-all OPTIONS so preflight to any path gets CORS headers.
    app.router.add_route("OPTIONS", "/{tail:.*}", lambda r: _cors(web.Response(status=204)))
    return app


def run(config: AdapterConfig) -> None:
    app = build_app(config)
    print(
        f"pie-openhistoria adapter → Pie {config.pie_uri} "
        f"(inferlet {config.inferlet}), serving OpenAI API on "
        f"http://{config.host}:{config.port}/v1",
        flush=True,
    )
    web.run_app(app, host=config.host, port=config.port, print=None)
