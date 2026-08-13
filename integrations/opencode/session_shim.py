#!/usr/bin/env python3
"""OpenAI `/v1/chat/completions` shim fronting the long-lived opencode-session
inferlet (Strategy B).

Stock opencode speaks OpenAI over HTTP. The session inferlet is a *process*
reached over a WebSocket. This shim is the adapter between those two facts, and
nothing more: it owns one pie connection and one long-lived inferlet process,
multiplexes HTTP requests onto it by `req_id`, and translates the inferlet's
envelope events back into SSE frames.

**It is deliberately pure transport.** Every OpenAI semantic — chunk shapes,
finish_reason, tool-call framing, usage, the 400-vs-500 discipline — is produced
inferlet-side by `pie-openai-serving`, the same crate Strategy A serves from.
That is what keeps the A/B honest: the two arms differ in where the KV lives,
not in how the wire is rendered. Any wire logic that creeps in here becomes a
confound in the measurement.

## Why a shim at all, rather than a gateway route

Strategy A goes through `gateway/src/ingress/openai.rs`, which launches one
inferlet per request. That mechanism cannot express "keep talking to the process
you launched last turn" — and the gateway's OpenAI ingress is shared with the
frozen Strategy A arm, so teaching it sessions would put the control at risk.
The gateway already exposes exactly what is needed (`GET /v1/ws` with sticky
affinity, `launch_process`, `signal_process`), so the session path needs no
gateway change at all.

## Shim-owned wire obligations

  - `Content-Type: text/event-stream` on streaming 200s, a `data: [DONE]`
    terminator, and `: ping` keepalives while the model is busy. opencode's
    watchdog resets on raw bytes, upstream of its SSE parser, so a comment frame
    is a valid keepalive (verified against its source).
  - 400 + an OpenAI error body for malformed requests. **500 is reserved for
    genuine faults**: opencode retries 5xx without bound.

Usage:
    python3 session_shim.py --pie ws://127.0.0.1:18080 --port 8123 \
        --wasm <opencode_session.wasm> --manifest <Pie.toml>
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "client" / "python" / "src"))

from pie_client import Event, PieClient  # noqa: E402

# How long to wait on a silent inferlet before emitting an SSE keepalive.
# Comfortably inside opencode's idle watchdog.
KEEPALIVE_S = 15

LOG = lambda *a: print("[session-shim]", *a, file=sys.stderr, flush=True)


def openai_error(message, err_type="invalid_request_error", param=None, code=None):
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


class InferletBridge:
    """Owns the pie connection and the single long-lived session process.

    One process serves every conversation this shim sees. That is safe because
    the inferlet keys its retained KV on the *content address* of the history,
    not on a client-supplied session id: two interleaved conversations simply
    occupy two entries in its branch map and never resume each other's state.

    SEAM: one process per opencode session (keyed off a session header) would
    bound memory per conversation and let a dead session's KV be reclaimed
    promptly rather than by LRU. Correctness does not depend on it.
    """

    def __init__(self, pie_uri, identity, wasm, manifest, inferlet):
        self.pie_uri = pie_uri
        self.identity = identity
        self.wasm = wasm
        self.manifest = manifest
        self.inferlet = inferlet
        self.client = None
        self.proc = None
        self.queues = {}  # req_id -> asyncio.Queue of (event, data)
        # The inferlet serves one turn at a time: its turns share ONE working
        # set, so two concurrent turns would interleave writes into the same KV
        # prefix. This lock is the client-side half of that contract.
        self.turn_lock = asyncio.Lock()
        self._reader = None

    async def start(self):
        await self._connect_client()
        await self.client.install_program(self.wasm, self.manifest, force_overwrite=True)
        await self._launch()

    async def _connect_client(self):
        self.client = PieClient(self.pie_uri, identity=self.identity)
        await self.client.connect()
        await self.client.authenticate("session-shim")

    async def _launch(self):
        self.proc = await self.client.launch_process(
            self.inferlet, input={}, capture_outputs=True
        )
        LOG(f"session inferlet up: {self.inferlet} process={self.proc.process_id}")
        self._reader = asyncio.create_task(self._pump())

    async def _pump(self):
        """Single reader: demux inferlet messages to per-request queues."""
        try:
            while True:
                event, payload = await self.proc.recv()
                if event == Event.Message:
                    try:
                        msg = json.loads(payload)
                    except (TypeError, ValueError):
                        LOG(f"unparseable inferlet message: {payload!r:.200}")
                        continue
                    q = self.queues.get(msg.get("req_id"))
                    if q is not None:
                        q.put_nowait((msg.get("event"), msg.get("data")))
                elif event in (Event.Stdout, Event.Stderr):
                    # The inferlet's per-turn `cached=… prefill=…` line comes
                    # through here; it is the resume measurement, so it is
                    # surfaced rather than swallowed.
                    LOG(f"inferlet {event.value}: {payload}".rstrip())
                elif event in (Event.Return, Event.Error):
                    LOG(f"inferlet terminal event {event.value}: {payload!r:.500}")
                    break
        except Exception as e:  # connection torn down, etc.
            LOG(f"reader stopped: {e!r}")

        # The process died. Every retained working set died with it, so the next
        # turn is a clean full rebuild — degraded to Strategy A's cost, not to
        # wrong output. Fail the in-flight turns, then rebuild the chain.
        for q in self.queues.values():
            q.put_nowait(
                ("error", {"status": 500, "error": openai_error(
                    "session inferlet exited", "server_error")["error"]})
            )
        self.queues.clear()
        self.proc = None
        # The WebSocket itself may be the casualty (gateway restart, silence
        # kill), so a relaunch needs a live client first: rebuild the whole
        # chain with backoff rather than assuming the connection survived.
        for delay in (0, 2, 5, 10, 30):
            await asyncio.sleep(delay)
            try:
                await self._launch()
                return
            except Exception:
                try:
                    await self._connect_client()
                    await self._launch()
                    return
                except Exception as e:
                    LOG(f"reconnect attempt failed: {e!r}")
        LOG("giving up on relaunch; the next request will retry")

    async def request(self, body):
        """Async generator of (event, data) for one chat-completion request."""
        req_id = uuid.uuid4().hex
        q = asyncio.Queue()
        self.queues[req_id] = q
        try:
            async with self.turn_lock:
                if self.proc is None:
                    await self._launch()
                await self.proc.signal(json.dumps({"req_id": req_id, "body": body}))
                while True:
                    try:
                        event, data = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_S)
                    except asyncio.TimeoutError:
                        yield ("keepalive", None)
                        continue
                    yield (event, data)
                    if event in ("done", "error"):
                        return
        finally:
            self.queues.pop(req_id, None)


# ---------------------------------------------------------------------------
# Minimal HTTP/1.1 server (stdlib-only; chunked responses for SSE)
# ---------------------------------------------------------------------------


class Http:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.chunked = False

    async def read_request(self):
        line = await self.reader.readline()
        if not line:
            return None
        try:
            method, path, _ = line.decode("latin-1").split(" ", 2)
        except ValueError:
            return None
        headers = {}
        while True:
            h = await self.reader.readline()
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        body = b""
        n = int(headers.get("content-length", 0) or 0)
        if n:
            body = await self.reader.readexactly(n)
        return method, path.split("?")[0], headers, body

    def start_response(self, status, ctype, chunked=False, extra=()):
        self.chunked = chunked
        head = [f"HTTP/1.1 {status}"]
        head += [
            f"Content-Type: {ctype}",
            "Access-Control-Allow-Origin: *",
            "Cache-Control: no-cache",
            "Connection: close",
        ]
        head += list(extra)
        if chunked:
            head.append("Transfer-Encoding: chunked")
        self.writer.write(("\r\n".join(head) + "\r\n\r\n").encode())

    def send(self, data: bytes):
        if self.chunked:
            self.writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            self.writer.write(data)

    async def finish(self):
        if self.chunked:
            self.writer.write(b"0\r\n\r\n")
        await self.writer.drain()
        self.writer.close()


def aggregate(chunks):
    """Fold chunk objects into one chat.completion response (stream:false).

    Only reached if a non-streaming request somehow gets a chunk stream; the
    inferlet answers `stream:false` with a single `response` event. Kept as a
    safety net so a protocol drift degrades to a valid body instead of a hang.
    """
    content, tool_calls, finish, role = [], {}, None, "assistant"
    model, cid, usage = None, None, None
    for c in chunks:
        model = c.get("model", model)
        cid = c.get("id", cid)
        usage = c.get("usage") or usage
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            role = d.get("role", role)
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0),
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                slot["function"]["arguments"] += fn.get("arguments", "")
            finish = ch.get("finish_reason") or finish
    message = {"role": role, "content": "".join(content)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    out = {
        "id": cid or "chatcmpl-0",
        "object": "chat.completion",
        "model": model or "pie",
        "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}],
    }
    if usage:
        out["usage"] = usage
    return out


async def handle(bridge, model_name, reader, writer):
    http = Http(reader, writer)
    try:
        req = await http.read_request()
        if req is None:
            writer.close()
            return
        method, path, headers, body = req

        if method == "OPTIONS":
            http.start_response(
                "204 No Content",
                "text/plain",
                extra=(
                    "Access-Control-Allow-Methods: POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers: Content-Type, Authorization",
                ),
            )
            await http.finish()
            return

        if method == "GET" and path in ("/v1/models", "/models"):
            # opencode probes this to validate the provider before its first
            # completion; a 404 here reads to it as a dead endpoint.
            http.start_response("200 OK", "application/json")
            http.send(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_name,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "pie",
                            }
                        ],
                    }
                ).encode()
            )
            await http.finish()
            return

        if method == "GET":
            http.start_response("200 OK", "application/json")
            http.send(
                json.dumps({"status": "ok", "service": "pie opencode-session shim"}).encode()
            )
            await http.finish()
            return

        if method != "POST" or path not in ("/chat/completions", "/v1/chat/completions"):
            http.start_response("404 Not Found", "application/json")
            http.send(
                json.dumps(openai_error(f"no route {method} {path}", code="not_found")).encode()
            )
            await http.finish()
            return

        # Same admission contract as Strategy A's ingress
        # (`gateway/src/ingress/openai.rs::extract_identity`): the trust-edge
        # header when present, otherwise a non-empty Bearer. The token is NOT
        # verified — key checking is an edge concern (private bind / mTLS /
        # edge proxy) in both arms. What matters is that the two arms REJECT
        # the same requests, or the acceptance suite is measuring the shim's
        # laxness rather than the server.
        #
        # SEAM: Strategy A also derives a per-token pie identity
        # (`default/key-<blake3[..8]>`) so per-key attribution survives. This
        # shim holds one WebSocket under one identity, so every caller shares
        # it; per-token attribution would need a connection per identity.
        auth = headers.get("authorization", "")
        bearer = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        if not headers.get("x-pie-identity") and not bearer:
            http.start_response("401 Unauthorized", "application/json")
            http.send(
                json.dumps(
                    openai_error(
                        "missing Authorization: Bearer or x-pie-identity",
                        "authentication_error",
                    )
                ).encode()
            )
            await http.finish()
            return

        try:
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as e:
            http.start_response("400 Bad Request", "application/json")
            http.send(json.dumps(openai_error(f"invalid JSON body: {e}")).encode())
            await http.finish()
            return

        stream = bool(parsed.get("stream"))
        started = False
        collected = []
        async for event, data in bridge.request(parsed):
            if event == "keepalive":
                if stream:
                    if not started:
                        http.start_response("200 OK", "text/event-stream", chunked=True)
                        started = True
                    http.send(b": ping\n\n")
                    await writer.drain()
                continue
            if event == "error":
                data = data if isinstance(data, dict) else {}
                status = data.get("status") or 500
                err = data.get("error") or openai_error(str(data), "server_error")["error"]
                if started:
                    # The stream is already open, so the status line is spent:
                    # emit the error as a frame and close.
                    http.send(("data: " + json.dumps({"error": err}) + "\n\n").encode())
                else:
                    http.start_response(f"{status} Error", "application/json")
                    http.send(json.dumps({"error": err}).encode())
                await http.finish()
                return
            if event == "response":
                http.start_response("200 OK", "application/json")
                http.send(json.dumps(data).encode())
                await http.finish()
                return
            if event == "chunk":
                if stream:
                    if not started:
                        http.start_response("200 OK", "text/event-stream", chunked=True)
                        started = True
                    http.send(
                        ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()
                    )
                    await writer.drain()
                else:
                    collected.append(data)
            elif event == "done":
                if stream:
                    if not started:
                        http.start_response("200 OK", "text/event-stream", chunked=True)
                        started = True
                    http.send(b"data: [DONE]\n\n")
                    await http.finish()
                    return
                if collected:
                    http.start_response("200 OK", "application/json")
                    http.send(json.dumps(aggregate(collected)).encode())
                    await http.finish()
                    return
                # `done` with nothing before it on a non-streaming request: the
                # turn produced no body. Answering 200-with-nothing would look
                # like success, so say what happened.
                http.start_response("500 Error", "application/json")
                http.send(
                    json.dumps(openai_error("turn produced no response", "server_error")).encode()
                )
                await http.finish()
                return

        if not started:
            http.start_response("500 Error", "application/json")
            http.send(
                json.dumps(openai_error("stream ended unexpectedly", "server_error")).encode()
            )
        await http.finish()
    except (ConnectionResetError, asyncio.IncompleteReadError):
        writer.close()
    except Exception as e:
        LOG(f"handler error: {e!r}")
        try:
            http.start_response("500 Error", "application/json")
            http.send(json.dumps(openai_error(str(e), "server_error")).encode())
            await http.finish()
        except Exception:
            writer.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie", default="ws://127.0.0.1:18080")
    ap.add_argument("--identity", default="default/session-shim")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument(
        "--wasm",
        default=str(REPO / "target/wasm32-wasip2/release/opencode_session.wasm"),
    )
    ap.add_argument("--manifest", default=str(REPO / "inferlets/opencode-session/Pie.toml"))
    ap.add_argument("--inferlet", default="opencode-session@0.1.0")
    ap.add_argument(
        "--model-name",
        default="pie",
        help="the id reported by /v1/models (opencode matches its config against it)",
    )
    args = ap.parse_args()

    bridge = InferletBridge(args.pie, args.identity, args.wasm, args.manifest, args.inferlet)
    await bridge.start()

    server = await asyncio.start_server(
        lambda r, w: handle(bridge, args.model_name, r, w), args.host, args.port
    )
    LOG(
        f"OpenAI session shim on http://{args.host}:{args.port}/v1/chat/completions "
        f"-> {args.pie} ({args.inferlet})"
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
