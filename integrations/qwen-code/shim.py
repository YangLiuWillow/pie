#!/usr/bin/env python3
"""OpenAI /v1/chat/completions shim fronting a pie chat-completions inferlet.

The rewritten engine removed in-guest HTTP serving (world.wit imports only
wasi:http/client), so the OpenAI surface lives here: a stdlib-only asyncio
HTTP server that owns one WebSocket session to `pie serve` and one
long-lived chat-completions inferlet process.

Per request: signal the process with {"req_id", "body"}, then translate the
inferlet's {"req_id", "event": "chunk"|"done"|"error", "data"} messages into
SSE `chat.completion.chunk` frames (or one aggregated JSON response for
stream:false). The shim is pure transport: all OpenAI semantics (chunk
shapes, finish_reason, usage, tool-call framing) are produced inferlet-side.

Shim-owned wire obligations (docs/qwen-code-rl-audit.md §1):
  - `Content-Type: text/event-stream` on streaming 200s, `data: [DONE]`
    terminator, `: ping` keepalives while the model is busy (qwen-code
    aborts after 240 s without bytes).
  - 400 + OpenAI error JSON for malformed requests; 500 reserved for
    genuine faults (500s trigger 7×-app × 3×-SDK retry storms).

Usage:
    python3 shim.py --pie ws://127.0.0.1:18080 --port 8123 \
                    --wasm <chat_completions.wasm> --manifest <Pie.toml>
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

from pie_client import PieClient, Event  # noqa: E402

KEEPALIVE_S = 15
LOG = lambda *a: print("[shim]", *a, file=sys.stderr, flush=True)


def openai_error(message, err_type="invalid_request_error", param=None, code=None):
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


class InferletBridge:
    """Owns the pie connection and the single long-lived inferlet process."""

    def __init__(self, pie_uri, identity, wasm, manifest, inferlet):
        self.pie_uri = pie_uri
        self.identity = identity
        self.wasm = wasm
        self.manifest = manifest
        self.inferlet = inferlet
        self.client = None
        self.proc = None
        self.queues = {}  # req_id -> asyncio.Queue of (event, data)
        self.turn_lock = asyncio.Lock()  # inferlet handles one request at a time
        self._reader = None

    async def start(self):
        await self._connect_client()
        await self.client.install_program(self.wasm, self.manifest, force_overwrite=True)
        await self._launch()

    async def _connect_client(self):
        self.client = PieClient(self.pie_uri, identity=self.identity)
        await self.client.connect()
        await self.client.authenticate("shim")

    async def _launch(self):
        self.proc = await self.client.launch_process(self.inferlet, input={}, capture_outputs=True)
        LOG(f"inferlet up: {self.inferlet} process={self.proc.process_id}")
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
                    LOG(f"inferlet {event.value}: {payload}".rstrip())
                elif event in (Event.Return, Event.Error):
                    # Terminal: the client-side queue is gone; must relaunch.
                    LOG(f"inferlet terminal event {event.value}: {payload!r:.500}")
                    break
        except Exception as e:  # connection torn down, etc.
            LOG(f"reader stopped: {e!r}")
        # Fail all in-flight requests, then relaunch for the next one.
        for q in self.queues.values():
            q.put_nowait(("error", {"status": 500,
                                    "error": openai_error("inferlet exited", "server_error")["error"]}))
        self.queues.clear()
        # The WS itself may be the casualty (gateway restart, silence kill):
        # relaunch needs a live client first, so rebuild the whole chain with
        # backoff rather than assuming the connection survived the process.
        self.proc = None
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
        LOG("giving up on relaunch; next request will retry via _launch")

    async def request(self, body):
        """Async generator of (event, data) for one chat-completion request."""
        req_id = uuid.uuid4().hex
        q = asyncio.Queue()
        self.queues[req_id] = q
        try:
            async with self.turn_lock:
                if self.proc is None:
                    await self._launch()
                await self.proc.signal(json.dumps(
                    {"req_id": req_id, "body": body, "now": int(time.time())}))
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
        head += [f"Content-Type: {ctype}", "Access-Control-Allow-Origin: *",
                 "Cache-Control: no-cache", "Connection: close"]
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
    """Fold chunk objects into one chat.completion response (stream:false)."""
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
                slot = tool_calls.setdefault(tc.get("index", 0), {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
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
    out = {"id": cid or "chatcmpl-0", "object": "chat.completion", "model": model or "pie",
           "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}]}
    if usage:
        out["usage"] = usage
    return out


async def handle(bridge, reader, writer):
    http = Http(reader, writer)
    try:
        req = await http.read_request()
        if req is None:
            writer.close()
            return
        method, path, headers, body = req

        if method == "OPTIONS":
            http.start_response("204 No Content", "text/plain", extra=(
                "Access-Control-Allow-Methods: POST, GET, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type, Authorization"))
            await http.finish()
            return
        if method == "GET":
            http.start_response("200 OK", "application/json")
            http.send(json.dumps({"status": "ok", "service": "pie chat-completions shim"}).encode())
            await http.finish()
            return
        if method != "POST" or path not in ("/chat/completions", "/v1/chat/completions"):
            http.start_response("404 Not Found", "application/json")
            http.send(json.dumps(openai_error(f"no route {method} {path}", code="not_found")).encode())
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
                # Inferlet envelope: data = {"status": u16, "error": {...}}
                data = data if isinstance(data, dict) else {}
                status = data.get("status") or 500
                err = data.get("error") or openai_error(str(data), "server_error")["error"]
                if started:
                    # Stream already going: best effort — emit and close.
                    http.send(("data: " + json.dumps({"error": err}) + "\n\n").encode())
                else:
                    http.start_response(f"{status} Error", "application/json")
                    http.send(json.dumps({"error": err}).encode())
                await http.finish()
                return
            if event == "response":
                # Non-streaming: the inferlet assembled the full chat.completion.
                http.start_response("200 OK", "application/json")
                http.send(json.dumps(data).encode())
                await http.finish()
                return
            if event == "chunk":
                if stream:
                    if not started:
                        http.start_response("200 OK", "text/event-stream", chunked=True)
                        started = True
                    http.send(("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode())
                    await writer.drain()
                else:
                    collected.append(data)
            elif event == "done":
                if stream:
                    if not started:
                        http.start_response("200 OK", "text/event-stream", chunked=True)
                        started = True
                    http.send(b"data: [DONE]\n\n")
                else:
                    http.start_response("200 OK", "application/json")
                    http.send(json.dumps(aggregate(collected)).encode())
                await http.finish()
                return
        # Generator ended without done/error (relaunch path already queued an
        # error for us normally; this is belt-and-braces).
        if not started:
            http.start_response("500 Error", "application/json")
            http.send(json.dumps(openai_error("stream ended unexpectedly", "server_error")).encode())
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
    ap.add_argument("--identity", default="default/shim")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--wasm", default=str(
        REPO / "tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm"))
    ap.add_argument("--manifest", default=str(
        REPO / "tests/inferlets/chat-completions/Pie.toml"))
    ap.add_argument("--inferlet", default="chat-completions@0.1.0")
    args = ap.parse_args()

    bridge = InferletBridge(args.pie, args.identity, args.wasm, args.manifest, args.inferlet)
    await bridge.start()

    server = await asyncio.start_server(
        lambda r, w: handle(bridge, r, w), args.host, args.port)
    LOG(f"OpenAI shim on http://{args.host}:{args.port}/v1/chat/completions -> {args.pie}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
