#!/usr/bin/env python3
"""Tiny recording server for auditing opencode's OpenAI-compatible wire traffic.

Listens on 127.0.0.1:8123. Every request is dumped to wire/req-NNN.json as
{"seq", "method", "path", "headers", "body"}. POST /v1/chat/completions gets a
minimal valid streaming response (chat.completion.chunk SSE).

Modes (env RECORD_MODE):
  text  (default) - always answer with a single text delta "Hello."
  tool            - if the incoming request does NOT yet contain a role:"tool"
                    message, answer with a tool call for a real tool taken from
                    the request's own tools array (prefers "read", arguments
                    {"filePath": $RECORD_TOOL_FILE}); once the follow-up request
                    arrives carrying the tool result history, answer with text.
                    This captures the history-replay shapes (assistant
                    tool_calls message + role:"tool" result message).

Env:
  RECORD_MODE       text | tool (default text)
  RECORD_TOOL_FILE  absolute path used as the read tool's filePath argument
  RECORD_DIR        where to write wire/req-NNN.json (default: dir of this file)
  RECORD_PORT       default 8123

The SSE stream deliberately includes a `: keepalive` comment line so the
capture also verifies that the AI SDK's SSE parser tolerates comment lines.
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", HERE)
WIRE_DIR = os.path.join(RECORD_DIR, "wire")
MODE = os.environ.get("RECORD_MODE", "text")
TOOL_FILE = os.environ.get("RECORD_TOOL_FILE", "/tmp/hello.txt")
PORT = int(os.environ.get("RECORD_PORT", "8123"))

os.makedirs(WIRE_DIR, exist_ok=True)


def next_seq() -> int:
    seqs = [
        int(m.group(1))
        for f in os.listdir(WIRE_DIR)
        if (m := re.match(r"req-(\d+)\.json$", f))
    ]
    return max(seqs, default=0) + 1


def sse(chunks, include_done=True):
    out = []
    for c in chunks:
        if isinstance(c, str):  # raw line (e.g. comment)
            out.append(c + "\n\n")
        else:
            out.append("data: " + json.dumps(c) + "\n\n")
    if include_done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def base_chunk(**extra):
    d = {
        "id": "chatcmpl-record-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    d.update(extra)
    return d


def delta_chunk(delta, finish=None):
    c = base_chunk()
    c["choices"][0]["delta"] = delta
    c["choices"][0]["finish_reason"] = finish
    return c


def usage_chunk(prompt=100, completion=5, cached=0):
    c = base_chunk()
    c["choices"] = []
    c["usage"] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    return c


def text_response():
    return sse(
        [
            ": keepalive",  # SSE comment line — parser must ignore it
            delta_chunk({"role": "assistant"}),
            delta_chunk({"content": "Hello."}),
            delta_chunk({}, finish="stop"),
            usage_chunk(),
        ]
    )


def pick_tool(body):
    """Pick a real tool from the request's own tools array (prefer read)."""
    tools = body.get("tools") or []
    names = [
        t.get("function", {}).get("name")
        for t in tools
        if t.get("type") == "function"
    ]
    if "read" in names:
        return "read", json.dumps({"filePath": TOOL_FILE})
    if names:
        return names[0], "{}"
    return None, None


def tool_response(body):
    name, args = pick_tool(body)
    if name is None:
        return text_response()
    return sse(
        [
            ": keepalive",
            delta_chunk({"role": "assistant"}),
            delta_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_record_001",
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                }
            ),
            delta_chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": args}}
                    ]
                }
            ),
            delta_chunk({}, finish="tool_calls"),
            usage_chunk(prompt=120, completion=15, cached=64),
        ]
    )


def has_tool_result(body):
    return any(m.get("role") == "tool" for m in body.get("messages", []))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _record(self, body_bytes):
        seq = next_seq()
        try:
            body = json.loads(body_bytes) if body_bytes else None
        except json.JSONDecodeError:
            body = body_bytes.decode(errors="replace")
        rec = {
            "seq": seq,
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": body,
        }
        path = os.path.join(WIRE_DIR, f"req-{seq:03d}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[record] {self.command} {self.path} -> {path}", flush=True)
        return body

    def _respond_sse(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_json(self, obj, status=200):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self._record(self.rfile.read(length))
        if self.path.endswith("/chat/completions") and isinstance(body, dict):
            if MODE == "tool" and not has_tool_result(body):
                self._respond_sse(tool_response(body))
            else:
                self._respond_sse(text_response())
        else:
            self._respond_json({"error": {"message": "unexpected path"}}, 404)

    def do_GET(self):
        self._record(b"")
        if self.path.endswith("/models"):
            self._respond_json(
                {"object": "list", "data": [{"id": "test-model", "object": "model"}]}
            )
        else:
            self._respond_json({"ok": True})

    def log_message(self, *a):  # quiet default access log
        pass


if __name__ == "__main__":
    print(f"[record] mode={MODE} port={PORT} wire={WIRE_DIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
