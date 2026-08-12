#!/usr/bin/env python3
"""Tiny recording server for auditing OpenClaw's OpenAI-compatible wire traffic.

Adapted from ../opencode/record_server.py (same conventions: every request is
dumped to wire/req-NNN.json as {"seq","method","path","headers","body"};
POST */chat/completions gets a minimal valid streaming response). Differences:

- Tool-call arguments are synthesized from the request's OWN tool schema
  (OpenClaw's `read` tool need not use opencode's `filePath` key): required
  string properties whose name looks path-like get $RECORD_TOOL_FILE, other
  required props get type-appropriate placeholders.
- GET */models answers with a one-model list (OpenClaw's self-hosted provider
  setup probes it for discovery; opencode never did).

Modes (env RECORD_MODE):
  text  (default) - always answer with a single text delta "Hello."
  tool            - if the incoming request does NOT yet contain a role:"tool"
                    message, answer with a tool call (prefer `read`); the
                    follow-up request then carries the history-replay shapes
                    (assistant tool_calls message + role:"tool" result).

Env:
  RECORD_MODE       text | tool (default text)
  RECORD_TOOL_FILE  absolute path used for path-like tool arguments
  RECORD_DIR        where to write wire/req-NNN.json (default: dir of this file)
  RECORD_PORT       default 8123

The SSE stream deliberately includes a `: keepalive` comment line so the
capture also verifies the client's SSE parser tolerates comment lines.
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

PREFERRED_TOOLS = ["read", "Read", "fs_read", "cat"]
PATHISH = re.compile(r"path|file", re.IGNORECASE)


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


def placeholder_for(name, schema):
    t = schema.get("type")
    if t == "string":
        return TOOL_FILE if PATHISH.search(name) else "hello"
    if t == "integer" or t == "number":
        return schema.get("minimum", 1)
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return "hello"


def synth_arguments(fn):
    """Build a schema-valid arguments object from the tool's own parameters."""
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    args = {}
    for key in required:
        args[key] = placeholder_for(key, props.get(key, {}))
    if not args:  # no required props: still try a path-like optional one
        for key, schema in props.items():
            if PATHISH.search(key) and schema.get("type") == "string":
                args[key] = TOOL_FILE
                break
    return json.dumps(args)


def pick_tool(body):
    """Pick a real tool from the request's own tools array (prefer read)."""
    fns = {
        t["function"]["name"]: t["function"]
        for t in body.get("tools") or []
        if t.get("type") == "function" and t.get("function", {}).get("name")
    }
    if not fns:
        return None, None
    for name in PREFERRED_TOOLS:
        if name in fns:
            return name, synth_arguments(fns[name])
    # No read-like tool in the surface (e.g. lean mode): don't synthesize a
    # call to an arbitrary tool — exec-like tools would run the placeholder
    # as a real command. Fall back to a text answer instead.
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
