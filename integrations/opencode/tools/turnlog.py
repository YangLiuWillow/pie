#!/usr/bin/env python3
"""A transparent proxy that times every model call, identically on every engine.

## Why this exists

`run_swebench.py` reports one number per instance: agent wall clock. That number
answers "which stack finished first" and nothing else, and on this benchmark it
has already misled once -- the vLLM arm's 10-second runs read as "faster" when
they were an agent giving up after two calls. Wall clock confounds three
independent things:

    turns x (prefill + decode)

An engine can lose on wall clock by being slow at prefill, slow at decode, or by
provoking an agent into doing more turns. Those have nothing to do with each
other and three different fixes.

So this sits between opencode and whichever server is under test and records,
per `/v1/chat/completions` call:

  * **TTFT** -- request sent to first content byte. This is prefill, plus queue,
    plus whatever the engine does before it starts.
  * **decode wall** -- first content byte to last. This is generation.
  * **prompt_tokens / completion_tokens** -- from the server's OWN usage record,
    never estimated, so the rate is tokens the server says it made.

Same proxy, same clock, same fields, for pie, vLLM-metal and mlx-lm. That is the
whole point: `results-three-engine.md` is only honest because one harness asked
all three the same question, and this extends that property from throughput to
the agentic path.

## The streaming constraint

opencode streams, and the proxy MUST NOT buffer. A proxy that accumulates the
response and forwards it at the end reports a TTFT equal to the total, and --
worse -- changes the thing being measured, because the agent's own overlap with
generation disappears. Every write is flushed as it arrives.

## Getting usage out of a stream

An OpenAI-compatible stream carries no usage record unless asked. This injects
`stream_options: {"include_usage": true}` into streaming requests. If a server
rejects the field (4xx), the request is retried once VERBATIM and the call is
recorded with `usage: null` and delta-counted tokens instead -- degraded, and
labelled as degraded, rather than silently absent. An instrument that quietly
reports zero is the failure mode this repo has hit three times already.

Usage:
    python3 turnlog.py --upstream http://127.0.0.1:8080 --port 8099 \
        --out /tmp/turns-pie.jsonl [--tag pie]
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
LOCK = threading.Lock()


def record(row: dict) -> None:
    """One JSON object per line, flushed. A crashed run keeps its turns."""
    with LOCK:
        with open(ARGS.out, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()


def sse_events(chunk: bytes, carry: bytearray):
    """Yield complete `data:` payloads from a byte chunk, carrying a partial.

    SSE frames do not align with TCP reads. Parsing each read as if it were a
    whole frame drops the tail of every split frame -- which shows up as a
    completion-token undercount that grows with response length, i.e. exactly
    the shape of a plausible-looking wrong answer.
    """
    carry.extend(chunk)
    while b"\n\n" in carry:
        frame, _, rest = bytes(carry).partition(b"\n\n")
        carry[:] = rest
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                yield line[5:].strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # the access log is noise; the JSONL is the log
        pass

    def do_GET(self):
        self._forward(b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(n) if n else b"")

    def _forward(self, body: bytes):
        is_chat = self.path.rstrip("/").endswith("/chat/completions")
        streaming = False
        req = None
        if is_chat and body:
            try:
                req = json.loads(body)
                streaming = bool(req.get("stream"))
            except json.JSONDecodeError:
                req = None

        # Two injections, and they are NOT equally droppable.
        #
        # `stream_options` is bookkeeping: losing it costs a token count, and
        # the row says `usage_degraded` so the report can exclude it.
        #
        # `chat_template_kwargs` is CORRECTNESS. Qwen3.6 is a thinking model
        # whose template prefills `<think>` unless told otherwise, while pie's
        # renderer hard-codes the no-think cue. An arm that loses this field
        # answers a DIFFERENT PROMPT than the others, and the difference shows
        # up as accuracy and token count -- attributed to the engine, which did
        # nothing wrong. So it is dropped only after `stream_options` has
        # already been dropped, and when it is dropped the row is stamped
        # `template_dropped` so the report can refuse to compare that arm.
        probe = dict(req) if req is not None else None
        asked_usage = False
        asked_template = False
        if probe is not None and streaming and "stream_options" not in req:
            probe["stream_options"] = {"include_usage": True}
            asked_usage = True
        if probe is not None and ARGS.template_kwargs and "chat_template_kwargs" not in req:
            probe["chat_template_kwargs"] = ARGS.template_kwargs
            asked_template = True
        sent = json.dumps(probe).encode() if (asked_usage or asked_template) else body

        t0 = time.time()
        status, headers, conn = self._open(sent)

        # A 4xx here may be our injected fields, not the agent's request. Back
        # them off one at a time, most-droppable first, so the run continues.
        degraded = False
        template_dropped = False
        if asked_usage and 400 <= status < 500:
            conn.close()
            degraded = True
            retry = dict(req)
            if asked_template:
                retry["chat_template_kwargs"] = ARGS.template_kwargs
            t0 = time.time()
            status, headers, conn = self._open(json.dumps(retry).encode())
        if asked_template and 400 <= status < 500:
            conn.close()
            template_dropped = True
            t0 = time.time()
            status, headers, conn = self._open(body)

        self.send_response(status)
        for k, v in headers:
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        first_content = None
        deltas = 0
        usage = None
        finish = None
        carry = bytearray()
        raw = bytearray() if not streaming else None

        try:
            while True:
                # read1, NOT read. `HTTPResponse.read(n)` blocks until it has n
                # bytes or EOF, so it accumulates the whole generation and hands
                # it over at the end -- which reports TTFT == total and turns the
                # proxy into the buffering proxy this file exists to avoid. It
                # measured 0.751 s TTFT and 0.0003 s decode for 51 tokens before
                # this line was right.
                chunk = self._resp.read1(65536)
                if not chunk:
                    break
                # Forward FIRST, measure second: the agent's clock must not
                # wait on our bookkeeping.
                self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()

                if not is_chat:
                    continue
                if streaming:
                    for payload in sse_events(chunk, carry):
                        if payload == b"[DONE]":
                            continue
                        try:
                            ev = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if ev.get("usage"):
                            usage = ev["usage"]
                        for ch in ev.get("choices") or []:
                            d = ch.get("delta") or {}
                            text = d.get("content") or ""
                            tc = d.get("tool_calls")
                            if text or tc:
                                if first_content is None:
                                    first_content = time.time()
                                deltas += 1
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
                else:
                    raw.extend(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

        t1 = time.time()
        if not is_chat:
            return

        if not streaming and raw:
            try:
                ev = json.loads(bytes(raw))
                usage = ev.get("usage")
                ch = (ev.get("choices") or [{}])[0]
                finish = ch.get("finish_reason")
                deltas = 0
            except json.JSONDecodeError:
                pass

        comp = (usage or {}).get("completion_tokens")
        ttft = (first_content - t0) if first_content else None
        dec = (t1 - first_content) if first_content else None
        record({
            "tag": ARGS.tag,
            "t_start": t0,
            "status": status,
            "total_s": round(t1 - t0, 4),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "decode_s": round(dec, 4) if dec is not None else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": comp,
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get(
                "cached_tokens"),
            "delta_count": deltas,
            "decode_tok_s": (
                round(comp / dec, 2) if comp and dec and dec > 0 else None),
            "finish_reason": finish,
            "n_messages": len(req.get("messages") or []) if req else None,
            "usage_degraded": degraded,
            "template_kwargs": ARGS.template_kwargs or None,
            "template_dropped": template_dropped,
        })

    def _open(self, body: bytes):
        """Send upstream and return (status, headers, conn), response in hand."""
        conn = http.client.HTTPConnection(ARGS.host, ARGS.port_up, timeout=1800)
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection",
                                     "accept-encoding")}
        hdrs["Content-Length"] = str(len(body))
        hdrs["Accept-Encoding"] = "identity"
        conn.request(self.command, self.path, body=body, headers=hdrs)
        r = conn.getresponse()
        self._resp = r
        return r.status, r.getheaders(), conn


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="http://127.0.0.1:8080")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    # One control point for prompt shape across every arm. Setting this on the
    # servers instead would mean four different mechanisms (a pie renderer
    # constant, a vLLM flag, an mlx request field, a llama.cpp flag) and no
    # single place that proves they agree.
    p.add_argument("--template-kwargs", default=None,
                   help='JSON merged into every chat request, e.g. \'{"enable_thinking":false}\'')
    ARGS = p.parse_args()
    if ARGS.template_kwargs:
        ARGS.template_kwargs = json.loads(ARGS.template_kwargs)
    up = ARGS.upstream.split("://", 1)[-1]
    ARGS.host, _, port = up.partition(":")
    ARGS.port_up = int(port or 80)

    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    srv.daemon_threads = True
    print(f"turnlog: :{ARGS.port} -> {ARGS.host}:{ARGS.port_up}  log={ARGS.out}",
          file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
