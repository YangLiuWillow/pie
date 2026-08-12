#!/usr/bin/env python3
"""Live acceptance tests for the chat-completions daemon.

Asserts every row of the audit §1 hard-requirements table
(docs/qwen-code-rl-audit.md) against a running daemon with raw HTTP —
no OpenAI SDK, so the assertions are on actual wire bytes.

Prereqs: `pie serve` running with a model, daemon launched on --port
(see run_pie_qwen.sh, or launch_daemon.py). Usage:

    python3 test_acceptance.py [--base http://127.0.0.1:8123]

Exit code 0 iff all checks pass.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))


def post(base: str, body: dict | str | bytes, path: str = "/v1/chat/completions"):
    data = body if isinstance(body, bytes) else (
        body.encode() if isinstance(body, str) else json.dumps(body).encode()
    )
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def sse_events(raw: bytes) -> list:
    """Parse SSE frames; returns list of parsed-JSON chunks, '[DONE]' kept as str."""
    events = []
    for frame in raw.decode("utf-8", "replace").split("\n\n"):
        frame = frame.strip()
        if not frame.startswith("data: "):
            continue
        payload = frame[len("data: "):]
        events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


def stream_request(base: str, extra: dict | None = None, messages=None) -> list:
    # Shaped like a real qwen-code request: thinking disabled (it injects
    # enable_thinking:false for ^qwen models on non-DashScope endpoints).
    body = {
        "model": "test",
        "messages": messages
        or [{"role": "user", "content": "Reply with the single word: hello"}],
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(extra or {})
    status, headers, raw = post(base, body)
    check("streaming 200", status == 200, f"got {status}")
    ctype = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
    check("SSE Content-Type is text/event-stream", "text/event-stream" in ctype, ctype)
    return sse_events(raw)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Absolute path"}},
            "required": ["file_path"],
        },
    },
}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # ── liveness ──
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            check("GET /health", r.status == 200)
    except Exception as e:  # noqa: BLE001
        check("GET /health", False, str(e))
        print("daemon unreachable — aborting")
        sys.exit(1)

    # ── text turn over SSE ──
    events = stream_request(base)
    chunks = [e for e in events if isinstance(e, dict)]
    check("[DONE] terminator present", events and events[-1] == "[DONE]")
    check("first chunk announces assistant role",
          bool(chunks) and chunks[0]["choices"][0]["delta"].get("role") == "assistant")
    content = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks if c.get("choices")
    )
    check("non-empty content for text turn", len(content.strip()) > 0)
    finishes = [c["choices"][0].get("finish_reason")
                for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    check("finish_reason present on final content chunk", bool(finishes), str(finishes))
    check("finish_reason is never error_finish", all(f != "error_finish" for f in finishes))
    usage_chunks = [c for c in chunks if not c.get("choices") and c.get("usage")]
    check("include_usage → usage chunk with empty choices", bool(usage_chunks))
    if usage_chunks:
        u = usage_chunks[-1]["usage"]
        check("usage has prompt/completion/total",
              all(k in u for k in ("prompt_tokens", "completion_tokens", "total_tokens")))
        check("usage reports cached_tokens detail",
              "cached_tokens" in u.get("prompt_tokens_details", {}))

    # ── tool-call turn + id uniqueness across sequential requests ──
    tool_msgs = [
        {"role": "system", "content": "You are a coding agent. Use the read_file tool."},
        {"role": "user", "content": "Read the file /tmp/example.txt using the read_file tool."},
    ]
    ids = []
    for i in range(2):
        events = stream_request(base, extra={"tools": TOOLS, "max_tokens": 512}, messages=tool_msgs)
        chunks = [e for e in events if isinstance(e, dict)]
        calls = [tc for c in chunks if c.get("choices")
                 for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
        check(f"request {i}: model produced a tool call", bool(calls))
        for tc in calls:
            check(f"request {i}: tool call has id", bool(tc.get("id")))
            check(f"request {i}: tool call has name+arguments",
                  bool(tc.get("function", {}).get("name")) and "arguments" in tc.get("function", {}))
            ids.append(tc.get("id"))
        finishes = [c["choices"][0].get("finish_reason")
                    for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
        if calls:
            check(f"request {i}: finish_reason tool_calls", "tool_calls" in finishes, str(finishes))
    check("tool-call ids unique across requests (fresh-instance counter bug)",
          len(ids) == len(set(ids)), str(ids))

    # ── error semantics ──
    status, _, raw = post(base, b"{not json")
    check("malformed JSON → 400 (not 500)", status == 400, f"got {status}")
    try:
        err = json.loads(raw)
        check("error body is OpenAI-shaped", "error" in err and "message" in err["error"])
    except Exception:  # noqa: BLE001
        check("error body is OpenAI-shaped", False, raw[:100].decode("utf-8", "replace"))
    status, _, _ = post(base, {"messages": [], "stream": False})
    check("empty messages → 400", status == 400, f"got {status}")

    # ── unknown-field tolerance (qwen-code injects chat_template_kwargs etc.) ──
    status, _, raw = post(base, {
        "model": "x",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say ok."}]}],
        "max_tokens": 16,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "some_unknown_field": {"a": 1},
        "parallel_tool_calls": True,
    })
    check("unknown fields ignored (parts content, non-stream 200)", status == 200, f"got {status}")
    if status == 200:
        body = json.loads(raw)
        msg = body["choices"][0]["message"]
        check("non-stream: content non-empty", bool((msg.get("content") or "").strip()))
        check("non-stream: finish_reason present",
              body["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls"))

    # ── KV-session reuse across a simulated follow-up turn ──
    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Pick a color and say only its name."},
    ]
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}}
    status, _, raw = post(base, {"messages": convo, "max_tokens": 256, "stream": False, **kwargs})
    check("turn 1 (non-stream) 200", status == 200, f"got {status}")
    if status == 200:
        body = json.loads(raw)
        reply = body["choices"][0]["message"]
        convo = convo + [
            {"role": "assistant", "content": reply.get("content")},
            {"role": "user", "content": "Now say it again in uppercase."},
        ]
        status, _, raw = post(base, {"messages": convo, "max_tokens": 256, "stream": False, **kwargs})
        check("turn 2 (echo-back history) 200", status == 200, f"got {status}")
        if status == 200:
            cached = json.loads(raw)["usage"]["prompt_tokens_details"]["cached_tokens"]
            check("turn 2 resumed from KV snapshot (cached_tokens > 0)",
                  cached > 0, f"cached_tokens={cached}")

    # ── Concurrency ──────────────────────────────────────────────────
    # A sequential suite cannot see this class at all. The
    # liu/opencode-integration serving path degrades EVERY request at N>=2
    # on both a dense and a hybrid model — a rejected launch
    # (`pie_metal_launch failed with status -1`) is turned by the degrade
    # discipline into `finish_reason:"length"` with a one-token answer,
    # which is indistinguishable on the wire from a model that stopped.
    # Their 25/25 green never covered it, and neither did our 33.
    #
    # The assertion is on token COUNT, not status: every request here is
    # given a budget it should exhaust, so a turn that stops at "length"
    # after a handful of tokens is nonsense on its face — "length" means
    # the budget was hit. That shape also survives the `'…'` placeholder
    # check, because one *real* token is not the placeholder.
    import threading

    def _one(i, out, n_tokens=64):
        try:
            st, _, raw = post(base, {
                "messages": [{"role": "user",
                              "content": f"Count from {i} to thirty, one per line."}],
                "max_tokens": n_tokens, "stream": False, **kwargs})
            d = json.loads(raw)
            out[i] = (st, d["usage"]["completion_tokens"],
                      d["choices"][0]["finish_reason"])
        except Exception as e:  # noqa: BLE001 - reported, not raised
            out[i] = (None, None, repr(e)[:60])

    for n in (2, 4):
        out: dict = {}
        threads = [threading.Thread(target=_one, args=(i, out)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bad = [i for i in range(n)
               if out.get(i, (None,))[0] != 200
               or (out[i][2] == "length" and (out[i][1] or 0) < 5)]
        check(f"{n} concurrent requests all produce real turns",
              not bad,
              "degraded: " + ", ".join(
                  f"req{i}={out.get(i)}" for i in bad))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
