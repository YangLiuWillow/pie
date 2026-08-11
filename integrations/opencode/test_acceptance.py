#!/usr/bin/env python3
"""Live acceptance suite for the opencode ↔ Pie OpenAI surface (PA.3).

Asserts every hard requirement from the opencode wire audit
(tests/inferlets/fixtures/opencode/AUDIT.md) plus the gateway ingress
contract (gateway/src/ingress/openai.rs module docs) against a RUNNING
`pie serve`, using raw HTTP over the stdlib only (http.client/urllib —
no pip deps), so assertions are on actual wire bytes.

Usage:
    python3 test_acceptance.py                 # run everything
    python3 test_acceptance.py --collect-only  # list tests, no server needed
    python3 test_acceptance.py --only stream   # substring filter

Environment:
    PIE_BASE_URL        server base (default http://127.0.0.1:8080)
    PIE_API_KEY         Bearer token to send (default pie-local; value is
                        not verified by the gateway — key checking is an
                        edge concern)
    PIE_TEST_TIMEOUT    per-request timeout seconds (default 600 — Metal
                        prefill of the 7.5k-token opencode prompt is slow)
    PIE_TEST_MAX_TOKENS clamp applied to replayed fixture bodies whose
                        captured max_tokens is 32000 (default 1024)
    PIE_FIXTURE_DIR     override tests/inferlets/fixtures/opencode/wire

Hard vs soft: wire-SHAPE requirements are hard assertions (fail the run).
Model-BEHAVIOR expectations on Qwen3-0.6B (e.g. "the model chooses to call
a tool") are soft — reported as WARN, never a failure. Whenever tool calls
DO appear, their wire shape is asserted hard.

Exit codes: 0 all hard checks pass; 1 hard failures; 2 server unreachable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from http.client import HTTPConnection
from urllib.parse import urlparse

# ── configuration ────────────────────────────────────────────────────────────

BASE = os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("PIE_API_KEY", "pie-local")
TIMEOUT = float(os.environ.get("PIE_TEST_TIMEOUT", "600"))
FIXTURE_MAX_TOKENS = int(os.environ.get("PIE_TEST_MAX_TOKENS", "1024"))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURE_DIR = os.environ.get(
    "PIE_FIXTURE_DIR",
    os.path.join(_REPO, "tests", "inferlets", "fixtures", "opencode", "wire"),
)

CHAT = "/v1/chat/completions"

# ── plumbing ────────────────────────────────────────────────────────────────


class Resp:
    """One raw HTTP exchange: status, lowercased headers, body bytes, timing."""

    def __init__(self, status, headers, raw, gaps, ttfb):
        self.status = status
        self.headers = headers
        self.raw = raw
        self.gaps = gaps  # inter-read gaps (seconds), soft-logged only
        self.ttfb = ttfb  # time to first body byte

    def json(self):
        return json.loads(self.raw)

    @property
    def content_type(self):
        return self.headers.get("content-type", "")


def http(method, path, body=None, headers=None, auth=True, timeout=None):
    """Raw request. `body` may be dict (JSON-encoded), bytes, or None."""
    url = urlparse(BASE)
    conn = HTTPConnection(url.hostname, url.port or 80, timeout=timeout or TIMEOUT)
    hdrs = {"Content-Type": "application/json", "Accept": "*/*"}
    if auth:
        hdrs["Authorization"] = f"Bearer {API_KEY}"
    if headers:
        hdrs.update(headers)
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        conn.request(method, path, body=data, headers=hdrs)
        t0 = time.monotonic()
        resp = conn.getresponse()
        raw = b""
        gaps = []
        ttfb = None
        last = time.monotonic()
        while True:
            chunk = resp.read(65536)
            now = time.monotonic()
            if not chunk:
                break
            if ttfb is None:
                ttfb = now - t0
            gaps.append(now - last)
            last = now
            raw += chunk
        return Resp(resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw, gaps, ttfb)
    finally:
        conn.close()


def parse_sse(raw):
    """Split an SSE body into (frames, comments, junk_lines).

    frames: parsed-JSON chunks in order; the '[DONE]' sentinel kept as str.
    comments: ':'-prefixed keepalive lines (invisible to the JSON layer).
    junk: any non-empty line that is neither a data line nor a comment —
    stdout/debug leakage detection (the envelope contract says process
    stdout/stderr never reach the wire).
    """
    frames, comments, junk = [], [], []
    for line in raw.decode("utf-8", "replace").split("\n"):
        line = line.rstrip("\r")
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                frames.append("[DONE]")
            else:
                try:
                    frames.append(json.loads(payload))
                except json.JSONDecodeError:
                    junk.append(line)
        elif line.startswith(":"):
            comments.append(line)
        elif line.strip():
            junk.append(line)
    return frames, comments, junk


class StreamResult:
    def __init__(self, resp):
        self.resp = resp
        self.frames, self.comments, self.junk = parse_sse(resp.raw)

    @property
    def chunks(self):
        return [f for f in self.frames if isinstance(f, dict)]

    def deltas(self):
        return [
            c["choices"][0].get("delta", {})
            for c in self.chunks
            if c.get("choices")
        ]

    def content(self):
        return "".join(d.get("content") or "" for d in self.deltas())

    def finish_reasons(self):
        return [
            c["choices"][0].get("finish_reason")
            for c in self.chunks
            if c.get("choices") and c["choices"][0].get("finish_reason")
        ]

    def tool_call_deltas(self):
        """Flat list of tool_call delta entries, stream order."""
        out = []
        for d in self.deltas():
            out.extend(d.get("tool_calls") or [])
        return out

    def usage_chunks(self):
        return [c for c in self.chunks if c.get("usage") and not c.get("choices")]


# every finish_reason observed anywhere this run (for the never-error_finish
# sweep), plus every stream result (for the leakage sweeps)
RECORD = {"finish_reasons": [], "streams": []}


def record_nonstream(body):
    for ch in body.get("choices", []):
        if ch.get("finish_reason"):
            RECORD["finish_reasons"].append(ch["finish_reason"])


def stream_chat(body, timeout=None):
    resp = http("POST", CHAT, body=body, timeout=timeout)
    sr = StreamResult(resp)
    RECORD["finish_reasons"].extend(sr.finish_reasons())
    RECORD["streams"].append(sr)
    return sr


_CACHE = {}


def basic_stream():
    """One shared plain-text streaming turn, shaped like real opencode traffic
    (always max_tokens + stream_options.include_usage, no temperature)."""
    if "basic" not in _CACHE:
        _CACHE["basic"] = stream_chat({
            "model": "qwen3-0.6b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the single word: hello"},
            ],
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
    return _CACHE["basic"]


def req004_stream():
    """The captured first agent request (10 real opencode tools, ~7.5k-token
    prompt), replayed verbatim except max_tokens clamped for runtime."""
    if "req004" not in _CACHE:
        _CACHE["req004"] = stream_chat(load_fixture("req-004"))
    return _CACHE["req004"]


def load_fixture(name):
    path = os.path.join(FIXTURE_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        body = copy.deepcopy(json.load(f)["body"])
    # Captured bodies carry max_tokens: 32000 (opencode's OUTPUT_TOKEN_MAX
    # clamp); bound the live run so a degenerate decode can't run for minutes.
    if body.get("max_tokens", 0) > FIXTURE_MAX_TOKENS:
        body["max_tokens"] = FIXTURE_MAX_TOKENS
    return body


SYNTH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": (
            "Get the current time in a timezone. ALWAYS call this tool when "
            "the user asks anything about the current time; you cannot know "
            "the time yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. Europe/Paris",
                }
            },
            "required": ["timezone"],
        },
    },
}


def synthetic_tool_body():
    return {
        "model": "qwen3-0.6b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a function-calling assistant. You MUST answer by "
                    "calling one of the provided tools. Never answer from memory."
                ),
            },
            {
                "role": "user",
                "content": "What time is it right now in Paris? Use the get_time tool.",
            },
        ],
        "tools": [SYNTH_TOOL],
        "tool_choice": "auto",
        "max_tokens": 256,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


# ── assertion helpers ───────────────────────────────────────────────────────

WARNINGS = []  # (test_name, message) — soft findings, never failures
_CURRENT_TEST = [""]


def soft(ok, msg):
    if not ok:
        WARNINGS.append((_CURRENT_TEST[0], msg))


def is_openai_error(resp):
    """Body is the OpenAI error envelope {"error":{message,type,...}}."""
    try:
        e = resp.json()
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(e, dict) and "error" in e and "message" in e["error"]


def assert_toolcall_wire_shape(sr, advertised, label):
    """Hard wire-shape assertions on the tool-call deltas of one stream —
    applied only when calls are present (model behavior is not asserted).

    AUDIT §3: the FIRST delta for a tool_call index must carry both `id` and
    `function.name` (the AI SDK throws InvalidResponseDataError otherwise);
    `arguments` must accumulate to valid JSON; ≥1 call ⇒ finish_reason
    tool_calls; ids unique within the turn.
    Returns the list of tool-call ids seen.
    """
    calls = sr.tool_call_deltas()
    if not calls:
        return []
    seen_first = {}
    args_by_index = {}
    ids = []
    for tc in calls:
        assert "index" in tc, f"{label}: tool_call delta missing index: {tc}"
        idx = tc["index"]
        if idx not in seen_first:
            assert tc.get("id"), f"{label}: first delta for index {idx} lacks id"
            assert tc.get("function", {}).get("name"), (
                f"{label}: first delta for index {idx} lacks function.name "
                "(AI SDK InvalidResponseDataError)"
            )
            seen_first[idx] = tc
            ids.append(tc["id"])
        args_by_index.setdefault(idx, []).append(
            tc.get("function", {}).get("arguments") or ""
        )
    for idx, parts in args_by_index.items():
        joined = "".join(parts)
        try:
            json.loads(joined)
        except json.JSONDecodeError:
            raise AssertionError(
                f"{label}: accumulated arguments for index {idx} are not "
                f"valid JSON: {joined[:200]!r}"
            )
    assert "tool_calls" in sr.finish_reasons(), (
        f"{label}: calls emitted but finish_reason tool_calls missing "
        f"(got {sr.finish_reasons()})"
    )
    assert len(ids) == len(set(ids)), f"{label}: duplicate tool-call ids {ids}"
    # Name byte-match is a model-quality property (a bad name degrades into
    # opencode's hidden `invalid` tool — a wasted round-trip, not a crash).
    names = [seen_first[i]["function"]["name"] for i in seen_first]
    soft(
        all(n in advertised for n in names),
        f"tool-call name(s) {names} not all in advertised set (opencode will "
        "bounce them to the `invalid` tool)",
    )
    return ids


# ── tests ───────────────────────────────────────────────────────────────────
# Order matters only for readability; each test is independently meaningful.


def test_health_endpoint():
    """GET /health answers 200 (no auth required)."""
    r = http("GET", "/health", auth=False)
    assert r.status == 200, f"/health -> {r.status}"


def test_models_endpoint():
    """GET /v1/models is a JSON model list with at least one id."""
    r = http("GET", "/v1/models", auth=False)
    assert r.status == 200, f"/v1/models -> {r.status}"
    body = r.json()
    assert body.get("object") == "list", body
    data = body.get("data")
    assert isinstance(data, list) and data, body
    assert all("id" in m for m in data), body


def test_auth_missing_bearer_401():
    """No Authorization and no x-pie-identity -> 401 with the OpenAI error
    shape, type authentication_error."""
    r = http("POST", CHAT, body={"model": "x", "messages": [
        {"role": "user", "content": "hi"}], "max_tokens": 8}, auth=False)
    assert r.status == 401, f"expected 401, got {r.status}: {r.raw[:200]!r}"
    assert is_openai_error(r), f"401 body not OpenAI-shaped: {r.raw[:200]!r}"
    assert r.json()["error"].get("type") == "authentication_error", r.json()


def test_auth_bearer_accepted():
    """Any Bearer token is accepted (key checking is an edge concern) and a
    minimal non-streaming request completes with 200."""
    r = http("POST", CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 16,
        "stream": False,
    })
    assert r.status == 200, f"expected 200, got {r.status}: {r.raw[:300]!r}"
    record_nonstream(r.json())


def test_invalid_json_400():
    """Malformed JSON -> 400 with an OpenAI error body — never 500
    (opencode retries 5xx forever)."""
    r = http("POST", CHAT, body=b"{not json")
    assert r.status == 400, f"expected 400, got {r.status}"
    assert r.status < 500, "5xx on malformed input (opencode retries forever)"
    assert is_openai_error(r), f"error body not OpenAI-shaped: {r.raw[:200]!r}"


def test_non_object_body_400():
    """A JSON body that is not an object -> 400, OpenAI error shape."""
    r = http("POST", CHAT, body=b"[1, 2, 3]")
    assert r.status == 400, f"expected 400, got {r.status}"
    assert is_openai_error(r), f"error body not OpenAI-shaped: {r.raw[:200]!r}"


def test_empty_messages_400():
    """messages: [] -> 400 from the inferlet's parse (never 500)."""
    r = http("POST", CHAT, body={"model": "x", "messages": [], "stream": False})
    assert r.status == 400, f"expected 400, got {r.status}: {r.raw[:300]!r}"
    assert is_openai_error(r), f"error body not OpenAI-shaped: {r.raw[:200]!r}"


def test_unknown_fields_tolerated():
    """AUDIT §1c/§6 must-tolerate list: $schema-laden tool parameters,
    maximum: 9007199254740991 (2^53-1), unknown top-level fields,
    tool_choice auto, opencode session headers -> 200."""
    schema_tool = {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Schema-noise probe.",
            "parameters": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "maximum": 9007199254740991},
                },
                "additionalProperties": False,
            },
        },
    }
    r = http("POST", CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 16,
        "stream": False,
        "tools": [schema_tool],
        "tool_choice": "auto",
        "stream_options": {"include_usage": True},
        "some_future_field": {"a": 1},
        "parallel_tool_calls": True,
    }, headers={
        "x-session-id": "ses_acceptance",
        "x-session-affinity": "ses_acceptance",
    })
    assert r.status == 200, f"expected 200, got {r.status}: {r.raw[:300]!r}"
    record_nonstream(r.json())


def test_stream_content_type():
    """stream:true responds Content-Type text/event-stream."""
    sr = basic_stream()
    assert sr.resp.status == 200, f"stream -> {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert "text/event-stream" in sr.resp.content_type, sr.resp.content_type


def test_stream_role_first_chunk():
    """First delta announces role assistant (before any content)."""
    sr = basic_stream()
    deltas = sr.deltas()
    assert deltas, "no delta chunks"
    assert deltas[0].get("role") == "assistant", deltas[0]


def test_stream_content_accumulates():
    """Content deltas accumulate to a non-empty assistant turn."""
    sr = basic_stream()
    assert sr.content().strip(), "accumulated stream content is empty"


def test_stream_finish_reason_stop_or_length():
    """Plain text turn finishes with stop or length (a finish_reason is
    always present — its absence maps to 'unknown' client-side)."""
    sr = basic_stream()
    finishes = sr.finish_reasons()
    assert finishes, "no finish_reason chunk in stream"
    assert finishes[-1] in ("stop", "length"), finishes


def test_stream_usage_chunk():
    """include_usage -> final usage chunk (empty choices) with
    prompt/completion/total tokens + prompt_tokens_details.cached_tokens,
    cached_tokens <= prompt_tokens (else the SDK's noCache goes negative)."""
    sr = basic_stream()
    usage_chunks = sr.usage_chunks()
    assert usage_chunks, "no usage-only chunk despite include_usage"
    u = usage_chunks[-1]["usage"]
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert isinstance(u.get(k), int), f"usage.{k} missing/not int: {u}"
    details = u.get("prompt_tokens_details")
    assert isinstance(details, dict) and "cached_tokens" in details, (
        f"prompt_tokens_details.cached_tokens missing: {u}"
    )
    assert details["cached_tokens"] <= u["prompt_tokens"], u


def test_stream_done_terminator():
    """The stream ends with data: [DONE]."""
    sr = basic_stream()
    assert sr.frames and sr.frames[-1] == "[DONE]", (
        f"last frame {sr.frames[-1] if sr.frames else None!r}"
    )


def test_stream_chunk_ids_consistent():
    """All chunks of one stream share a single completion id, and every
    chunk is object chat.completion.chunk."""
    sr = basic_stream()
    ids = {c.get("id") for c in sr.chunks if "id" in c}
    assert len(ids) == 1, f"chunk ids inconsistent within one stream: {ids}"
    objs = {c.get("object") for c in sr.chunks if "object" in c}
    assert objs <= {"chat.completion.chunk"}, objs


def test_stream_no_envelope_leakage():
    """The gateway⇄inferlet envelope's {"status":u16} header must never
    appear as an SSE payload, and no chunk carries a top-level status key."""
    for sr in RECORD["streams"] or [basic_stream()]:
        for c in sr.chunks:
            assert set(c.keys()) != {"status"}, f"envelope leaked: {c}"
            assert "status" not in c, f"'status' key leaked into chunk: {c}"


def test_stream_no_stdout_leakage():
    """Every non-empty SSE line is a data line or a ':' comment — process
    stdout/stderr instrumentation never reaches the wire."""
    sr = basic_stream()
    assert not sr.junk, f"non-SSE lines on the wire: {sr.junk[:5]}"


def test_non_streaming_body():
    """stream:false -> exactly one JSON body: non-empty content, a
    finish_reason, and usage."""
    r = http("POST", CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Name one color, one word."}],
        "max_tokens": 64,
        "stream": False,
    })
    assert r.status == 200, f"got {r.status}: {r.raw[:300]!r}"
    assert "application/json" in r.content_type, r.content_type
    body = r.json()  # exactly one JSON document, or this raises
    record_nonstream(body)
    choice = body["choices"][0]
    assert (choice["message"].get("content") or "").strip(), choice
    assert choice.get("finish_reason") in ("stop", "length", "tool_calls"), choice
    assert isinstance(body.get("usage"), dict), body.get("usage")


def test_tool_turn_fixture_req004():
    """Replay the captured first agent request (req-004: ~7.5k-token system
    prompt + 10 real opencode tools). Hard: 200, clean SSE, a finish_reason,
    [DONE]. Soft: the 0.6B model actually chooses to call a tool (the
    capture's answer was a `read` call). If calls appear, their wire shape
    is asserted hard."""
    sr = req004_stream()
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert "text/event-stream" in sr.resp.content_type, sr.resp.content_type
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    assert sr.finish_reasons(), "no finish_reason chunk"
    assert not sr.junk, f"non-SSE lines on the wire: {sr.junk[:5]}"
    advertised = {t["function"]["name"] for t in load_fixture("req-004")["tools"]}
    ids = assert_toolcall_wire_shape(sr, advertised, "req-004")
    soft(bool(ids), "0.6B model produced no tool call on the req-004 prompt "
                    "(capture's reference answer was a `read` call)")


def test_tool_call_forced_synthetic():
    """Single simple tool + a prompt that strongly demands using it.
    Soft: a call appears. Hard when it does: atomic first delta carries
    index+id+function.name, arguments accumulate to valid JSON,
    finish_reason tool_calls, unique ids."""
    sr = stream_chat(synthetic_tool_body())
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    ids = assert_toolcall_wire_shape(sr, {"get_time"}, "synthetic")
    soft(bool(ids), "model declined the strongly-forced get_time call "
                    "(0.6B behavior, not a wire fault)")
    _CACHE["synth1_ids"] = ids


def test_tool_ids_unique_across_processes():
    """Two sequential requests are two separate inferlet processes; their
    tool-call ids must never collide (ids are instance-id-derived —
    a fresh-counter scheme like call_0 would collide every time)."""
    first = _CACHE.get("synth1_ids")
    if first is None:
        sr1 = stream_chat(synthetic_tool_body())
        first = assert_toolcall_wire_shape(sr1, {"get_time"}, "uniq-run-1")
    sr2 = stream_chat(synthetic_tool_body())
    second = assert_toolcall_wire_shape(sr2, {"get_time"}, "uniq-run-2")
    if first and second:
        overlap = set(first) & set(second)
        assert not overlap, (
            f"tool-call ids collided across processes: {overlap} "
            f"(run1={first}, run2={second})"
        )
    else:
        soft(False, "one or both runs produced no calls; cross-process id "
                    "uniqueness not exercised this run")


def test_history_replay_req005():
    """Replay the captured tool-history follow-up (req-005: system, user,
    assistant with content:\"\" + tool_calls, role:tool result) -> 200 and a
    completed turn."""
    sr = stream_chat(load_fixture("req-005"))
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    assert sr.finish_reasons(), "no finish_reason chunk"
    soft(bool(sr.content().strip()) or bool(sr.tool_call_deltas()),
         "history-replay turn produced neither content nor tool calls")


def test_max_tokens_one_finish_length():
    """max_tokens: 1 -> finish_reason length."""
    r = http("POST", CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Write a long story about the sea."}],
        "max_tokens": 1,
        "stream": False,
    })
    assert r.status == 200, f"got {r.status}: {r.raw[:300]!r}"
    body = r.json()
    record_nonstream(body)
    assert body["choices"][0].get("finish_reason") == "length", body["choices"][0]


def test_keepalive_stream_completes():
    """The long-prefill req-004 stream completes end-to-end even when the
    first chunk takes the whole prefill window (the gateway's ': ping'
    comments keep any client chunkTimeout fed — AUDIT §2: opencode's
    watchdog resets on raw bytes, comments included). Timing is soft-logged,
    not asserted."""
    sr = req004_stream()
    assert sr.frames and sr.frames[-1] == "[DONE]", "long-prefill stream did not complete"
    max_gap = max(sr.resp.gaps) if sr.resp.gaps else 0.0
    ttfb = sr.resp.ttfb if sr.resp.ttfb is not None else 0.0
    soft(ttfb < 30.0,
         f"time-to-first-byte {ttfb:.1f}s (slow prefill; fine if keepalives "
         "were flowing)")
    soft(max_gap < 30.0, f"max inter-read gap {max_gap:.1f}s with no bytes "
                         "in between — check the keepalive interval")
    print(f"       keepalive: ttfb={ttfb:.1f}s max_gap={max_gap:.1f}s "
          f"comments={len(sr.comments)}")


def test_never_error_finish():
    """Sweep every finish_reason and SSE event observed this run: none is
    error_finish, and no in-stream error events were emitted."""
    assert RECORD["finish_reasons"], "no finish_reasons recorded (earlier tests all failed?)"
    bad = [f for f in RECORD["finish_reasons"] if f == "error_finish"]
    assert not bad, f"error_finish observed: {len(bad)} time(s)"
    for sr in RECORD["streams"]:
        for c in sr.chunks:
            assert "error" not in c, f"in-stream error event: {c}"


# ── runner ──────────────────────────────────────────────────────────────────

TESTS = [
    test_health_endpoint,
    test_models_endpoint,
    test_auth_missing_bearer_401,
    test_auth_bearer_accepted,
    test_invalid_json_400,
    test_non_object_body_400,
    test_empty_messages_400,
    test_unknown_fields_tolerated,
    test_stream_content_type,
    test_stream_role_first_chunk,
    test_stream_content_accumulates,
    test_stream_finish_reason_stop_or_length,
    test_stream_usage_chunk,
    test_stream_done_terminator,
    test_stream_chunk_ids_consistent,
    test_stream_no_envelope_leakage,
    test_stream_no_stdout_leakage,
    test_non_streaming_body,
    test_tool_turn_fixture_req004,
    test_tool_call_forced_synthetic,
    test_tool_ids_unique_across_processes,
    test_history_replay_req005,
    test_max_tokens_one_finish_length,
    test_keepalive_stream_completes,
    test_never_error_finish,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collect-only", action="store_true",
                    help="list test names and exit (no server needed)")
    ap.add_argument("--only", default="",
                    help="run only tests whose name contains this substring")
    args = ap.parse_args()

    selected = [t for t in TESTS if args.only in t.__name__]
    if args.collect_only:
        for t in selected:
            print(t.__name__)
        print(f"\n{len(selected)} tests")
        return 0

    # Preflight: is the server there at all?
    try:
        http("GET", "/health", auth=False, timeout=10)
    except OSError as e:
        print(f"server unreachable at {BASE}: {e}")
        print("boot it with integrations/opencode/run_pie_opencode.sh")
        return 2

    passed = failed = 0
    for t in selected:
        _CURRENT_TEST[0] = t.__name__
        t0 = time.monotonic()
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__} — {e}")
        except Exception as e:  # noqa: BLE001 — a crashed test is a failed test
            failed += 1
            print(f"[FAIL] {t.__name__} — {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"[PASS] {t.__name__} ({time.monotonic() - t0:.1f}s)")

    for name, msg in WARNINGS:
        print(f"[WARN] {name} — {msg}")
    print(f"\n{passed} passed, {failed} failed, {len(WARNINGS)} warnings "
          f"({len(selected)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
