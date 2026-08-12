#!/usr/bin/env python3
"""Live acceptance suite for the OpenClaw ↔ Pie OpenAI surface (oc-PA.3).

Imports the opencode suite (../opencode/test_acceptance.py) for its plumbing
(raw-HTTP client, SSE parsing, tool-call wire-shape assertions) and its
client-agnostic wire tests, then swaps the fixture-specific and
client-policy tests for OpenClaw's — hard requirements from
tests/inferlets/fixtures/openclaw/AUDIT.md (§2, §6, §8):

  - openclaw fixtures replay (req-003 full 34-tool surface, req-004 history
    replay with assistant content:null, req-005 lean surface);
  - max_completion_tokens honored without max_tokens (D-4);
  - content-part-array user messages accepted (D-2 — repo-HEAD OpenClaw
    sends parts; the npm client sends strings; both must work);
  - every finish_reason ∈ OpenClaw's mapped set {stop, length, tool_calls}
    (D-9: any other string fails the whole turn client-side);
  - keepalives are empty-delta DATA chunks sharing the stream's chunk id
    (D-1: SSE comments are dropped by OpenClaw's sanitizer and never reset
    its watchdogs).

Same CLI and environment as the opencode suite (PIE_BASE_URL, PIE_API_KEY,
PIE_TEST_TIMEOUT, PIE_TEST_MAX_TOKENS, PIE_FIXTURE_DIR — which defaults to
the openclaw wire dir here). Exit codes: 0 pass; 1 hard failures; 2 server
unreachable.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Must be set before the base module loads (it reads FIXTURE_DIR at import).
os.environ.setdefault(
    "PIE_FIXTURE_DIR",
    os.path.join(_REPO, "tests", "inferlets", "fixtures", "openclaw", "wire"),
)

_spec = importlib.util.spec_from_file_location(
    "opencode_acceptance", os.path.join(HERE, "..", "opencode", "test_acceptance.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def load_fixture(name):
    """OpenClaw captures carry max_completion_tokens (never max_tokens, D-4);
    clamp whichever is present so a degenerate decode can't run for minutes."""
    path = os.path.join(base.FIXTURE_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        body = copy.deepcopy(json.load(f)["body"])
    for key in ("max_tokens", "max_completion_tokens"):
        if body.get(key, 0) > base.FIXTURE_MAX_TOKENS:
            body[key] = base.FIXTURE_MAX_TOKENS
    return body


_CACHE = {}


def req003_stream():
    """The captured first tool request: full default surface (34 tools,
    ~24k-token prompt), replayed verbatim except the token clamp."""
    if "req003" not in _CACHE:
        _CACHE["req003"] = base.stream_chat(load_fixture("req-003"))
    return _CACHE["req003"]


# ── OpenClaw-specific tests ─────────────────────────────────────────────────


def test_tool_turn_fixture_req003():
    """Replay the captured full-surface first agent request (req-003:
    ~24k-token prompt, 34 tools with strict:false — S-2). Hard: 200, clean
    SSE, finish_reason, [DONE]; tool-call wire shape when calls appear."""
    sr = req003_stream()
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert "text/event-stream" in sr.resp.content_type, sr.resp.content_type
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    assert sr.finish_reasons(), "no finish_reason chunk"
    assert not sr.junk, f"non-SSE lines on the wire: {sr.junk[:5]}"
    advertised = {t["function"]["name"] for t in load_fixture("req-003")["tools"]}
    ids = base.assert_toolcall_wire_shape(sr, advertised, "req-003")
    base.soft(bool(ids), "0.6B model produced no tool call on the req-003 "
                         "prompt (capture's reference answer was a `read` call)")


def test_history_replay_req004():
    """Replay the captured tool-history follow-up (req-004: assistant with
    content:null + tool_calls — D-3 — then role:tool string result) -> 200
    and a completed turn."""
    sr = base.stream_chat(load_fixture("req-004"))
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    assert sr.finish_reasons(), "no finish_reason chunk"
    base.soft(bool(sr.content().strip()) or bool(sr.tool_call_deltas()),
              "history-replay turn produced neither content nor tool calls")


def test_lean_fixture_req005():
    """Replay the lean-mode capture (req-005: 9-tool-class surface reduced
    to 4 on the npm client — S-3) -> 200 and a completed turn."""
    sr = base.stream_chat(load_fixture("req-005"))
    assert sr.resp.status == 200, f"got {sr.resp.status}: {sr.resp.raw[:300]!r}"
    assert sr.frames and sr.frames[-1] == "[DONE]", "no [DONE] terminator"
    assert sr.finish_reasons(), "no finish_reason chunk"


def test_max_completion_tokens_only():
    """D-4: OpenClaw sends max_completion_tokens and never max_tokens; a
    1-token budget must be honored (finish_reason length) without any
    max_tokens key present."""
    r = base.http("POST", base.CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Write a long story about the sea."}],
        "max_completion_tokens": 1,
        "stream": False,
    })
    assert r.status == 200, f"got {r.status}: {r.raw[:300]!r}"
    body = r.json()
    base.record_nonstream(body)
    assert body["choices"][0].get("finish_reason") == "length", body["choices"][0]


def test_content_parts_user_message():
    """D-2/S-1: repo-HEAD OpenClaw sends user content as a parts array (the
    npm client sends plain strings) — both shapes must complete. System
    stays a plain string, exactly as OpenClaw serializes."""
    r = base.http("POST", base.CHAT, body={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Reply with the single"},
                {"type": "text", "text": "word: hello"},
            ]},
        ],
        "max_completion_tokens": 32,
        "stream": False,
    })
    assert r.status == 200, f"got {r.status}: {r.raw[:300]!r}"
    body = r.json()
    base.record_nonstream(body)
    assert (body["choices"][0]["message"].get("content") or "").strip(), body


def test_keepalives_are_empty_delta_chunks():
    """D-1: during the ~24k-token prefill the gateway must keep the stream
    alive with empty-delta DATA chunks (OpenClaw's sanitizer drops
    comment-only frames; its watchdogs reset only on parsed chunks). Hard:
    every keepalive-shaped chunk shares the stream's single chunk id (the
    gateway mirrors identity — opencode pins chunk-id consistency). Soft:
    if the wire went >20s without bytes, keepalives were not flowing."""
    sr = req003_stream()
    ids = {c.get("id") for c in sr.chunks if "id" in c}
    assert len(ids) == 1, f"chunk ids inconsistent (keepalive mirroring broken?): {ids}"
    keepalives = [
        c for c in sr.chunks
        if c.get("choices")
        and c["choices"][0].get("delta") == {}
        and c["choices"][0].get("finish_reason") is None
    ]
    max_gap = max(sr.resp.gaps) if sr.resp.gaps else 0.0
    ttfb = sr.resp.ttfb if sr.resp.ttfb is not None else 0.0
    base.soft(max_gap < 20.0,
              f"max inter-read gap {max_gap:.1f}s — keepalive chunks not "
              "flowing during silence (gateway interval is 15s)")
    print(f"       keepalive: ttfb={ttfb:.1f}s max_gap={max_gap:.1f}s "
          f"empty_delta_chunks={len(keepalives)} comments={len(sr.comments)}")


def test_finish_reasons_in_mapped_set():
    """D-9 sweep: every finish_reason observed this run is in OpenClaw's
    mapped set — any other string fails the whole turn client-side
    (`Provider finish_reason: <x>` -> transport throw)."""
    assert base.RECORD["finish_reasons"], "no finish_reasons recorded"
    allowed = {"stop", "length", "tool_calls"}
    bad = [f for f in base.RECORD["finish_reasons"] if f not in allowed]
    assert not bad, f"finish_reason(s) outside OpenClaw's mapped set: {bad}"


# ── test list: base generic tests + OpenClaw-specific ───────────────────────

_OPENCODE_SPECIFIC = {
    # Replaced by the openclaw fixture/policy tests above.
    "test_tool_turn_fixture_req004",
    "test_history_replay_req005",
    "test_keepalive_stream_completes",
}

TESTS = [t for t in base.TESTS if t.__name__ not in _OPENCODE_SPECIFIC] + [
    test_tool_turn_fixture_req003,
    test_history_replay_req004,
    test_lean_fixture_req005,
    test_max_completion_tokens_only,
    test_content_parts_user_message,
    test_keepalives_are_empty_delta_chunks,
    test_finish_reasons_in_mapped_set,
]


if __name__ == "__main__":
    # The base runner (argparse, preflight, WARN report) drives our list.
    base.TESTS = TESTS
    sys.exit(base.main())
