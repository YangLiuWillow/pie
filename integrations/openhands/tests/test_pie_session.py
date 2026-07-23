"""Unit tests for PieLLM's KV-session protocol (openhands-coder-session).

The inferlet is stateless between invocations; the host carries the previous
prompt render's (length, hash) and echoes it back so the inferlet can run its
token-level extension check. These tests pin down that echo protocol without
a Pie server, by mocking ``_call_pie``.
"""

from __future__ import annotations

import types

from pie_openhands import PieLLM


def _make_llm(**overrides):
    defaults = dict(
        model="qwen3-coder-32b",
        pie_uri="ws://127.0.0.1:8080",
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=1.0,
    )
    defaults.update(overrides)
    return PieLLM(**defaults)


def _patch_call_pie_scripted(llm: PieLLM, responses: list[dict]):
    """Replace _call_pie with a mock that pops from ``responses`` per call.

    Returns the list of captured (messages, tools, gen_params) per call.
    """
    calls: list[dict] = []
    remaining = list(responses)

    async def fake(self, messages, tools, gen_params):
        calls.append({
            "messages": messages,
            "tools": tools,
            "gen_params": gen_params,
        })
        return remaining.pop(0)

    llm._call_pie = types.MethodType(fake, llm)
    return calls


def _session_response(*, mode: str, length: int, hash_: str, prefill: int) -> dict:
    return {
        "text": "ok",
        "tool_calls": [],
        "stop_reason": "stop",
        "prompt_tokens": length + 3,
        "tokens_generated": 2,
        "session": {
            "id": "ignored-by-host",
            "mode": mode,
            "len": length,
            "hash": hash_,
            "prefill_tokens": prefill,
        },
    }


# ---------------------------------------------------------------
# Request-side protocol
# ---------------------------------------------------------------


def test_session_fields_absent_by_default():
    llm = _make_llm()
    calls = _patch_call_pie_scripted(llm, [{"text": "ok"}])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    gp = calls[0]["gen_params"]
    assert "session_id" not in gp
    assert "session_prev_len" not in gp
    assert "kv_verify" not in gp


def test_first_session_call_sends_only_the_id():
    """Self-keyed APC: `session_id` names the cache namespace and nothing else
    goes on the wire. The inferlet derives the lookup key from the token render
    itself, so there is no length/hash for the host to echo."""
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(
        llm, [_session_response(mode="rebuilt", length=100, hash_="abc123", prefill=100)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    gp = calls[0]["gen_params"]
    assert gp["session_id"]
    assert "session_prev_len" not in gp
    assert "session_prev_hash" not in gp
    assert "kv_verify" not in gp


def test_second_call_reuses_the_session_id_and_echoes_nothing():
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(llm, [
        _session_response(mode="rebuilt", length=100, hash_="abc123", prefill=100),
        _session_response(mode="extended", length=140, hash_="def456", prefill=40),
    ])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    llm._transport_call(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "z"},
    ])

    first, second = calls[0]["gen_params"], calls[1]["gen_params"]
    # The id is what ties the two calls to one cache namespace; the reuse
    # itself is found by the inferlet, not directed by the host.
    assert second["session_id"] == first["session_id"]
    assert "session_prev_len" not in second
    assert "session_prev_hash" not in second


def test_kv_verify_flag_is_forwarded():
    llm = _make_llm(pie_session=True, pie_kv_verify=True)
    calls = _patch_call_pie_scripted(
        llm, [_session_response(mode="fresh", length=10, hash_="ff", prefill=10)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert calls[0]["gen_params"]["kv_verify"] is True


def test_stateless_inferlet_response_leaves_session_state_unchanged():
    """Pointing pie_session at a non-session inferlet (no `session` block in
    the response) is degraded but correct: the host keeps sending the same id
    and records no session telemetry."""
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(llm, [{"text": "ok"}, {"text": "ok"}])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert calls[1]["gen_params"]["session_id"] == calls[0]["gen_params"]["session_id"]
    assert "session_prev_len" not in calls[1]["gen_params"]
    assert llm.pie_session_summary()["num_calls"] == 0


# ---------------------------------------------------------------
# Close / teardown
# ---------------------------------------------------------------


def test_close_session_sends_delete_and_resets_state():
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(llm, [
        _session_response(mode="fresh", length=100, hash_="abc123", prefill=100),
        {"text": "", "session": {"mode": "deleted"}},
    ])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    sid = calls[0]["gen_params"]["session_id"]

    llm.close_pie_session()

    delete_call = calls[1]["gen_params"]
    assert delete_call == {"session_id": sid, "session_action": "delete"}
    assert calls[1]["messages"] == []
    assert llm._pie_session_id is None
    assert llm._pie_session_len == 0
    assert llm._pie_session_hash is None


def test_close_session_is_noop_without_session():
    llm = _make_llm()  # pie_session=False
    calls = _patch_call_pie_scripted(llm, [])
    llm.close_pie_session()
    assert calls == []

    llm2 = _make_llm(pie_session=True)  # enabled but never used
    calls2 = _patch_call_pie_scripted(llm2, [])
    llm2.close_pie_session()
    assert calls2 == []


def test_close_session_swallows_transport_errors():
    llm = _make_llm(pie_session=True)
    _patch_call_pie_scripted(
        llm, [_session_response(mode="fresh", length=5, hash_="aa", prefill=5)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "x"}])

    async def boom(self, messages, tools, gen_params):
        raise RuntimeError("server down")

    llm._call_pie = types.MethodType(boom, llm)
    llm.close_pie_session()  # must not raise
    assert llm._pie_session_id is None


# ---------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------


def test_session_summary_aggregates_modes_and_token_counts():
    llm = _make_llm(pie_session=True)
    _patch_call_pie_scripted(llm, [
        _session_response(mode="fresh", length=100, hash_="a1", prefill=100),
        _session_response(mode="extended", length=150, hash_="a2", prefill=50),
        _session_response(mode="extended", length=180, hash_="a3", prefill=30),
        _session_response(mode="rebuilt", length=90, hash_="a4", prefill=90),
    ])
    for _ in range(4):
        llm._transport_call(messages=[{"role": "user", "content": "x"}])

    summary = llm.pie_session_summary()
    assert summary["num_calls"] == 4
    assert summary["prompt_tokens_rendered"] == 100 + 150 + 180 + 90
    assert summary["prompt_tokens_prefilled"] == 100 + 50 + 30 + 90
    assert summary["modes"] == {"fresh": 1, "extended": 2, "rebuilt": 1}


# ---------------------------------------------------------------
# Fork / delegation glue (model_copy)
# ---------------------------------------------------------------


def _make_parent_with_live_session(*, length=100, hash_="abc123", **overrides):
    """A PieLLM that has completed one session call (has a live snapshot)."""
    llm = _make_llm(pie_session=True, **overrides)
    _patch_call_pie_scripted(
        llm, [_session_response(mode="fresh", length=length, hash_=hash_, prefill=length)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "parent turn"}])
    return llm


def test_model_copy_gives_the_child_its_own_session_identity():
    """The one invariant `model_copy` has to hold: a delegated child must not
    inherit the parent's session id, or the two conversations would interleave
    inside one cache namespace. Nothing else needs passing down — the child's
    shared task prefix hashes to a name the parent already saved, so reuse is
    found by content, not handed over."""
    parent = _make_parent_with_live_session(length=100, hash_="abc123")
    parent_sid = parent._pie_session_id

    child = parent.model_copy()

    # Child does NOT inherit the parent's session identity.
    assert child._pie_session_id is None
    assert child._pie_session_len == 0
    assert child._pie_session_hash is None
    # Parent is untouched — its snapshot must survive the copy.
    assert parent._pie_session_id == parent_sid
    assert parent._pie_session_len == 100


def test_child_first_call_uses_a_fresh_id_and_sends_no_fork_fields():
    """Delegation no longer needs an explicit fork handshake. The child gets its
    own namespace, and reuse of the parent's prefix happens implicitly: the
    child's leading tokens equal a boundary the parent already saved, so the
    content-addressed lookup hits it. All `model_copy` has to do is clear the
    inherited session id."""
    parent = _make_parent_with_live_session(length=100, hash_="abc123", pie_kv_verify=True)
    parent_sid = parent._pie_session_id
    child = parent.model_copy()

    calls = _patch_call_pie_scripted(
        child, [_session_response(mode="extended", length=113, hash_="def456", prefill=13)]
    )
    child._transport_call(messages=[{"role": "user", "content": "child turn"}])

    gp = calls[0]["gen_params"]
    assert "session_fork_from" not in gp
    assert "session_fork_prev_len" not in gp
    assert "session_fork_prev_hash" not in gp
    # Child got its own fresh session id, distinct from the parent's.
    assert gp["session_id"] and gp["session_id"] != parent_sid
    assert gp["kv_verify"] is True  # kv_verify inherited across the copy


def test_child_keeps_one_id_across_its_own_turns():
    parent = _make_parent_with_live_session(length=100, hash_="abc123")
    child = parent.model_copy()
    calls = _patch_call_pie_scripted(child, [
        _session_response(mode="extended", length=113, hash_="def456", prefill=13),
        _session_response(mode="extended", length=150, hash_="ghi789", prefill=37),
    ])
    child._transport_call(messages=[{"role": "user", "content": "turn 1"}])
    child._transport_call(messages=[{"role": "user", "content": "turn 2"}])

    first, second = calls[0]["gen_params"], calls[1]["gen_params"]
    assert second["session_id"] == first["session_id"]
    assert "session_fork_from" not in second
    assert "session_prev_len" not in second


def test_double_copy_still_leaves_the_child_without_a_session():
    """The SDK copies twice (parent→child, then a stream-flip copy). Clearing an
    already-cleared id is a no-op, so the grandchild is in the same clean state
    as the child and neither can touch the parent's namespace."""
    parent = _make_parent_with_live_session(length=100, hash_="abc123")
    parent_sid = parent._pie_session_id

    child = parent.model_copy()
    grandchild = child.model_copy()  # simulates manager.py's stream-flip copy

    assert grandchild._pie_session_id is None
    assert grandchild._pie_session_len == 0
    assert grandchild._pie_session_hash is None
    assert parent._pie_session_id == parent_sid  # parent survives both copies


def test_model_copy_without_live_session_is_harmless():
    parent = _make_llm(pie_session=True)  # never called → no snapshot
    child = parent.model_copy()
    assert child._pie_session_id is None

    calls = _patch_call_pie_scripted(
        child, [_session_response(mode="rebuilt", length=50, hash_="zz", prefill=50)]
    )
    child._transport_call(messages=[{"role": "user", "content": "x"}])
    assert "session_fork_from" not in calls[0]["gen_params"]


def test_copy_does_not_alias_parent_telemetry_list():
    parent = _make_parent_with_live_session()
    child = parent.model_copy()
    assert child._pie_session_stats == []
    child._pie_session_stats.append({"mode": "forked"})
    # Parent's telemetry must be unaffected.
    assert len(parent._pie_session_stats) == 1
    assert parent._pie_session_stats[0]["mode"] == "fresh"
