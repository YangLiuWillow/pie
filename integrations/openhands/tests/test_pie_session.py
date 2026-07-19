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


def test_first_session_call_sends_id_and_zero_prev_len():
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(
        llm, [_session_response(mode="fresh", length=100, hash_="abc123", prefill=100)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    gp = calls[0]["gen_params"]
    assert gp["session_id"]
    assert gp["session_prev_len"] == 0
    assert "session_prev_hash" not in gp
    assert "kv_verify" not in gp


def test_second_call_echoes_len_and_hash_with_stable_id():
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(llm, [
        _session_response(mode="fresh", length=100, hash_="abc123", prefill=100),
        _session_response(mode="extended", length=140, hash_="def456", prefill=40),
    ])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    llm._transport_call(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "z"},
    ])

    first, second = calls[0]["gen_params"], calls[1]["gen_params"]
    assert second["session_id"] == first["session_id"]
    assert second["session_prev_len"] == 100
    assert second["session_prev_hash"] == "abc123"


def test_kv_verify_flag_is_forwarded():
    llm = _make_llm(pie_session=True, pie_kv_verify=True)
    calls = _patch_call_pie_scripted(
        llm, [_session_response(mode="fresh", length=10, hash_="ff", prefill=10)]
    )
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert calls[0]["gen_params"]["kv_verify"] is True


def test_stateless_inferlet_response_leaves_session_state_unchanged():
    """Pointing pie_session at a non-session inferlet (no `session` block in
    the response) must keep prev_len at 0 — every call then renders fresh,
    which is degraded but correct."""
    llm = _make_llm(pie_session=True)
    calls = _patch_call_pie_scripted(llm, [{"text": "ok"}, {"text": "ok"}])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert calls[1]["gen_params"]["session_prev_len"] == 0
    assert "session_prev_hash" not in calls[1]["gen_params"]
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
