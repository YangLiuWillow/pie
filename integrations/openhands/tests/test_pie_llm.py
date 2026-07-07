"""Unit tests for PieLLM.

We avoid spinning up a real Pie server by mocking _call_pie at the instance
level — it captures the structured messages/tools/gen_params PieLLM would
otherwise ship to the inferlet.
"""

from __future__ import annotations

import pytest
from litellm.types.utils import ModelResponse

from pie_openhands import PieLLM


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _patch_call_pie(llm: PieLLM, response: dict):
    """Replace _call_pie with an async lambda that returns ``response``.

    Captures the messages/tools/gen params on the LLM instance so the test
    can assert on them.
    """
    captured = {}

    async def fake(self, messages, tools, gen_params):
        captured["messages"] = messages
        captured["tools"] = tools
        captured["gen_params"] = gen_params
        return response

    # bind as a method so `self` lines up
    import types
    llm._call_pie = types.MethodType(fake, llm)
    return captured


def _make_llm(**overrides):
    defaults = dict(
        model="qwen3-coder-32b",
        pie_uri="ws://127.0.0.1:8080",
        # Belts and braces — keep retry quick for tests
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=1.0,
    )
    defaults.update(overrides)
    return PieLLM(**defaults)


# ---------------------------------------------------------------
# Transport tests
# ---------------------------------------------------------------


def test_transport_call_forwards_structured_messages_and_tools():
    llm = _make_llm()
    captured = _patch_call_pie(
        llm,
        response={
            "text": "Hello back.",
            "tool_calls": [],
            "stop_reason": "stop",
            "prompt_tokens": 7,
            "tokens_generated": 3,
        },
    )

    resp = llm._transport_call(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        temperature=0.0,
        max_tokens=64,
        stop=["<eot>"],
    )

    # Pie call got the structured messages, unflattened role/content intact.
    assert captured["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    assert captured["tools"] == []
    assert captured["gen_params"]["max_tokens"] == 64
    assert captured["gen_params"]["temperature"] == 0.0
    assert captured["gen_params"]["stop"] == ["<eot>"]
    assert captured["gen_params"]["model"] == "qwen3-coder-32b"

    # Response is a litellm ModelResponse
    assert isinstance(resp, ModelResponse)
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "Hello back."
    assert resp.choices[0].message.tool_calls is None
    assert resp.usage.prompt_tokens == 7
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 10
    assert resp.model == "qwen3-coder-32b"


def test_transport_call_flattens_vision_style_content():
    """OpenHands sometimes emits structured content (vision); we flatten it
    to a plain string before it reaches the inferlet's JSON contract."""
    llm = _make_llm()
    captured = _patch_call_pie(llm, {"text": "ok"})

    llm._transport_call(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "hello world"}]},
        ],
    )

    assert captured["messages"] == [{"role": "user", "content": "hello world"}]


def test_transport_call_passes_tool_calls_through():
    llm = _make_llm()
    captured = _patch_call_pie(llm, {"text": "ok"})

    llm._transport_call(
        messages=[
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "cats"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "no results"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "search", "description": "Search the web", "parameters": {}},
            }
        ],
    )

    assert captured["messages"][1]["tool_calls"][0]["function"]["name"] == "search"
    assert captured["messages"][2]["tool_call_id"] == "call_0"
    assert captured["tools"][0]["function"]["name"] == "search"


def test_transport_call_builds_native_tool_calls_in_response():
    llm = _make_llm()
    _patch_call_pie(
        llm,
        {
            "text": "",
            "tool_calls": [{"id": "call_0", "name": "search", "arguments": '{"q": "cats"}'}],
            "stop_reason": "tool_calls",
            "prompt_tokens": 10,
            "tokens_generated": 5,
        },
    )

    resp = llm._transport_call(messages=[{"role": "user", "content": "search for cats"}])

    assert resp.choices[0].finish_reason == "tool_calls"
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_0"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == '{"q": "cats"}'


def test_transport_call_maps_length_stop_reason():
    llm = _make_llm()
    _patch_call_pie(llm, {"text": "trunc", "stop_reason": "length"})
    resp = llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert resp.choices[0].finish_reason == "length"


def test_transport_call_maps_eos_to_stop():
    llm = _make_llm()
    _patch_call_pie(llm, {"text": "done", "stop_reason": "eos"})
    resp = llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert resp.choices[0].finish_reason == "stop"


def test_streaming_is_unsupported():
    llm = _make_llm()
    with pytest.raises(NotImplementedError, match="streaming"):
        llm._transport_call(
            messages=[{"role": "user", "content": "x"}],
            enable_streaming=True,
            on_token=lambda c: None,
        )


def test_temperature_defaults_to_zero_when_not_provided():
    llm = _make_llm(temperature=None)
    captured = _patch_call_pie(llm, {"text": "ok"})
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert captured["gen_params"]["temperature"] == 0.0


def test_max_tokens_falls_back_to_max_output_tokens():
    llm = _make_llm(max_output_tokens=512)
    captured = _patch_call_pie(llm, {"text": "ok"})
    llm._transport_call(messages=[{"role": "user", "content": "x"}])
    assert captured["gen_params"]["max_tokens"] == 512


def test_native_tool_calling_defaults_to_true():
    """PieLLM now populates real tool_calls, so the base class's native
    tool-calling path (not its prompt-mocked one) should be used."""
    llm = _make_llm()
    assert llm.native_tool_calling is True


# ---------------------------------------------------------------
# Return-value parsing
# ---------------------------------------------------------------


def test_parse_return_value_handles_json_string():
    from pie_openhands.llm import _parse_return_value

    out = _parse_return_value('{"text": "hi", "tokens_generated": 2}', [])
    assert out == {"text": "hi", "tokens_generated": 2}


def test_parse_return_value_handles_plain_string():
    from pie_openhands.llm import _parse_return_value

    out = _parse_return_value("just text", [])
    assert out == {"text": "just text"}


def test_parse_return_value_handles_none_falls_back_to_stdout():
    from pie_openhands.llm import _parse_return_value

    out = _parse_return_value(None, ["chunk1 ", "chunk2"])
    assert out == {"text": "chunk1 chunk2"}


def test_parse_return_value_handles_dict():
    from pie_openhands.llm import _parse_return_value

    payload = {"text": "x", "stop_reason": "stop"}
    assert _parse_return_value(payload, []) == payload
