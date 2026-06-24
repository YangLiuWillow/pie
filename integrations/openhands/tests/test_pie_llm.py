"""Unit tests for PieLLM.

We avoid spinning up a real Pie server or downloading a real tokenizer by
(a) mocking _call_pie at the instance level, and (b) using the raw_concat
render strategy so transformers is not imported.
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

    Captures the rendered prompt and gen params on the LLM instance so the
    test can assert on them.
    """
    captured = {}

    async def fake(self, prompt, gen_params):
        captured["prompt"] = prompt
        captured["gen_params"] = gen_params
        return response

    # bind as a method so `self` lines up
    import types
    llm._call_pie = types.MethodType(fake, llm)
    return captured


def _make_llm(**overrides):
    defaults = dict(
        model="qwen3-coder-32b",
        pie_render_strategy="raw_concat",
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
# Rendering tests
# ---------------------------------------------------------------


def test_raw_concat_renders_simple_conversation():
    llm = _make_llm()
    prompt = llm._render_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        tools=None,
    )
    assert "system: You are helpful." in prompt
    assert "user: Hi" in prompt
    assert prompt.endswith("assistant:")


def test_raw_concat_handles_list_content():
    """OpenHands sometimes emits structured content (vision); we flatten."""
    llm = _make_llm()
    prompt = llm._render_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}],
        tools=None,
    )
    assert "hello world" in prompt


def test_unknown_render_strategy_raises():
    llm = _make_llm(pie_render_strategy="from_scratch")
    with pytest.raises(ValueError, match="Unknown pie_render_strategy"):
        llm._render_prompt([{"role": "user", "content": "x"}], tools=None)


# ---------------------------------------------------------------
# Transport tests
# ---------------------------------------------------------------


def test_transport_call_translates_pie_response_into_model_response():
    llm = _make_llm()
    captured = _patch_call_pie(
        llm,
        response={
            "text": "Hello back.",
            "stop_reason": "stop",
            "prompt_tokens": 7,
            "tokens_generated": 3,
        },
    )

    resp = llm._transport_call(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.0,
        max_tokens=64,
        stop=["<eot>"],
    )

    # Pie call was made
    assert "user: Hi" in captured["prompt"]
    assert captured["gen_params"]["max_tokens"] == 64
    assert captured["gen_params"]["temperature"] == 0.0
    assert captured["gen_params"]["stop"] == ["<eot>"]
    assert captured["gen_params"]["model"] == "qwen3-coder-32b"

    # Response is a litellm ModelResponse
    assert isinstance(resp, ModelResponse)
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "Hello back."
    assert resp.usage.prompt_tokens == 7
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 10
    assert resp.model == "qwen3-coder-32b"


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


def test_streaming_is_unsupported_in_phase_1():
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
