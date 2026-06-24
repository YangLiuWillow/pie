"""End-to-end (OpenHands side) smoke test for PieLLM.

What "end-to-end" means here: we exercise the full openhands-sdk pipeline —
format_messages_for_llm, retry decoration, _transport_call dispatch, and
LLMResponse construction — using a real Agent + Conversation. The only thing
replaced is the wire to Pie: ``PieLLM._call_pie`` is monkey-patched to return
a canned dict (exactly the shape our Rust inferlet emits).

This proves Phase 1 of the integration plugs in correctly **without** needing
a running Pie server. The Pie-side wire-protocol smoke test (booting pie-serve
+ dummy driver + Python client) is a separate concern.
"""

from __future__ import annotations

import os
import tempfile
import types

import pytest

# Suppress the SDK's print-on-import banner so test output stays readable.
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from openhands.sdk import Agent, Conversation, Message, TextContent  # noqa: E402
from openhands.sdk.event import MessageEvent  # noqa: E402

from pie_openhands import PieLLM  # noqa: E402


# ─── Helpers ──────────────────────────────────────────────────────────────


def _mock_pie_response(text: str, stop_reason: str = "stop"):
    """Build the dict shape that the openhands-completion inferlet returns."""
    return {
        "text": text,
        "stop_reason": stop_reason,
        "prompt_tokens": 42,
        "tokens_generated": max(1, len(text) // 4),
    }


def _patch_call_pie(llm: PieLLM, responses: list[dict]):
    """Replace _call_pie with an iterator over canned dicts.

    Each PieLLM.completion() call (and therefore each Agent.step()) consumes
    one entry. If exhausted, raises — that's a much clearer failure than the
    test timing out.
    """
    queue = list(responses)
    captured = {"calls": []}

    async def fake(self, prompt, gen_params):
        captured["calls"].append({"prompt": prompt, "gen_params": gen_params})
        if not queue:
            raise AssertionError(
                f"_call_pie called more times than scripted "
                f"({len(captured['calls'])} calls, 0 responses left)"
            )
        return queue.pop(0)

    llm._call_pie = types.MethodType(fake, llm)
    return captured


def _make_pie_llm(**overrides) -> PieLLM:
    """Build a PieLLM configured for the smoke test.

    Uses raw_concat rendering so transformers does NOT get invoked
    (no real tokenizer download), and disables retries so a bad mock fails
    loudly instead of being retried 5x.
    """
    defaults = dict(
        model="qwen3-coder-32b",
        pie_render_strategy="raw_concat",
        usage_id="agent",
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=1.0,
        # No tools and no system_prompt_suffix in this test, so a small
        # context window is fine. Bypass the >=16k minimum check.
    )
    defaults.update(overrides)
    os.environ.setdefault("ALLOW_SHORT_CONTEXT_WINDOWS", "1")
    return PieLLM(**defaults)


# ─── Test A: direct LLM.completion() ──────────────────────────────────────


def test_direct_completion_runs_full_sdk_pipeline():
    """PieLLM.completion(...) goes through format_messages_for_llm + retry +
    _transport_call + LLMResponse construction. Verify each piece works."""
    llm = _make_pie_llm()
    captured = _patch_call_pie(llm, [_mock_pie_response("Hi there!")])

    response = llm.completion(
        messages=[Message(role="user", content=[TextContent(text="Hello")])],
    )

    # Parent class converted our dict into an LLMResponse
    from openhands.sdk.llm.llm_response import LLMResponse
    assert isinstance(response, LLMResponse)
    # And the Message round-tripped through format / unformat
    assert response.message.role == "assistant"
    # OpenHands stores assistant text in content as a list of TextContent
    text = "".join(
        p.text for p in response.message.content if hasattr(p, "text")
    )
    assert "Hi there!" in text

    # Our mock got the rendered prompt
    assert len(captured["calls"]) == 1
    rendered = captured["calls"][0]["prompt"]
    assert "Hello" in rendered
    assert rendered.endswith("assistant:")


def test_direct_completion_passes_temperature_and_max_tokens():
    llm = _make_pie_llm(temperature=0.0, max_output_tokens=128)
    captured = _patch_call_pie(llm, [_mock_pie_response("ok")])

    llm.completion(
        messages=[Message(role="user", content=[TextContent(text="x")])],
    )

    gen = captured["calls"][0]["gen_params"]
    assert gen["temperature"] == 0.0
    # max_tokens should fall through from max_output_tokens
    assert gen["max_tokens"] == 128


# ─── Test B: real Agent + Conversation ────────────────────────────────────


def test_agent_step_through_pie_llm(tmp_path):
    """Build a real Agent + Conversation around PieLLM, send one message,
    run, and verify a MessageEvent comes out the other side.

    This is the load-bearing smoke test: it exercises Agent.step,
    prepare_llm_messages, condenser pass-through, and the on_event callback.
    """
    llm = _make_pie_llm()
    # Plain text response with no tool calls → the agent will append a
    # MessageEvent and run will return (nothing more to execute).
    _patch_call_pie(
        llm,
        [_mock_pie_response("I'll just say hi back.", stop_reason="stop")],
    )

    agent = Agent(llm=llm, tools=[])

    events: list = []

    def collect(event):
        events.append(event)

    conv = Conversation(
        agent=agent,
        workspace=str(tmp_path),
        callbacks=[collect],
        max_iteration_per_run=1,
        visualizer=None,
    )

    conv.send_message("Hello")
    conv.run()

    # At minimum: one event with the assistant message we scripted
    msg_events = [e for e in events if isinstance(e, MessageEvent)]
    assert msg_events, f"expected MessageEvent in {[type(e).__name__ for e in events]}"
    # Find the assistant message we generated
    assistants = [
        m for m in msg_events
        if getattr(m, "source", None) == "agent"
        or getattr(getattr(m, "llm_message", None), "role", None) == "assistant"
    ]
    assert assistants, "no assistant MessageEvent produced"


# ─── Test C: failure modes ────────────────────────────────────────────────


def test_streaming_request_is_rejected():
    """OpenHands may set stream=True; Phase 1 explicitly refuses."""
    llm = _make_pie_llm(stream=True)
    with pytest.raises(NotImplementedError, match="streaming"):
        llm.completion(
            messages=[Message(role="user", content=[TextContent(text="x")])],
            on_token=lambda c: None,
        )
