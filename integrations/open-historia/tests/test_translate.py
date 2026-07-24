"""Unit tests for the pure OpenAI ⇄ inferlet translation layer.

No Pie server, no GPU — these exercise request/response mapping only.
"""

from __future__ import annotations

from pie_openhistoria import translate


# ── request → inferlet input ────────────────────────────────────────────────

def test_basic_request_maps_messages_and_defaults():
    body = {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "You are the game master."},
            {"role": "user", "content": "Advance to 1914."},
        ],
    }
    payload = translate.chat_request_to_inferlet_input(body)
    assert payload["messages"][0] == {"role": "system", "content": "You are the game master."}
    assert payload["max_tokens"] == translate.DEFAULT_MAX_TOKENS
    assert payload["temperature"] == translate.DEFAULT_TEMPERATURE
    assert payload["top_p"] == translate.DEFAULT_TOP_P
    assert payload["stop"] == []
    assert payload["model"] == "qwen3"
    assert "tools" not in payload


def test_max_completion_tokens_alias_and_stop_string():
    body = {"messages": [], "max_completion_tokens": 4096, "stop": "END"}
    payload = translate.chat_request_to_inferlet_input(body)
    assert payload["max_tokens"] == 4096
    assert payload["stop"] == ["END"]


def test_forced_tool_choice_forwards_tools():
    body = {
        "messages": [{"role": "user", "content": "jump"}],
        "tools": [{"type": "function", "function": {
            "name": "submit_jump_result",
            "description": "d",
            "parameters": {"type": "object"},
        }}],
        "tool_choice": "required",
    }
    payload = translate.chat_request_to_inferlet_input(body)
    assert translate.wants_tools(body) is True
    assert payload["tools"] == [{"function": {
        "name": "submit_jump_result", "description": "d", "parameters": {"type": "object"},
    }}]


def test_tool_choice_none_drops_tools():
    body = {
        "messages": [],
        "tools": [{"function": {"name": "x"}}],
        "tool_choice": "none",
    }
    assert translate.wants_tools(body) is False
    assert "tools" not in translate.chat_request_to_inferlet_input(body)


def test_array_content_is_flattened():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
    ]}]}
    payload = translate.chat_request_to_inferlet_input(body)
    assert payload["messages"][0]["content"] == "ab"


def test_assistant_tool_call_history_preserved():
    body = {"messages": [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "submit_actions", "arguments": '{"topics":[]}',
        }}],
    }]}
    payload = translate.chat_request_to_inferlet_input(body)
    msg = payload["messages"][0]
    assert "content" not in msg  # null content dropped
    assert msg["tool_calls"] == [{"function": {
        "name": "submit_actions", "arguments": '{"topics":[]}',
    }}]


# ── inferlet output → chat completion ───────────────────────────────────────

def test_plain_text_output():
    out = {"text": "Hello envoy.", "stop_reason": "stop",
           "prompt_tokens": 10, "tokens_generated": 3}
    resp = translate.inferlet_output_to_chat_completion(
        out, model="pie", request_id="chatcmpl-1", created=123)
    choice = resp["choices"][0]
    assert choice["message"]["content"] == "Hello envoy."
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"
    assert resp["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_tool_call_output_sets_finish_reason_and_null_content():
    out = {"text": "", "stop_reason": "tool_calls", "tool_calls": [
        {"id": "call_0", "name": "submit_jump_result", "arguments": '{"events":[]}'},
    ]}
    resp = translate.inferlet_output_to_chat_completion(
        out, model="pie", request_id="r", created=0)
    msg = resp["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "submit_jump_result"
    assert msg["tool_calls"][0]["type"] == "function"
    assert resp["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_call_promotes_stop_finish_reason():
    # inferlet reported "stop" but emitted a tool call — normalize to tool_calls.
    out = {"text": "", "stop_reason": "stop", "tool_calls": [
        {"name": "submit_actions", "arguments": "{}"},
    ]}
    resp = translate.inferlet_output_to_chat_completion(
        out, model="pie", request_id="r", created=0)
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp["choices"][0]["message"]["tool_calls"][0]["id"] == "call_0"


# ── inferlet output → stream chunks ─────────────────────────────────────────

def test_stream_chunks_reassemble_to_content_and_finish():
    out = {"text": "War breaks out.", "stop_reason": "stop"}
    chunks = translate.inferlet_output_to_stream_chunks(
        out, model="pie", request_id="r", created=0)
    # First chunk announces the role; a later chunk carries content; last has finish.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "War breaks out."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_stream_chunks_carry_tool_call_arguments():
    out = {"text": "", "stop_reason": "tool_calls", "tool_calls": [
        {"id": "c1", "name": "submit_game_master", "arguments": '{"ops":[]}'},
    ]}
    chunks = translate.inferlet_output_to_stream_chunks(
        out, model="pie", request_id="r", created=0)
    args = "".join(
        tc["function"].get("arguments", "")
        for c in chunks
        for tc in c["choices"][0]["delta"].get("tool_calls", []))
    assert args == '{"ops":[]}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
