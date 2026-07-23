"""Tests for the vendored vLLM qwen3_coder tool parser and the PieLLM
host-side re-parse path (``pie_python_tool_parser``)."""
from __future__ import annotations

import json
import types

from litellm.types.utils import ModelResponse

from pie_openhands import PieLLM
from pie_openhands import qwen3coder_parser as qp

FILE_EDITOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_editor",
            "parameters": {
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "view_range": {"type": "array"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "parameters": {
                "properties": {
                    "command": {"type": "string"},
                    "is_input": {"type": "boolean"},
                },
            },
        },
    },
]

STR_REPLACE_XML = (
    "I'll fix the check_filterable method now.\n"
    "<tool_call>\n"
    "<function=file_editor>\n"
    "<parameter=command>\n"
    "str_replace\n"
    "</parameter>\n"
    "<parameter=path>\n"
    "/workspace/repo/django/db/models/sql/query.py\n"
    "</parameter>\n"
    "<parameter=old_str>\n"
    "        if not getattr(expression, 'filterable', True):\n"
    "</parameter>\n"
    "<parameter=new_str>\n"
    "        if isinstance(expression, BaseExpression):\n"
    "            if not getattr(expression, 'filterable', True):\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_parses_str_replace_xml():
    calls = qp.extract_tool_calls(STR_REPLACE_XML, FILE_EDITOR_TOOLS)
    assert len(calls) == 1
    assert calls[0]["name"] == "file_editor"
    args = json.loads(calls[0]["arguments"])
    assert args["command"] == "str_replace"
    assert args["path"].endswith("query.py")
    assert args["old_str"] == "        if not getattr(expression, 'filterable', True):"
    assert "isinstance(expression, BaseExpression)" in args["new_str"]
    # old_str/new_str present — the exact params the grammar-forced path dropped.
    assert set(args) == {"command", "path", "old_str", "new_str"}


def test_type_coercion_matches_schema():
    xml = (
        "<tool_call>\n<function=terminal>\n"
        "<parameter=command>\nls -la\n</parameter>\n"
        "<parameter=is_input>\nfalse\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    args = json.loads(qp.extract_tool_calls(xml, FILE_EDITOR_TOOLS)[0]["arguments"])
    assert args["command"] == "ls -la"
    assert args["is_input"] is False  # coerced boolean, not the string "false"


def test_lenient_backoff_without_tool_call_wrapper():
    # vLLM's back-off: a <function=…> block with no <tool_call> wrapper still
    # parses. This is the leniency that recovers stalled coder-session turns.
    xml = (
        "Here is the edit:\n"
        "<function=file_editor>\n"
        "<parameter=command>\nview\n</parameter>\n"
        "<parameter=path>\n/workspace/x.py\n</parameter>\n"
        "</function>"
    )
    calls = qp.extract_tool_calls(xml, FILE_EDITOR_TOOLS)
    assert len(calls) == 1
    assert calls[0]["name"] == "file_editor"
    assert json.loads(calls[0]["arguments"])["command"] == "view"


def test_no_function_marker_returns_empty():
    assert qp.extract_tool_calls("Just some prose, no tool call.", FILE_EDITOR_TOOLS) == []


def test_multiple_calls():
    xml = (
        "<tool_call>\n<function=terminal>\n<parameter=command>\npwd\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = qp.extract_tool_calls(xml, FILE_EDITOR_TOOLS)
    assert [json.loads(c["arguments"])["command"] for c in calls] == ["pwd", "ls"]


# ---------------------------------------------------------------
# PieLLM transport path with pie_python_tool_parser=True
# ---------------------------------------------------------------


def _patch_call_pie(llm: PieLLM, response: dict):
    async def fake(self, messages, tools, gen_params):
        fake.captured = {"messages": messages, "tools": tools, "gen_params": gen_params}
        return response

    llm._call_pie = types.MethodType(fake, llm)
    return fake


def _make_llm(**overrides):
    defaults = dict(
        model="qwen3-coder-30b",
        pie_uri="ws://127.0.0.1:8080",
        native_tool_calling=True,
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=1.0,
    )
    defaults.update(overrides)
    return PieLLM(**defaults)


def test_python_parser_recovers_call_inferlet_missed():
    """Inferlet returns EMPTY tool_calls (the 13028 stall), but the raw
    generation contains a valid <function=> call — the Python path recovers it
    and forces use_grammar off + skips JSON few-shots."""
    llm = _make_llm(pie_python_tool_parser=True)
    fake = _patch_call_pie(
        llm,
        response={
            "text": "I'll fix the check_filterable method now.",
            "tool_calls": [],  # inferlet decoder found nothing
            "stop_reason": "stop",
            "debug_full_text": STR_REPLACE_XML,
        },
    )

    resp = llm._transport_call(
        messages=[{"role": "user", "content": "fix it"}],
        tools=FILE_EDITOR_TOOLS,
        temperature=0.0,
        max_tokens=64,
    )

    # Unconstrained generation was forced, and no JSON few-shot was injected.
    assert fake.captured["gen_params"]["use_grammar"] is False
    assert "START OF EXAMPLE" not in fake.captured["messages"][0]["content"]

    assert isinstance(resp, ModelResponse)
    assert resp.choices[0].finish_reason == "tool_calls"
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].function.name == "file_editor"
    args = json.loads(tcs[0].function.arguments)
    assert args["command"] == "str_replace"
    assert "old_str" in args and "new_str" in args
    # Content is trimmed to the prose before the tool call.
    assert resp.choices[0].message.content == "I'll fix the check_filterable method now.\n"


def test_python_parser_leaves_inferlet_calls_when_no_recovery():
    """If the raw text has no <function=>, don't clobber the inferlet's own
    tool_calls (never regress a turn the Rust decoder handled)."""
    llm = _make_llm(pie_python_tool_parser=True)
    _patch_call_pie(
        llm,
        response={
            "text": "done",
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls"}'}],
            "stop_reason": "tool_calls",
            "debug_full_text": "just prose, no markup",
        },
    )

    resp = llm._transport_call(
        messages=[{"role": "user", "content": "go"}],
        tools=FILE_EDITOR_TOOLS,
        temperature=0.0,
    )
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].function.name == "terminal"
