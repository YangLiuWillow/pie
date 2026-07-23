"""Unit tests for the editor-escape repair monkeypatch.

Exercises the real ``openhands.sdk`` conversion function end-to-end (no
mocking of it) to make sure the wrapped version still returns correctly
shaped messages and only repairs the confirmed bug signature.
"""

from __future__ import annotations

import json

from pie_openhands import editor_repair


def _fake_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "file_editor",
                "description": "Edit a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "file_text": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        }
    ]


def _assistant_fn_message(fn_body: str) -> dict:
    return {
        "role": "assistant",
        "content": f"<function=file_editor>\n{fn_body}\n</function>",
    }


def _convert(fn_body: str):
    from openhands.sdk.llm.mixins.non_native_fc import (
        convert_non_fncall_messages_to_fncall_messages,
    )

    editor_repair.install()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        _assistant_fn_message(fn_body),
    ]
    return convert_non_fncall_messages_to_fncall_messages(messages, _fake_tools())


def _params_of(converted: list[dict]) -> dict:
    tool_call = converted[-1]["tool_calls"][0]
    return json.loads(tool_call["function"]["arguments"])


def test_install_is_idempotent():
    from openhands.sdk.llm.mixins import non_native_fc

    editor_repair.install()
    once = non_native_fc.convert_non_fncall_messages_to_fncall_messages
    editor_repair.install()
    assert non_native_fc.convert_non_fncall_messages_to_fncall_messages is once


def test_repairs_escaped_multiline_new_str():
    before = editor_repair.repair_count()
    fn_body = (
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/repo/isympy.py</parameter>\n"
        "<parameter=old_str>print(f(1))</parameter>\n"
        "<parameter=new_str>if __name__ == '__main__':\\n    print(f(1))\\n    print(f(2))</parameter>"
    )
    params = _params_of(_convert(fn_body))
    assert params["new_str"] == "if __name__ == '__main__':\n    print(f(1))\n    print(f(2))"
    assert editor_repair.repair_count() == before + 1


def test_leaves_real_newlines_untouched():
    fn_body = (
        "<parameter=command>create</parameter>\n"
        "<parameter=path>/repo/new_file.py</parameter>\n"
        "<parameter=file_text>\ndef f():\n    return 1\n</parameter>"
    )
    params = _params_of(_convert(fn_body))
    # SDK itself .strip()s parameter values; only checking real newlines survive.
    assert params["file_text"] == "def f():\n    return 1"


def test_leaves_single_intentional_escape_untouched():
    """A short, single ``\\n`` inside an edited string literal is genuine
    source code (e.g. editing ``print("hello\\n")``), not the bug — must
    not be mangled."""
    fn_body = (
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/repo/greet.py</parameter>\n"
        "<parameter=old_str>print(\"hi\")</parameter>\n"
        "<parameter=new_str>print(\"hello\\n\")</parameter>"
    )
    params = _params_of(_convert(fn_body))
    assert params["new_str"] == 'print("hello\\n")'
