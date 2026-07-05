"""Repair layer for a confirmed Qwen2.5-Coder + OpenHands non-native
tool-calling bug (not Pie-specific — reproduces identically on the plain
litellm backend, see docs/openhands-integration.md investigation notes).

Root cause: when filling ``old_str``/``new_str``/``file_text`` for the
``file_editor``/``str_replace_editor`` tool, the model sometimes emits the
two literal characters ``\\`` and ``n`` where a real newline belongs
(confirmed via raw completion logs — ``create``'s ``file_text`` is usually
fine, ``str_replace``'s ``new_str``/``old_str`` is where this shows up).
Applied literally, this corrupts the edit into a single garbled line.

This module monkeypatches the OpenHands SDK's non-native-tool-calling
message converter to detect and repair that exact signature after parsing,
before the edit is ever applied. It intentionally does *not* touch values
that already contain a real newline (those aren't exhibiting the bug), nor
short single-escape strings (too easy to confuse with a deliberate
``"\\n"`` inside an edited string literal) — see ``_looks_like_escape_bug``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

_EDITOR_TOOL_NAMES = {"file_editor", "str_replace_editor"}
_EDITOR_STRING_PARAMS = ("old_str", "new_str", "file_text")

# `\n` immediately followed by indentation-like whitespace then more text is
# the fingerprint from the confirmed bug trace (e.g. the raw completion had
# literal `...:\n    print(f(1))\n    print(f(2))`) — real single-line string
# literals containing an intentional "\n" essentially never look like this.
_ESCAPED_CONTINUATION = re.compile(r"\\n[ \t]+\S")

_repair_count = 0


def repair_count() -> int:
    """Number of tool-call arguments repaired since process start (for tests/logs)."""
    return _repair_count


def _looks_like_escape_bug(value: str) -> bool:
    if "\n" in value or "\r" in value:
        return False  # already has real newlines — not the bug
    if "\\n" not in value:
        return False
    return value.count("\\n") >= 2 or bool(_ESCAPED_CONTINUATION.search(value))


def _unescape(value: str) -> str:
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _repair_tool_call(tool_call: dict[str, Any]) -> None:
    global _repair_count
    fn = tool_call.get("function")
    if not isinstance(fn, dict) or fn.get("name") not in _EDITOR_TOOL_NAMES:
        return
    try:
        params = json.loads(fn["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return
    if not isinstance(params, dict):
        return

    changed = False
    for key in _EDITOR_STRING_PARAMS:
        val = params.get(key)
        if isinstance(val, str) and _looks_like_escape_bug(val):
            params[key] = _unescape(val)
            changed = True

    if changed:
        _repair_count += 1
        logger.warning(
            "editor_repair: un-escaped literal '\\n'/'\\t' in %s arguments "
            "(path=%r) — see docs/openhands-integration.md stuck-loop notes",
            fn.get("name"),
            params.get("path"),
        )
        fn["arguments"] = json.dumps(params)


def _wrap(
    original: Callable[..., list[dict]],
) -> Callable[..., list[dict]]:
    def wrapped(messages, tools, include_security_params=False):
        converted = original(
            messages, tools, include_security_params=include_security_params
        )
        for message in converted:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    _repair_tool_call(tool_call)
        return converted

    wrapped.__pie_editor_repair_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def install() -> None:
    """Idempotently monkeypatch the SDK's non-native tool-call converter.

    Affects both the ``pie`` and ``litellm`` backends equally (they share
    this base-class code path), which matters since the bug was confirmed
    to reproduce identically on both — see the investigation notes.
    """
    from openhands.sdk.llm.mixins import non_native_fc

    current = non_native_fc.convert_non_fncall_messages_to_fncall_messages
    if getattr(current, "__pie_editor_repair_wrapped__", False):
        return
    non_native_fc.convert_non_fncall_messages_to_fncall_messages = _wrap(current)
