"""Make the [system + tools] prompt head instance-invariant.

OpenHands' ``FileEditorTool.create`` appends the per-conversation workspace
path to the tool description ("Your current working directory is: ...").
That path is the ONLY per-instance byte in the first render unit the
openhands-coder-session inferlet snapshots (``render_prompt`` folds the tool
schemas into the system turn), so with it present the shared-namespace head
snapshot can never be hit across conversations — measured 2026-07-31: the
two-instance global-APC arm's second instance still cold-rebuilt.

The line is also redundant for this harness: ``_format_user_prompt`` states
the checkout path in the first user message, and tool output leaks it from
the first ``view``/``terminal`` call onward.

This module monkeypatches ``FileEditorTool.create`` to strip that suffix.
Installed by ``build_llm`` for every backend — pie and litellm arms see
byte-identical tool schemas, preserving cross-arm prompt equivalence.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Matches the exact suffix appended in openhands/tools/file_editor/
# definition.py::FileEditorTool.create (DOTALL: the path is arbitrary).
_CWD_SUFFIX = re.compile(
    r"\n\nYour current working directory is: .*?"
    r"\nWhen exploring project structure, start with this directory "
    r"instead of the root filesystem\.\s*$",
    re.DOTALL,
)


def strip_cwd_suffix(description: str) -> str:
    return _CWD_SUFFIX.sub("", description)


def install() -> None:
    """Idempotently monkeypatch ``FileEditorTool.create`` to drop the suffix.

    The tool registry snapshots the bound ``create`` when the definition
    module registers itself at import (``_resolver_from_subclass`` does
    ``getattr(cls, "create")`` once), so patching the class alone is
    invisible to spec-resolved tools — re-register after patching so the
    resolver is rebuilt from the wrapped classmethod.
    """
    from openhands.sdk.tool import register_tool
    from openhands.tools.file_editor.definition import FileEditorTool

    original = FileEditorTool.create.__func__
    if getattr(original, "__pie_cwd_strip_wrapped__", False):
        return

    def wrapped(cls, conv_state):
        stripped = []
        for tool in original(cls, conv_state):
            desc = tool.description or ""
            new_desc = strip_cwd_suffix(desc)
            if new_desc != desc:
                tool = tool.model_copy(update={"description": new_desc})
            stripped.append(tool)
        return stripped

    wrapped.__pie_cwd_strip_wrapped__ = True  # type: ignore[attr-defined]
    FileEditorTool.create = classmethod(wrapped)  # type: ignore[method-assign]
    register_tool(FileEditorTool.name, FileEditorTool)
