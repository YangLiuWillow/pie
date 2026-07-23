# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Standalone, dependency-free port of vLLM 0.16.0's
# ``vllm/tool_parsers/qwen3coder_tool_parser.py`` (Qwen3CoderToolParser).
#
# Why this exists: our openhands-coder-session inferlet generates the model's
# native Qwen3-Coder XML tool-call format (``<tool_call><function=NAME>
# <parameter=K>V</parameter></function></tool_call>``) but parses it with a
# stricter Rust decoder. The litellm+vLLM baseline uses THIS parser, which is
# far more lenient (it back-offs to treating the whole output as a candidate
# when the ``<tool_call>`` wrapper is missing, and only needs the substring
# ``<function=`` to appear). Parsing the inferlet's raw generation with this
# exact logic maximizes tool-call parity with the baseline.
#
# Only the *non-streaming* path is ported (``extract_tool_calls`` and its
# helpers). The streaming machinery, vLLM protocol types, and tokenizer
# dependency are dropped. ``tools`` are plain OpenAI-style dicts
# (``{"type": "function", "function": {"name", "parameters": {...}}}``) rather
# than vLLM pydantic objects, so ``_get_arguments_config`` is adapted to dicts.
#
# Returns a list of ``{"name": str, "arguments": <json-string>}`` — the shape
# the inferlet already produces and that ``PieLLM._wrap_as_model_response``
# consumes.
#
# One deliberate deviation: vLLM imports the third-party ``regex`` module,
# this port uses the standard library's ``re``. None of the patterns below
# use a ``regex``-only feature, and the full parser test suite passes either
# way, so the dependency buys nothing here — and it was never declared in
# our pyproject (it arrived transitively via ``transformers``). Do not
# "restore" it to match vLLM without a failing test to justify it.
from __future__ import annotations

import ast
import json
import re
from typing import Any

# Regex patterns — copied verbatim from the vLLM parser.
_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
)
_TOOL_CALL_FUNCTION_REGEX = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
)
_TOOL_CALL_PARAMETER_REGEX = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)

_TOOL_CALL_PREFIX = "<function="
_TOOL_CALL_START_TOKEN = "<tool_call>"


def _get_arguments_config(
    func_name: str, tools: list[dict[str, Any]] | None
) -> dict:
    """Extract the ``properties`` schema for ``func_name`` from OpenAI tools."""
    if not tools:
        return {}
    for config in tools:
        if not isinstance(config, dict) or config.get("type") != "function":
            continue
        fn = config.get("function") or {}
        if fn.get("name") != func_name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict) and "properties" in params:
            return params["properties"]
        elif isinstance(params, dict):
            return params
        return {}
    return {}


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """Coerce a raw XML parameter string to its schema type (vLLM logic)."""
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        return param_value

    if (
        isinstance(param_config.get(param_name), dict)
        and "type" in param_config[param_name]
    ):
        param_type = str(param_config[param_name]["type"]).strip().lower()
    else:
        param_type = "string"

    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        return param_value == "true"
    else:
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        try:
            param_value = ast.literal_eval(param_value)
        except (ValueError, SyntaxError, TypeError):
            pass
        return param_value


def _parse_xml_function_call(
    function_call_str: str, tools: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Parse a single ``NAME>...<parameter=...>`` body into name+arguments."""
    try:
        end_index = function_call_str.index(">")
    except ValueError:
        return None
    function_name = function_call_str[:end_index]
    param_config = _get_arguments_config(function_name, tools)
    parameters = function_call_str[end_index + 1:]
    param_dict: dict[str, Any] = {}
    for match_text in _TOOL_CALL_PARAMETER_REGEX.findall(parameters):
        if ">" not in match_text:
            continue
        idx = match_text.index(">")
        param_name = match_text[:idx]
        param_value = str(match_text[idx + 1:])
        # Remove one leading / trailing newline, as vLLM does.
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config, function_name
        )
    return {
        "name": function_name,
        "arguments": json.dumps(param_dict, ensure_ascii=False),
    }


def _get_function_calls(model_output: str) -> list[str]:
    """Extract the raw ``<function=...>`` bodies, with vLLM's back-off."""
    matched_ranges = _TOOL_CALL_REGEX.findall(model_output)
    raw_tool_calls = [m[0] if m[0] else m[1] for m in matched_ranges]

    # Back-off: no <tool_call> tags -> treat the whole output as a candidate.
    if len(raw_tool_calls) == 0:
        raw_tool_calls = [model_output]

    raw_function_calls = []
    for tool_call in raw_tool_calls:
        raw_function_calls.extend(_TOOL_CALL_FUNCTION_REGEX.findall(tool_call))

    return [m[0] if m[0] else m[1] for m in raw_function_calls]


def extract_tool_calls(
    model_output: str, tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Return ``[{"name", "arguments"}]`` parsed from a raw generation string.

    Mirrors ``Qwen3CoderToolParser.extract_tool_calls`` (non-streaming). Returns
    an empty list when no ``<function=`` marker is present or on any parse error.
    """
    if _TOOL_CALL_PREFIX not in model_output:
        return []
    try:
        function_calls = _get_function_calls(model_output)
        if not function_calls:
            return []
        parsed = [
            _parse_xml_function_call(fc, tools) for fc in function_calls
        ]
        return [p for p in parsed if p is not None]
    except Exception:
        return []
