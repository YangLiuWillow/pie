"""PieLLM: an openhands.sdk.LLM that routes the LiteLLM HTTP call through Pie.

We keep all of OpenHands' message formatting, retry decoration, telemetry,
and LLMResponse construction. We only override `_transport_call`, which is
where the parent class would otherwise call `litellm.completion(...)`.
Instead, we launch a Pie inferlet that consumes the structured message/tool
history and returns generated text plus any native tool calls.

The inferlet (not PieLLM) owns chat-template rendering and history replay —
see `inferlets/openhands-completion/src/lib.rs` and
`integrations/openhands/docs/TOOL_CALL_HISTORY_REPLAY_DESIGN.md`.

See: pie/integrations/openhands/docs/SDK_INTERNALS.md  §1.2
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from openhands.sdk.llm import LLM
from openhands.sdk.llm.streaming import TokenCallbackType
from pydantic import Field
from pie_client import Event, PieClient


class PieLLM(LLM):
    """openhands.sdk.LLM subclass that routes the transport layer to Pie.

    All openhands-sdk machinery (retry, telemetry, message formatting, native
    tool calling, prompt caching markers, fallback strategies) continues to
    work — we hook in *below* it.
    """

    # ------- Pie-specific config (extra fields on top of LLM) ----------
    pie_uri: str = Field(
        default="ws://127.0.0.1:8080",
        description="Pie server WebSocket URI",
    )
    pie_username: str = Field(
        default="local-dev",
        description="Pie auth username (see `pie auth`)",
    )
    pie_inferlet: str = Field(
        default="openhands-completion",
        description="Inferlet name registered with `pie serve`",
    )
    pie_request_timeout_s: float = Field(
        default=600.0,
        description="Hard timeout per Pie request",
    )

    # `_wrap_as_model_response` now populates real `tool_calls` from the
    # inferlet's structured output (see `assistant_with_tool_calls`/
    # `answer_batch` on the runtime's `Instruct` trait), so the base class's
    # native tool-calling path is used instead of its prompt-mocked one.

    # `LLM.model_config` already sets extra='ignore', so unknown kwargs in
    # base_url / api_key / api_version that we don't use here are harmless.

    # ------------------------------------------------------------------
    # The override
    # ------------------------------------------------------------------
    def _transport_call(
        self,
        *,
        messages: list[dict[str, Any]],
        enable_streaming: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Replace litellm.completion(...) with a Pie inferlet round-trip.

        ``messages`` arrives already formatted as OpenAI-style chat dicts
        (the parent class ran format_messages_for_llm before us). We forward
        them — plus any ``tools`` — to the inferlet as structured JSON; the
        inferlet does its own chat-template rendering and history replay.

        Streaming is **not** supported yet — we raise if requested.
        """
        if enable_streaming:
            _ = on_token  # signature parity with parent; consumed in a later phase
            raise NotImplementedError(
                "PieLLM does not support streaming yet. "
                "Wire on_token through the inferlet's session.send chunks."
            )

        wire_messages = [_flatten_content(m) for m in messages]
        # In non-native mode the SDK embeds tool descriptions in the prompt
        # text and passes tools in a flat format the inferlet can't parse.
        tools = (kwargs.get("tools") or []) if self.native_tool_calling else []

        if tools and self.native_tool_calling:
            _inject_native_examples(wire_messages, tools)

        gen_params = self._extract_gen_params(kwargs)
        raw = asyncio.run(self._call_pie(wire_messages, tools, gen_params))
        _sanitize_tool_args(raw, tools)
        return self._wrap_as_model_response(raw)

    # ------------------------------------------------------------------
    # Parameter extraction from kwargs
    # ------------------------------------------------------------------
    def _extract_gen_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_tokens": int(kwargs.get("max_tokens") or self.max_output_tokens or 2048),
            "temperature": float(kwargs.get("temperature")
                                 if kwargs.get("temperature") is not None
                                 else (self.temperature if self.temperature is not None else 0.0)),
            "top_p": float(kwargs.get("top_p") or 0.95),
            "stop": list(kwargs.get("stop") or []),
            "model": self.model,
        }

    # ------------------------------------------------------------------
    # The Pie round-trip
    # ------------------------------------------------------------------
    async def _call_pie(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        gen_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Launch the inferlet, collect its Return payload, return as dict.

        Override this in tests with a mock to avoid spinning up a Pie server.
        """
        input_payload = {"messages": messages, "tools": tools, **gen_params}
        async with PieClient(self.pie_uri) as client:
            await client.authenticate(self.pie_username)
            proc = await client.launch_process(self.pie_inferlet, input=input_payload)
            stdout_chunks: list[str] = []
            while True:
                event, value = await asyncio.wait_for(
                    proc.recv(), timeout=self.pie_request_timeout_s
                )
                if event == Event.Stdout:
                    if isinstance(value, (bytes, bytearray)):
                        stdout_chunks.append(value.decode("utf-8", "replace"))
                    else:
                        stdout_chunks.append(str(value))
                elif event == Event.Return:
                    return _parse_return_value(value, stdout_chunks)
                elif event == Event.Error:
                    raise RuntimeError(f"Pie inferlet error: {value!r}")
                # Stderr / Message / File: ignore in Phase 1

    # ------------------------------------------------------------------
    # Pie response -> ModelResponse
    # ------------------------------------------------------------------
    def _wrap_as_model_response(self, pie_out: dict[str, Any]) -> ModelResponse:
        text = pie_out.get("text", "")
        stop_reason = pie_out.get("stop_reason") or "stop"
        prompt_tokens = int(pie_out.get("prompt_tokens") or 0)
        completion_tokens = int(
            pie_out.get("tokens_generated")
            or pie_out.get("completion_tokens")
            or _approx_token_count(text)
        )

        finish_reason = {
            "stop": "stop",
            "eos": "stop",
            "length": "length",
            "tool_calls": "tool_calls",
        }.get(stop_reason, "stop")

        # The inferlet numbers tool calls `call_0`, `call_1`, ... starting
        # over from zero on *every* request (it rebuilds the conversation
        # from scratch each turn — see its own module doc). Since most turns
        # emit exactly one tool call, that means nearly every tool call
        # across an entire multi-turn conversation would otherwise be handed
        # the identical id "call_0". OpenHands SDK's `ObservationUniquenessProperty`
        # dedups observations by `tool_call_id`, so it would then treat every
        # tool result after the first as a duplicate of the same call and
        # silently drop it from the context the model sees — starving the
        # agent of feedback and causing it to repeat the same action forever.
        # A uuid keeps ids unique across the whole conversation, not just
        # within one request.
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=Function(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in (pie_out.get("tool_calls") or [])
        ] or None

        msg = LiteLLMMessage(role="assistant", content=text, tool_calls=tool_calls)
        choice = Choices(index=0, message=msg, finish_reason=finish_reason)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ModelResponse(
            id=f"pie-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.model,
            object="chat.completion",
            choices=[choice],
            usage=usage,
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _flatten_content(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *message* with content guaranteed to be a string.

    OpenHands' format_messages_for_llm emits ``content`` as either a string
    or a list of content parts (when vision is active). HF chat templates
    only support string content; collapse parts to their text fields.
    """
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
        return {**message, "content": text}
    return message


def _sanitize_tool_args(
    pie_out: dict[str, Any],
    tools: list[dict[str, Any]],
) -> None:
    """Strip unknown keys from tool-call arguments in-place.

    Small models sometimes append garbage key-value pairs (e.g. ``", ": ","``
    after the real argument) that survive the layer-1 JSON grammar but get
    rejected by Pydantic ``extra='forbid'`` on OpenHands action models,
    sending the agent into a stuck error loop.
    """
    schema_map: dict[str, set[str]] = {}
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        name = fn.get("name", "")
        props = fn.get("parameters", {}).get("properties", {})
        if name and props:
            schema_map[name] = set(props.keys())

    for tc in pie_out.get("tool_calls") or []:
        valid_keys = schema_map.get(tc.get("name", ""))
        if valid_keys is None:
            continue
        try:
            args = json.loads(tc["arguments"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        sanitized = {k: v for k, v in args.items() if k in valid_keys}
        if sanitized != args:
            tc["arguments"] = json.dumps(sanitized)


def _parse_return_value(value: Any, stdout_chunks: list[str]) -> dict[str, Any]:
    """Normalize a Pie inferlet's Return payload into a dict.

    Inferlets may return:
      * a JSON-encoded string  -> parse it
      * a dict (already decoded by the client)
      * a plain string         -> treat as ``{"text": value}``
      * None                   -> fall back to concatenated stdout chunks
    """
    if value is None:
        return {"text": "".join(stdout_chunks)}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"text": value}
    return {"text": str(value)}


def _approx_token_count(text: str) -> int:
    """Rough character-count-based token estimate for usage fields.

    Real token counts should come from the inferlet's ``Output``. This is
    only a fallback when the inferlet doesn't report them.
    """
    return max(1, len(text) // 4)


# ----------------------------------------------------------------------
# Native-format few-shot examples
# ----------------------------------------------------------------------
# Mirrors OpenHands's non-native ``fn_call_examples`` but uses the native
# ``<tool_call>`` JSON format that Qwen's chat template expects.  Injected
# into the first user message so the model sees concrete examples of
# correct parameter usage — especially ``old_str``/``new_str`` for
# ``file_editor``'s ``str_replace`` command.

_TOOL_CALL = '<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'

_NATIVE_EXAMPLES: dict[str, dict[str, str]] = {
    "terminal": {
        "check_dir": (
            'ASSISTANT:\n'
            + _TOOL_CALL.format(
                name="terminal",
                args='{"command": "pwd && ls", "security_risk": "LOW", '
                     '"summary": "Check current directory and list files"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [terminal]:\n'
            '/workspace\nopenhands@runtime:~/workspace$\n'
        ),
    },
    "file_editor": {
        "create_file": (
            'ASSISTANT: Let me create the file:\n'
            + _TOOL_CALL.format(
                name="file_editor",
                args='{"command": "create", "path": "/workspace/app.py", '
                     '"file_text": "from flask import Flask\\napp = Flask(__name__)\\n\\n'
                     '@app.route(\\"/\\")\\ndef index():\\n    numbers = list(range(1, 11))\\n'
                     '    return str(numbers)\\n\\nif __name__ == \\"__main__\\":\\n'
                     '    app.run(port=5000)\\n", '
                     '"security_risk": "MEDIUM", '
                     '"summary": "Create Flask app.py with number list endpoint"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [file_editor]:\n'
            'File created successfully at: /workspace/app.py\n'
        ),
        "edit_file": (
            'ASSISTANT: Now let me edit the file:\n'
            + _TOOL_CALL.format(
                name="file_editor",
                args='{"command": "str_replace", "path": "/workspace/app.py", '
                     '"old_str": "return str(numbers)", '
                     '"new_str": "return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' '
                     'for i in numbers]) + \'</table>\'", '
                     '"security_risk": "MEDIUM", '
                     '"summary": "Update return statement to render HTML table"}',
            )
            + '\n\nUSER: EXECUTION RESULT of [file_editor]:\n'
            'The file /workspace/app.py has been edited. Here\'s the result of running `cat -n`:\n'
            '     5\tdef index():\n'
            '     6\t    numbers = list(range(1, 11))\n'
            '     7\t    return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' '
            'for i in numbers]) + \'</table>\'\n'
        ),
    },
    "finish": {
        "example": (
            'ASSISTANT: The task is complete.\n'
            + _TOOL_CALL.format(
                name="finish",
                args='{"message": "I have created the Flask app and updated it to display '
                     'numbers in a table format.", '
                     '"security_risk": "LOW", '
                     '"summary": "Complete task"}',
            )
        ),
    },
}


def _inject_native_examples(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Prepend native-format few-shot examples to the first user message."""
    tool_names = {
        t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"
    }

    parts: list[str] = []
    if "terminal" in tool_names:
        parts.append(_NATIVE_EXAMPLES["terminal"]["check_dir"])
    if "file_editor" in tool_names:
        parts.append(_NATIVE_EXAMPLES["file_editor"]["create_file"])
    if "file_editor" in tool_names:
        parts.append(_NATIVE_EXAMPLES["file_editor"]["edit_file"])
    if "finish" in tool_names:
        parts.append(_NATIVE_EXAMPLES["finish"]["example"])

    if not parts:
        return

    example_block = (
        "Here's a running example of how to perform a task with the provided tools.\n\n"
        "--------------------- START OF EXAMPLE ---------------------\n\n"
        "USER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.\n\n"
        + "\n".join(parts)
        + "\n\n--------------------- END OF EXAMPLE ---------------------\n\n"
        "Do NOT assume the environment is the same as in the example above.\n\n"
        "--------------------- NEW TASK DESCRIPTION ---------------------\n"
    )

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = example_block + content
            elif isinstance(content, list):
                if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                    content[0]["text"] = example_block + content[0].get("text", "")
            break
