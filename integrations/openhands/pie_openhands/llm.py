"""PieLLM: an openhands.sdk.LLM that routes the LiteLLM HTTP call through Pie.

Phase 1 ("dumb integration"): we keep all of OpenHands' message formatting,
tool-call mocking, retry decoration, telemetry, and LLMResponse construction.
We only override `_transport_call`, which is where the parent class would
otherwise call `litellm.completion(...)`. Instead, we launch a Pie inferlet
that consumes the rendered prompt and returns generated text.

See: pie/integrations/openhands/docs/SDK_INTERNALS.md  §1.2
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from litellm.types.utils import (
    Choices,
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
    and non-native tool calling, prompt caching markers, fallback strategies)
    continues to work — we hook in *below* it.
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
    pie_render_strategy: str = Field(
        default="hf_chat_template",
        description=(
            "How to convert OpenAI-format messages into a Pie prompt string. "
            "One of: 'hf_chat_template' (transformers.AutoTokenizer), 'raw_concat' "
            "(debug-only: role: content concatenation)."
        ),
    )

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
        (the parent class ran format_messages_for_llm before us).
        We render to a single prompt string, ship it to Pie, and translate
        the inferlet's response back into a ``ModelResponse``.

        Streaming is **not** supported in Phase 1 — we raise if requested.
        """
        if enable_streaming:
            _ = on_token  # signature parity with parent; consumed in Phase 2
            raise NotImplementedError(
                "PieLLM does not support streaming in Phase 1. "
                "Wire on_token through the inferlet's session.send chunks in Phase 2."
            )

        prompt = self._render_prompt(messages, kwargs.get("tools"))
        gen_params = self._extract_gen_params(kwargs)
        raw = asyncio.run(self._call_pie(prompt, gen_params))
        return self._wrap_as_model_response(raw, prompt)

    # ------------------------------------------------------------------
    # Rendering: messages (list[dict]) -> single prompt string
    # ------------------------------------------------------------------
    def _render_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
    ) -> str:
        strategy = self.pie_render_strategy
        if strategy == "hf_chat_template":
            return self._render_via_hf_chat_template(messages, tools)
        if strategy == "raw_concat":
            return self._render_via_raw_concat(messages)
        raise ValueError(f"Unknown pie_render_strategy: {strategy!r}")

    def _render_via_hf_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
    ) -> str:
        # Lazy import — keeps test suite fast when caller mocks _call_pie.
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        # Normalize message dicts: ensure 'content' is a string. OpenHands sometimes
        # emits structured content (list of parts) when vision is active; for
        # Phase 1 we collapse those to text.
        norm = [_flatten_content(m) for m in messages]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        return tok.apply_chat_template(norm, **kwargs)

    def _render_via_raw_concat(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            parts.append(f"{m.get('role', 'user')}: {content or ''}")
        parts.append("assistant:")
        return "\n".join(parts)

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
    async def _call_pie(self, prompt: str, gen_params: dict[str, Any]) -> dict[str, Any]:
        """Launch the inferlet, collect its Return payload, return as dict.

        Override this in tests with a mock to avoid spinning up a Pie server.
        """
        input_payload = {"prompt": prompt, **gen_params}
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
    def _wrap_as_model_response(
        self, pie_out: dict[str, Any], prompt: str
    ) -> ModelResponse:
        text = pie_out.get("text", "")
        stop_reason = pie_out.get("stop_reason") or "stop"
        prompt_tokens = int(pie_out.get("prompt_tokens") or _approx_token_count(prompt))
        completion_tokens = int(
            pie_out.get("tokens_generated")
            or pie_out.get("completion_tokens")
            or _approx_token_count(text)
        )

        finish_reason = {
            "stop": "stop",
            "eos": "stop",
            "length": "length",
        }.get(stop_reason, "stop")

        msg = LiteLLMMessage(role="assistant", content=text, tool_calls=None)
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
