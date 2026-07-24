"""Pure OpenAI ⇄ Pie-inferlet translation — no I/O, fully unit-testable.

Open Historia's browser transport (`src/Game/AI/main.jsx`,
`callOpenAIStyleChatCompletions`) speaks the OpenAI `/chat/completions` dialect.
This module maps that request onto the `openhands-completion` inferlet's input
contract and maps the inferlet's Return payload back into an OpenAI
ChatCompletion (or a stream of chunks).

The inferlet contract (authoritative source: the doc-comment atop
`inferlets/openhands-completion/src/lib.rs` — NOT the stale `prompt`-based
`Pie.toml [parameters]` block):

    Input:  { messages, tools?, max_tokens?, temperature?, top_p?, stop?, model? }
    Output: { text, tool_calls:[{id,name,arguments}], stop_reason,
              prompt_tokens, tokens_generated }

`arguments` is a JSON-encoded string on the way out, OpenAI-style.
"""

from __future__ import annotations

from typing import Any

# Open Historia floors its structured-task budget at 8192 (main.jsx: Math.max(8192,
# maxTokens)); we honor whatever it sends and only default when it sends nothing.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95

# stop_reason (inferlet) → finish_reason (OpenAI)
_FINISH_REASON = {
    "stop": "stop",
    "eos": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
}


def _coerce_content(content: Any) -> str | None:
    """OpenAI allows `content` to be a string, null, or an array of parts.

    The inferlet wants a plain string (or absent). Flatten the array form by
    concatenating its text parts; pass strings through; keep null as null.
    """
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _normalize_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def wants_tools(body: dict[str, Any]) -> bool:
    """Whether tools should be forwarded to the inferlet.

    Open Historia forces exactly one tool with `tool_choice: "required"` (the
    string form — llama.cpp servers reject the object form). We forward tools
    unless the caller explicitly opted out with `tool_choice: "none"` or sent no
    tools at all. The inferlet itself does the forcing when tools are present.
    """
    tools = body.get("tools")
    if not tools:
        return False
    choice = body.get("tool_choice")
    if choice == "none":
        return False
    return True


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages or []:
        norm: dict[str, Any] = {"role": msg.get("role", "user")}
        content = _coerce_content(msg.get("content"))
        if content is not None:
            norm["content"] = content
        # Replay past assistant tool-calls verbatim; the inferlet only reads the
        # nested `function` object, so keep that shape.
        if msg.get("tool_calls"):
            norm["tool_calls"] = [
                {"function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }}
                for tc in msg["tool_calls"]
            ]
        if msg.get("tool_call_id"):
            norm["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            norm["name"] = msg["name"]
        out.append(norm)
    return out


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function", tool)  # tolerate a bare function object
        out.append({"function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }})
    return out


def chat_request_to_inferlet_input(body: dict[str, Any]) -> dict[str, Any]:
    """Map an OpenAI /chat/completions request body → inferlet input dict."""
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")  # OpenAI's newer field
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS

    payload: dict[str, Any] = {
        "messages": _normalize_messages(body.get("messages", [])),
        "max_tokens": int(max_tokens),
        "temperature": float(body.get("temperature", DEFAULT_TEMPERATURE)),
        "top_p": float(body.get("top_p", DEFAULT_TOP_P)),
        "stop": _normalize_stop(body.get("stop")),
    }
    if body.get("model"):
        payload["model"] = body["model"]
    if wants_tools(body):
        payload["tools"] = _normalize_tools(body["tools"])
    return payload


def _tool_calls_out(out: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for i, tc in enumerate(out.get("tool_calls") or []):
        result.append({
            "id": tc.get("id") or f"call_{i}",
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                # inferlet already emits a JSON-encoded string
                "arguments": tc.get("arguments", ""),
            },
        })
    return result


def _usage(out: dict[str, Any]) -> dict[str, int]:
    prompt = int(out.get("prompt_tokens") or 0)
    completion = int(out.get("tokens_generated") or out.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def inferlet_output_to_chat_completion(
    out: dict[str, Any], *, model: str, request_id: str, created: int
) -> dict[str, Any]:
    """Map the inferlet Return payload → a buffered OpenAI ChatCompletion."""
    text = out.get("text") or ""
    tool_calls = _tool_calls_out(out)
    finish_reason = _FINISH_REASON.get(out.get("stop_reason") or "stop", "stop")

    message: dict[str, Any] = {"role": "assistant"}
    # OpenAI convention: content may be null when the turn is purely tool calls.
    message["content"] = text if text or not tool_calls else None
    if tool_calls:
        message["tool_calls"] = tool_calls
        if finish_reason == "stop":
            finish_reason = "tool_calls"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _usage(out),
    }


def inferlet_output_to_stream_chunks(
    out: dict[str, Any], *, model: str, request_id: str, created: int
) -> list[dict[str, Any]]:
    """Map the inferlet Return payload → an ordered list of SSE chunk objects.

    Scaffold behavior: the inferlet call is buffered, so we replay the finished
    result as a short sequence of chunks. Open Historia's `readOpenAIStreamedResponse`
    concatenates `delta.content` and `delta.tool_calls[].function.arguments`
    across chunks and reads `finish_reason`, so emitting the whole content /
    arguments in one delta is valid. (Real token-level streaming — which makes
    Cancel physical — is a TODO that needs the inferlet to emit tokens on
    Stdout; see README §Limitations.)
    """
    text = out.get("text") or ""
    tool_calls = _tool_calls_out(out)
    finish_reason = _FINISH_REASON.get(out.get("stop_reason") or "stop", "stop")
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    def chunk(delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    chunks = [chunk({"role": "assistant"}, None)]
    if text:
        chunks.append(chunk({"content": text}, None))
    for i, tc in enumerate(tool_calls):
        chunks.append(chunk({"tool_calls": [{
            "index": i,
            "id": tc["id"],
            "type": "function",
            "function": tc["function"],
        }]}, None))
    chunks.append(chunk({}, finish_reason))
    return chunks
