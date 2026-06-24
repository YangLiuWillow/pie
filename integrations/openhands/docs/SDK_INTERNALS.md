# openhands-sdk 1.21.1 — Verified Internals

These are the *verified* signatures and data flows from reading installed `openhands-sdk==1.21.1`. The spec at `pie/docs/openhands-integration.md` was written before reading the source — anywhere this doc and the spec disagree, **this doc wins** for Phase 1.

Source root for this doc: `pie/integrations/openhands/.venv/lib/python3.13/site-packages/openhands/sdk/`.

## 1. LLM class

**File:** `llm/llm.py` (1648 lines). Class `LLM(BaseModel, RetryMixin, NonNativeToolCallingMixin)`.

### 1.1 The completion path

```
LLM.completion(messages: list[Message], tools, ...) -> LLMResponse        # line 699
   ├─ format_messages_for_llm(messages) -> list[dict]                     # line 1409
   ├─ should_mock_tool_calls + pre_request_prompt_mock                    # tool-call mocking for non-native FC
   ├─ select_chat_options(...) -> call_kwargs
   └─ retry_decorator wraps _one_attempt:
        └─ _transport_call(*, messages=list[dict], **kwargs) -> ModelResponse  # line 1128
              └─ litellm_completion(model=self.model,
                                    api_base=self.base_url,
                                    api_key=...,
                                    messages=messages,
                                    **kwargs)                             # line 1171
   └─ Message.from_llm_chat_message(resp["choices"][0]["message"])
   └─ return LLMResponse(message=..., metrics=MetricsSnapshot(...), raw_response=resp)
```

### 1.2 The cleanest interception point: `_transport_call`

```python
# llm/llm.py:1128
def _transport_call(
    self,
    *,
    messages: list[dict[str, Any]],     # already formatted as OpenAI-style dicts
    enable_streaming: bool = False,
    on_token: TokenCallbackType | None = None,
    **kwargs,                           # tools, temperature, max_tokens, stop, ...
) -> ModelResponse:                     # litellm.types.utils.ModelResponse
    ...
    ret = litellm_completion(
        model=self.model, api_key=..., api_base=self.base_url, ...,
        messages=messages, **kwargs,
    )
    assert isinstance(ret, ModelResponse)
    return ret
```

**Why override this and not `completion()`:**

- All message formatting, prompt-caching markers, tool-call schema injection, retry decoration, telemetry, and `LLMResponse` construction stay in the parent class.
- We only need to translate `(messages: list[dict], kwargs)` → Pie request, and Pie response → `ModelResponse`.
- The retry wrapper is OUTSIDE `_transport_call`, so any exception we raise gets retried with exponential backoff (good).

### 1.3 The `ModelResponse` shape we must return

`ModelResponse` is `litellm.types.utils.ModelResponse` — Pydantic, OpenAI-shaped:

```jsonc
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen3-coder-32b",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop"|"length"|"tool_calls",
      "message": {
        "role": "assistant",
        "content": "<generated text or null>",
        "tool_calls": [{
          "id": "call_xxx",
          "type": "function",
          "function": {"name": "...", "arguments": "<json>"}
        }]
      }
    }
  ],
  "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
}
```

For Phase 1 we emit text only (no native tool calls); OpenHands' non-native-FC mock path will parse tool calls out of the text.

## 2. Message types

**File:** `llm/message.py`. `Message` is a Pydantic model; `Message.from_llm_chat_message(dict) -> Message` is the inverse of `to_chat_dict()`. Dict shape is OpenAI-style: `{"role": "user"|"assistant"|"system"|"tool", "content": str | list[content-part]}`.

For `_transport_call` we receive already-formatted `list[dict]`; we do **not** need to import `Message` for Phase 1.

## 3. Model registry / instantiation

`LLM` is a Pydantic model with these key fields (`llm/llm.py:135-450`):
- `model: str`
- `base_url: str | None`
- `api_key: SecretStr | None`
- `temperature: float | None`
- `max_output_tokens: int | None`
- `num_retries: int`, `retry_min_wait`, `retry_max_wait`, `retry_multiplier`
- `native_tool_calling: bool` (default depends on model_features)
- `usage_id: str | None` (discriminator: 'agent', 'condenser', 'planning_condenser', etc.)
- `model_config: ConfigDict(extra='ignore')` — so we can add Pie-specific fields and they won't fail serialization

**Subclassing is supported** — the app server does it (`StrictLLM(LLM)` with `extra='forbid'`).

## 4. Agent loop (Phase 2 concern)

**File:** `agent/agent.py` (1053 lines). To be documented in Phase 2.

## 5. Conversation (Phase 2 concern)

**File:** `conversation/conversation.py` (202 lines), `conversation/base.py` (376 lines). To be documented in Phase 2.

## 6. Condenser (Phase 2 concern)

**File:** `context/condenser/llm_summarizing_condenser.py` (340 lines), `context/condenser/base.py` (184 lines). To be documented in Phase 2.

## 7. Test rig

The SDK has a built-in `LLM` mock under `openhands.sdk.testing` — we should look at it (Phase 1 day 5) to see if there's a pattern we can match instead of hand-rolling a stub.
