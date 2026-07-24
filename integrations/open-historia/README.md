# pie-openhistoria

An **OpenAI-compatible HTTP adapter** that lets [Open Historia](https://github.com/Open-Historia/open-historia)
— or any client speaking OpenAI's `/chat/completions` — run against a model
served locally by [Pie](../../).

Open Historia is provider-agnostic: its browser AI layer already ships an
**"OpenAI Compatible"** provider (default endpoint `http://localhost:11434/v1`,
the catch-all for Ollama / vLLM / LM Studio). Pie, however, speaks its own
WebSocket + WASM-inferlet protocol, not HTTP. This adapter bridges the two:
it accepts OpenAI requests and translates each into a call to the
`openhands-completion` inferlet via `pie_client`.

> **Status: scaffold.** The buffered `/chat/completions` path and the
> OpenAI ⇄ inferlet translation are implemented and unit-tested. It has **not**
> yet been run end-to-end against a live Pie server + GPU. See *Roadmap*.

```
 Open Historia (browser)                    this adapter                 Pie
 ┌───────────────────────┐   OpenAI HTTP   ┌──────────────┐  pie_client  ┌────────┐
 │ callAI → /chat/compl. │ ──────────────▶ │ translate +  │ ───ws──────▶ │ server │
 │ tool_choice:"required"│ ◀────────────── │ launch inferl│ ◀─────────── │ +model │
 └───────────────────────┘   ChatCompletion└──────────────┘   Return     └────────┘
```

## Why an adapter (and not `PieLLM`)

The OpenHands integration's `PieLLM` hooks in at the Python
`LLM._transport_call` layer — it needs the OpenHands SDK in-process. Open
Historia's client is a **browser** doing `fetch(endpoint + "/chat/completions")`,
so the bridge has to be a real HTTP server. This package *is* that server; it
reuses the same inferlet (`openhands-completion`) and the same `pie_client`
round trip as `PieLLM._call_pie`.

## Install

```bash
cd pie/integrations/open-historia
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e ../../client/python    # pie_client
.venv/bin/pytest -q                             # translation unit tests, no GPU
```

## Run

1. **Build the inferlet** (once), from the repo root:
   ```bash
   # produces openhands-completion@0.1.0 in the local registry
   cargo run -p ... # see integrations/openhands/docs/RUNBOOK.md for the bakery step
   ```
2. **Start a Pie server** with a chat model loaded (Qwen3 / Qwen2.5 work well;
   Open Historia is natural-language-heavy, so use an instruct model, not a
   pure coder). See the Pie server docs.
3. **Start the adapter:**
   ```bash
   .venv/bin/pie-openhistoria --pie-uri ws://127.0.0.1:8080 --port 8000
   # or: python -m pie_openhistoria
   ```
4. **Point Open Historia at it.** In-game → Settings → AI provider →
   **OpenAI Compatible**, endpoint `http://localhost:8000/v1`, and set the model
   name to whatever you passed as `--model-id` (default `pie`). Leave the API
   key blank or put any placeholder.

### Options / env vars

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--pie-uri` | `PIE_URI` | `ws://127.0.0.1:8080` | Pie server WebSocket URI |
| `--pie-username` | `PIE_USERNAME` | `local-dev` | Pie auth username (`pie auth`) |
| `--inferlet` | `PIE_INFERLET` | `openhands-completion@0.1.0` | Completion inferlet name@version |
| `--model-id` | `PIE_MODEL_ID` | `pie` | Model id advertised on `/models` and echoed back |
| `--timeout` | `PIE_TIMEOUT_S` | `600` | Per-request timeout (s) |
| `--host` / `--port` | `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address |

## What Open Historia actually asks of this endpoint

Derived from `src/Game/AI/main.jsx` (`callOpenAIStyleChatCompletions`):

- **Forced single tool call**: `tool_choice: "required"` (string form) with one
  `tools[]` entry per structured task (jumps, catalysts, GM, stat sheets, …).
  The adapter forwards the tool; the inferlet returns `tool_calls`, mapped back
  to an OpenAI `tool_calls` message with `finish_reason: "tool_calls"`.
- **Fallback tolerance**: if a model can't force a tool, Open Historia degrades
  `tool → json_schema → json_object → text_json` and parses JSON out of prose
  (`extractJsonPayload`). So even imperfect tool-calling still works — the
  adapter just needs to return the model's text.
- **`max_tokens`** floored at 8192 for structured tasks; we honor whatever is
  sent (default 8192).
- **`GET /models`** for discovery when no model is set — returns `--model-id`.
- **Streaming** only for *local* endpoints, to make Cancel physical. See below.
- **CORS / relay**: the desktop build forwards through its own
  `/api/ai/relay`, so CORS isn't needed there; the adapter still sends wide-open
  CORS headers for the hosted-web → local-Pie case.

## Limitations / Roadmap

- **Streaming is buffered-then-replayed.** `stream: true` returns valid SSE, but
  the inferlet call completes first and the result is emitted as a short chunk
  sequence — so Cancel is *not* yet physical mid-generation. Real token
  streaming needs the inferlet to emit tokens on Stdout; then wire client
  disconnect → process terminate.
- **No KV pinning across turns.** Each request launches a fresh process (same as
  `openhands-completion`). Cross-turn prefix reuse is the `openhands-coder-session`
  protocol — a later phase, and the substrate for the **harness-in-inferlet**
  work (CoT/ToT/GoT scored against Open Historia's own validation rules — see
  `../../OPEN_HISTORIA_INTEGRATION_ASSESSMENT.md`).
- **Stale manifest.** `inferlets/openhands-completion/Pie.toml [parameters]`
  still documents the old Phase-1 `prompt` field; the live contract is
  `messages`/`tools` (see the doc-comment atop that inferlet's `src/lib.rs`),
  which is what this adapter sends.

## Layout

| File | Role |
|---|---|
| `pie_openhistoria/translate.py` | Pure OpenAI ⇄ inferlet mapping (unit-tested) |
| `pie_openhistoria/backend.py` | `pie_client` round trip (connect → launch → drain Return) |
| `pie_openhistoria/server.py` | aiohttp routes, CORS, buffered + SSE responses |
| `pie_openhistoria/__main__.py` | CLI / env config |
| `tests/test_translate.py` | Translation unit tests (no server, no GPU) |
