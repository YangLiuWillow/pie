# Codex ↔ Pie integration

The [Codex CLI](https://github.com/openai/codex) talking to Pie as its model
backend, through the `codex-responses` inferlet — a WASM HTTP server that
speaks the OpenAI Responses API (`wire_api = "responses"`) and uses Pie's
programmability where a stock server can't:

- **Content-addressed KV-session reuse.** Codex re-sends the whole
  conversation every turn. The inferlet saves its post-generation context as
  an engine-side snapshot named by a hash of (instructions, tools, item
  prefix); the next turn resumes it via `Context::take` and pays prefill
  only for the new tool outputs. No session table, no client cursor — a
  hash miss just falls back to a full rebuild. Reuse is reported to Codex in
  `usage.input_tokens_details.cached_tokens`.
- **Native tool calling.** Tool schemas render through the model's chat
  template (`tools::equip_after_system_prefix`), history replays
  byte-identically (`assistant_with_tool_calls_prefix` / `answer_batch_prefix`),
  and `<tool_call>` blocks stream out as structured `function_call` items
  the moment they close.

## Run

```bash
# once: build server + inferlet
cargo build --release -p pie-server
(cd inferlets/codex-responses && cargo build --target wasm32-wasip2 --release)

# once: python env for the launcher
cd integrations/codex
python3 -m venv .venv && ./.venv/bin/pip install -e ../../client/python

# every session
bash run_pie_codex.sh    # pie serve :18080 + HTTP daemon :8123
```

## Codex config (`~/.codex/config.toml`)

```toml
model = "qwen3-0.6b"
model_provider = "pie"

[model_providers.pie]
name = "Pie"
base_url = "http://127.0.0.1:8123/v1"
wire_api = "responses"
```

No `env_key` — the daemon is unauthenticated on localhost. Don't name the
provider `OpenAI` or `azure` (both trigger special-casing in Codex).

Then:

```bash
codex exec --skip-git-repo-check "run `ls` and tell me what you see"
```

## Smoke tests without Codex

```bash
# non-streaming
curl -s http://127.0.0.1:8123/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":false,"input":"Say hello in five words."}'

# streaming + tools (watch for function_call items and cached_tokens)
curl -sN http://127.0.0.1:8123/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,
       "instructions":"You are a helpful agent.",
       "tools":[{"type":"function","name":"shell","description":"Run a shell command",
                 "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],
       "input":[{"type":"message","role":"user","content":"List the files in the current directory."}]}'
```

## Notes / limitations

- `/no_think` is appended to every user turn (deterministically, so KV
  hashes stay stable); think blocks and tool markup are stripped from
  streamed text by `filter.rs`.
- Non-`function` tools (`custom` grammar tools like `apply_patch`) are
  accepted but not rendered into the template; Codex's fallback model info
  doesn't send them for unknown model slugs.
- Snapshots leak one per abandoned conversation (the latest turn's). Fine
  for local dev; a TTL/eviction policy is future work.
- The daemon gets a fresh WASM instance per request; all cross-request
  state is in the engine (that's what makes the snapshot design necessary).
