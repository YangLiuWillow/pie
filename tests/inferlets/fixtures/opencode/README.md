# opencode wire-capture fixtures

Real request bodies from stock opencode (CLI `opencode-ai@1.18.16`) talking to a local
OpenAI-compatible recorder, plus the audit derived from them: see `AUDIT.md`.

- `record_server.py` — recorder + canned-SSE responder (stdlib Python, port 8123)
- `opencode.json` — the provider config used for the capture (custom `pie` provider on
  `@ai-sdk/openai-compatible`)
- `wire/req-NNN.json` — one file per captured request: `{seq, method, path, headers, body}`

## Re-running the capture

Nothing here touches your real `~/.config/opencode` — isolation is via `XDG_*` env vars
pointed at a scratch dir.

```bash
FIX=/Users/yangliu/Desktop/Lin_startup/pie-opencode/tests/inferlets/fixtures/opencode
SCRATCH=$(mktemp -d)
mkdir -p "$SCRATCH/proj" "$SCRATCH/xdg"
cp "$FIX/opencode.json" "$SCRATCH/proj/"
echo "hello from pie fixture" > "$SCRATCH/proj/hello.txt"

# 1) simple text turn ------------------------------------------------------
RECORD_MODE=text python3 "$FIX/record_server.py" &   # writes $FIX/wire/req-NNN.json
cd "$SCRATCH/proj"
XDG_CONFIG_HOME=$SCRATCH/xdg/config XDG_DATA_HOME=$SCRATCH/xdg/data \
XDG_STATE_HOME=$SCRATCH/xdg/state  XDG_CACHE_HOME=$SCRATCH/xdg/cache \
OPENCODE_DISABLE_AUTOUPDATE=1 \
npx -y opencode-ai@latest run "say hello"
kill %1

# 2) tool round-trip (captures the history-replay request) -----------------
RECORD_MODE=tool RECORD_TOOL_FILE="$SCRATCH/proj/hello.txt" \
  python3 "$FIX/record_server.py" &
XDG_CONFIG_HOME=$SCRATCH/xdg/config XDG_DATA_HOME=$SCRATCH/xdg/data \
XDG_STATE_HOME=$SCRATCH/xdg/state  XDG_CACHE_HOME=$SCRATCH/xdg/cache \
OPENCODE_DISABLE_AUTOUPDATE=1 \
npx -y opencode-ai@latest run "read the hello file"
kill %1
```

Recorder behavior:

- `RECORD_MODE=text` — every `/v1/chat/completions` gets a minimal streaming answer
  ("Hello.", finish `stop`, usage chunk, `[DONE]`), preceded by a `: keepalive` SSE
  comment (deliberate: proves the client tolerates comment lines).
- `RECORD_MODE=tool` — a request whose history contains no `role:"tool"` message gets a
  tool-call answer (a real tool picked from the request's own `tools` array, preferring
  `read` with `{"filePath": $RECORD_TOOL_FILE}`, finish `tool_calls`); the follow-up
  request — which carries the assistant `tool_calls` message and the `role:"tool"`
  result — gets the text answer. Statefulness is inferred from history, not a counter,
  so restarts are harmless.
- Sequence numbers continue from whatever is already in `wire/`; delete `wire/*.json`
  first for a clean numbering.

Expect two extra small requests (one per `opencode run` session): the title-generation
side-call, with no `tools` field. That's stock behavior, keep them in the corpus.
