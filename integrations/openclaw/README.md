# pie-openclaw

Integration between [Pie](../../) and [OpenClaw](https://github.com/openclaw/openclaw).

Routes OpenClaw's LLM inference through a local Pie server, giving you
programmable inference (tool-call grammars, KV cache control, custom
samplers) for OpenClaw's personal assistant.

## Architecture

```
OpenClaw Gateway                         Pie Server
┌────────────────────┐                  ┌──────────────────┐
│  @pie-project/     │   WebSocket      │                  │
│    openclaw        │──(msgpack)───────▶  openclaw-chat   │
│                    │                  │  inferlet (WASM) │
│  StreamFn:         │◀──events────────│                  │
│   • text_delta     │                  │  • chat template │
│   • toolcall_end   │                  │  • tool grammars │
│   • done           │                  │  • KV cache      │
└────────────────────┘                  └──────────────────┘
```

## Components

| Component | Language | Description |
|---|---|---|
| `integrations/openclaw/` | TypeScript | OpenClaw extension — StreamFunction adapter |
| `inferlets/openclaw-chat/` | Rust/WASM | Chat completion inferlet with streaming |

## Setup

### 1. Build and install the inferlet

```bash
cd inferlets/openclaw-chat
pie build
pie install openclaw-chat
```

### 2. Start the Pie server

```bash
pie serve --model <your-model>
```

### 3. Configure OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json
{
  "providers": {
    "pie": {
      "pieUri": "ws://127.0.0.1:8080",
      "inferlet": "openclaw-chat"
    }
  }
}
```

### 4. Install the extension

```bash
cd integrations/openclaw
npm install
```

Then register in your OpenClaw setup:

```typescript
import { buildPieProvider } from '@pie-project/openclaw';

const { stream } = buildPieProvider({
    pieUri: 'ws://127.0.0.1:8080',
});
// Pass `stream` to OpenClaw's provider registry.
```

## Testing

Unit tests (no server needed):

```bash
cd integrations/openclaw
node --test tests/test_conversions.mjs
```

End-to-end smoke test (boots a real `pie serve` with dummy driver):

```bash
PIE_BIN=../../target/release/pie \
PIE_WASM=../../inferlets/openclaw-chat/target/wasm32-wasip2/release/openclaw_chat.wasm \
PIE_MANIFEST=../../inferlets/openclaw-chat/Pie.toml \
node --test tests/test_e2e_smoke.mjs
```

## Session persistence (KV cache pinning)

For multi-turn conversations, pass a `sessionId` to avoid replaying the
full message history on every turn. The inferlet saves the KV cache
under the session name after each generation; on follow-up turns it
opens the saved snapshot and only appends the new messages.

```typescript
const stream = createPieStream({ pieUri: 'ws://127.0.0.1:8080', inferlet: 'openclaw-chat' });

// Turn 1 — full history replay, saves KV cache.
for await (const event of stream(null, context1, { sessionId: 'conv-123' })) { ... }

// Turn 2 — resumes from saved KV, appends only new messages.
const { turnMessageCount } = stream.lastSession!;
for await (const event of stream(null, context2, {
    sessionId: 'conv-123',
    resumeFrom: turnMessageCount,
})) { ... }
```

If the snapshot is evicted or expired, the inferlet falls back to a full
replay transparently.

## Status

- [x] Package scaffold
- [x] openclaw-chat inferlet (streaming, tool calls)
- [x] TypeScript StreamFunction adapter
- [x] Setup and configuration
- [x] OpenClaw extension manifest (`extension.ts` + `definePluginEntry`)
- [x] End-to-end smoke tests
- [x] Unit tests
- [x] Multi-turn session persistence (KV cache pinning)
- [ ] ClawHub publishing
