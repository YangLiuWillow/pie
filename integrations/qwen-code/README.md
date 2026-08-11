# qwen-code ↔ Pie (rewritten engine)

Serve stock [qwen-code](https://github.com/QwenLM/qwen-code) from Pie via an
OpenAI-compatible `/v1/chat/completions` endpoint. Design + port rationale:
`docs/qwen-code-dev-port.md` (successor to `docs/qwen-code-integration-plan.md`,
which documents the pre-rewrite build and its results: 33/33 acceptance, 99.7%
KV reuse on M2, and the H200 A/B against vLLM).

## Architecture

The rewritten engine removed in-guest HTTP serving, so the OpenAI surface is a
client-side shim fronting a long-lived inferlet:

```
qwen-code ──HTTP/SSE──► shim.py ──WS /v1/ws──► pie serve ──► chat-completions
            (OpenAI       (transport only)      (gateway)      inferlet
             wire)                                             (all OpenAI
                                                                semantics)
```

- `shim.py` — stdlib-only asyncio HTTP server. Owns one WebSocket session,
  installs + launches `chat-completions` (from `tests/inferlets/chat-completions/`),
  forwards each HTTP request as a `signal` and translates the inferlet's
  `{req_id, event, data}` messages into SSE frames. Owns keepalives (`: ping`)
  and HTTP status mapping. Relaunches the inferlet if it ever exits.
- The inferlet loops on `session::receive()`, renders the conversation
  (byte-parity with the old verified renderer), generates via PTIR, streams
  `chat.completion.chunk` objects, and reuses KV across turns through indexed
  working sets (`update_index`/`from_index` — see the port doc §4).

## Run

```bash
# 1. one-time: build server + inferlet, import the model
cargo build --release -p pie-bin --features driver-metal
(cd tests/inferlets && cargo build --target wasm32-wasip2 --release -p chat-completions)

# 2. boot everything (imports the model on first run)
./run_pie_qwen.sh                       # Metal, Qwen3-0.6B-4bit
./run_pie_qwen.sh pie_config_dummy.toml # dummy driver: transport tests only

# 3. acceptance
python3 test_acceptance.py --base http://127.0.0.1:8123
```

Then point qwen-code at the shim with the audited profile printed by
`run_pie_qwen.sh` (env + `.qwen/settings.json` per `docs/qwen-code-rl-audit.md` §6).

## Memory on Apple Silicon

Two knobs matter in `pie_config.toml`:

- `max_model_len` sizes the M=1 KV ring **and** (on the simple families) the
  paged pool; the default is the driver ceiling (~14.6 GiB on Qwen3-0.6B) and
  will not fit a laptop. 16384 ≈ 2.5 GiB total resident here.
- The driver also refuses to boot unless the *machine* has the memory free
  right now (resident + ~2 GiB margin ≤ host-reclaimable). On an 8 GB box,
  close the browser/IDE first or the load is (correctly) refused — the guard
  exists because overcommitting Metal wedges the GPU unrecoverably.

The dummy profile (`pie_config_dummy.toml`) sidesteps all of this for
transport-level work: real gateway, real inferlet, fake tokens.

## Hazards carried over from the old integration

- H3 synthetic continuation and H11 git-status remain qwen-code fork-patch
  items (out of scope); within-rollout KV reuse survives H11, cross-rollout
  does not.
- KV index retention is all-or-nothing: the first pool-pressure event wipes
  every entry not currently open (rebuild-on-miss is correct, just slower).
- Hybrid models (qwen3.5 GDN) have no recurrent-state index — reuse is
  attention-family only; the inferlet falls back to rebuild-every-turn.
