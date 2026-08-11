# opencode ↔ Pie integration scaffolding (PA.3)

Everything needed to run stock [opencode](https://opencode.ai) against a
`pie serve` OpenAI surface, plus the live acceptance suite that pins the wire
contract. Companion docs: `docs/opencode-integration.md` (plan),
`docs/opencode-integration-progress.md` (log),
`tests/inferlets/fixtures/opencode/AUDIT.md` (the hazard table these tests
encode).

| file | what |
|---|---|
| `test_acceptance.py` | 25-test live acceptance suite (stdlib-only raw HTTP, incl. SSE) |
| `run_pie_opencode.sh` | boot `pie serve` → wait `/health` → run suite → shut down |
| `opencode.json` | stock-opencode provider profile pointing at `http://127.0.0.1:8080/v1` |
| `parity/` | renderer parity harness (P0.4, already green — token-exact) |

## Run the acceptance suite

One command (boots the server, tests, tears down):

```sh
integrations/opencode/run_pie_opencode.sh
```

Against an already-running server:

```sh
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/test_acceptance.py
python3 integrations/opencode/test_acceptance.py --collect-only   # list tests (no server)
python3 integrations/opencode/test_acceptance.py --only stream    # substring filter
```

Prereqs the script checks/handles: the release binary (`$PIE_BIN`, default
`../pie/target/release/pie` — the shared-target-dir build), the model
`Qwen--Qwen3-0.6B-optimized` in `$PIE_HOME/models`, and the
`chat-completions` inferlet in `$PIE_HOME/programs` (auto-refreshed from
`…/pie/target/wasm32-wasip2/release/chat_completions.wasm` when newer).

## Run stock opencode against pie (e2e)

```sh
integrations/opencode/run_pie_opencode.sh --serve-only   # terminal 1
cd integrations/opencode                                  # terminal 2 — picks up ./opencode.json
opencode run -m pie/qwen3-0.6b "read the file README.md in this directory and summarize it"
```

`opencode run` is the non-interactive path — one prompt, agent loop with the
default build-agent toolset (10 tools), exit. Expect TWO requests minimum per
session: the agent turn and a no-tools title side-call (AUDIT §1d). A
successful e2e = opencode prints a model answer (ideally after executing a
real `read` tool call) and exits cleanly; watch the serve log for the
tool-history follow-up request (`role:"tool"` in the body).

## What "green" looks like

- `test_acceptance.py` exits 0: **25 passed, 0 failed** — every hard
  (wire-shape) check. `[WARN]` lines are allowed: they are model-behavior
  findings on Qwen3-0.6B (e.g. "declined the forced tool call"), not
  contract violations.
- The four hazard classes from AUDIT.md all hold live: never-500 on malformed
  input (opencode retries 5xx forever), first tool_call delta carries
  `id`+`function.name` (AI SDK throws otherwise), usage chunk carries
  `prompt_tokens_details.cached_tokens`, `$schema`/2^53−1 schema noise
  tolerated.
- e2e: `opencode run` completes a tool-calling turn against the server with
  no client patches.

## Current blockers (2026-08-11)

1. **Machine RAM vs Metal admission.** `pie serve` on this machine needs
   ~3.2 GiB reclaimable for the Metal heap; last attempt had ~1.9 GiB and the
   driver refused admission. Mitigations: free memory, and/or lower
   `PIE_METAL_ROW_BUDGET_MB` (activation-row reservation, driver default
   1024 MB — see `driver/metal/src/context.cpp row_budget_bytes()`). Don't
   lower it so far that the ~7.5k-token opencode build-agent prompt gets
   refused (over-long prompts are refused, not chunked).
2. **Predicted first-run failure — gateway launches the inferlet by bare
   name.** `gateway/src/ingress/openai.rs` has
   `const CHAT_INFERLET: &str = "chat-completions"`, but the engine's
   `ProgramName::parse` (runtime/engine/src/inferlet/program.rs) requires
   `name@major.minor.patch` — a bare name is rejected, so every
   `/v1/chat/completions` turn would 500 with
   `Invalid program identifier 'chat-completions'`. The PA.2 integration
   tests didn't catch it because their stub worker never parses the name.
   Fix at live-run time: make the constant `chat-completions@0.1.0` (or teach
   launch to resolve bare names) and rebuild — the acceptance suite will
   surface it as universal 500s on the chat tests until then.
