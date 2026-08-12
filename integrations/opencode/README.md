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
`../pie/target/release/pie` — the shared-target-dir build), the model artifact
in `$PIE_HOME/models`, and the `chat-completions` inferlet in
`$PIE_HOME/programs` (auto-refreshed from
`…/pie/target/wasm32-wasip2/release/chat_completions.wasm` when newer).

### Which model

`PIE_MODEL` selects it; the keys match the model ids in `opencode.json`:

```sh
PIE_MODEL=qwen3.6-35b-a3b integrations/opencode/run_pie_opencode.sh   # the 35B run
integrations/opencode/run_pie_opencode.sh                             # default: 0.6B
```

| `PIE_MODEL` | artifact | notes |
|---|---|---|
| `qwen3-0.6b` | `Qwen--Qwen3-0.6B-optimized` | fast; model-behaviour tests only `[WARN]` |
| `qwen3.6-35b-a3b` | `mlx-community--Qwen3.6-35B-A3B-4bit` | ~22.6 GiB resident; 25/25 with no warnings |

Anything else is passed through as a raw artifact name (`pie model list`).

**The id in `opencode.json` is a label, not a selector.** pie serves whichever
model `pie serve` booted; the serving inferlet ignores the request's `model`
field. `opencode run -m pie/qwen3-0.6b` against a server booted with
`PIE_MODEL=qwen3.6-35b-a3b` talks to the 35B and says "qwen3-0.6b" while doing
it. Match them by hand, or read the boot line the script prints.

## Run stock opencode against pie (e2e)

```sh
integrations/opencode/run_pie_opencode.sh --serve-only   # terminal 1
cd integrations/opencode                                  # terminal 2 — picks up ./opencode.json
opencode run -m pie/qwen3.6-35b-a3b "read the file README.md in this directory and summarize it"
```

`opencode run` is the non-interactive path — one prompt, agent loop with the
default build-agent toolset (10 tools), exit. Expect TWO requests minimum per
session: the agent turn and a no-tools title side-call (AUDIT §1d). A
successful e2e = opencode prints a model answer (ideally after executing a
real `read` tool call) and exits cleanly; watch the serve log for the
tool-history follow-up request (`role:"tool"` in the body).

## What "green" looks like

- `test_acceptance.py` exits 0: **25 passed, 0 failed** — every hard
  (wire-shape) check. On `qwen3.6-35b-a3b` the 2026-08-12 run was **25 passed,
  0 failed, 0 warnings**. `[WARN]` lines are allowed and expected on the 0.6B:
  they are model-behaviour findings (e.g. "declined the forced tool call"), not
  contract violations — a capable model turns them into real assertions.
- The four hazard classes from AUDIT.md all hold live: never-500 on malformed
  input (opencode retries 5xx forever), first tool_call delta carries
  `id`+`function.name` (AI SDK throws otherwise), usage chunk carries
  `prompt_tokens_details.cached_tokens`, `$schema`/2^53−1 schema noise
  tolerated.
- e2e: `opencode run` completes a tool-calling turn against the server with
  no client patches.

## Status (2026-08-12): Phase A green, live

Both blockers this section used to list are gone, and the run is banked in
`results-Lius-MacBook-Pro.md`:

- **25/25 live** on `Qwen3-0.6B` *and* on `Qwen3.6-35B-A3B` (0 warnings on the
  35B).
- **Stock opencode 1.18.17, unmodified**, does multi-step agentic work on the
  35B: `Read notes.txt` → `Write summary.md`, correct file on disk.

The old blocker #2 (gateway launching the inferlet by a bare name, which
`ProgramName::parse` rejects) was fixed before the live run; the ingress test
now pins the `name@major.minor.patch` format. The old blocker #1 (Metal
admission) resolved into a sharper rule, which is worth keeping:

**Run one `pie serve` at a time, and stop it with `SIGTERM`.** The driver's
admission warning blames wired pages on abandoned GPU contexts "cleared only by
reboot" — but the identical warning appears when another `pie serve` is simply
holding its heap, which is what it was here. A clean `SIGTERM` to the other
server took wired from 24.17 GiB to 2.85 GiB with no reboot. Check for a second
`pie serve` before believing the leak reading. The hard-kill hazard is real, but
it comes from `kill -9` mid-fire, not from running the thing.

Known limitations, both recorded in the progress log:

- **Streaming leaks a reasoning preamble on Qwen3.6.** The stray `</think>` tag
  never reaches content, but the reasoning text before it does on the streamed
  path — a delta cannot be un-sent. Non-streaming is clean
  (`cut_leading_reasoning`). The fix is a lineage-aware open-block cue plus a
  filter that starts in think-mode.
- **`copy_kv` on Metal is gated on hybrid geometry** (`driver/metal/src/
  context.cpp:1978` tests `!facts_.has_linear_attn`), so KV snapshot resume —
  PA.1 milestone 2 — will work with the 35B and be refused on dense models.

