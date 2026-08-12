# OpenClaw ↔ Pie integration — handover

**Written:** 2026-08-12, at the end of Phase A implementation, for continuing
on a different machine. Read order: this file → `openclaw-integration.md`
(strategy) → `openclaw-integration-plan.md` (task list) →
`openclaw-integration-progress.md` (what happened, newest first) →
`tests/inferlets/fixtures/openclaw/AUDIT.md` (the wire contract everything
else is derived from).

---

## 1. Where the work lives

| What | Repo | Branch | Remote |
|---|---|---|---|
| Pie side: audit, fixtures, gateway, serving crate, harness | `pie` | `liu/openclaw-integration` (forked from `liu/opencode-integration` @ `1053a7790`) | `fork` = `github.com/YangLiuWillow/pie` |
| OpenClaw side: the `pie` provider extension | `openclaw` | `liu/pie-provider` (forked from `main` @ `0790d9f`-era) | see §2 — needs a fork |

Local layout is up to you; the docs' `~/Desktop/Lin_startup/...` paths are
historical. Scripts here resolve paths relative to the repo, so they survive
being moved — only prose and the capture profile ever hardcoded a location.

**If you relocate a checkout that has git worktrees** (pie has seven), the
registrations store absolute paths and every worktree goes `prunable` until
repaired. Move the whole tree as one unit so the siblings stay siblings, then
from the main checkout run `git worktree repair <new-path-of-each-worktree>`.
Bare `git worktree repair` with no arguments does NOT fix the registrations —
it only repairs the `.git` files pointing back — so pass the paths explicitly.

**Dependency:** this branch sits on top of `liu/opencode-integration` and
shares its crates (`inferlets/openai-serving`, `inferlets/chat-completions`,
`gateway/src/ingress/openai.rs`, `integrations/opencode/parity/`). Rebase onto
that branch when it moves; never fork the shared crates.

### Commits on `liu/openclaw-integration` (oldest → newest)

```
a3bd79214 docs: openclaw integration spec, execution plan, progress log
c909787af fixtures: openclaw source-derived wire audit + capture recorder
28a4039c6 fixtures: real openclaw wire captures + audit capture-verification
08fdb4316 parity: all 5 openclaw fixtures token-exact; Phase 0 complete
9125634b2 serving: openclaw audit fixes — empty-delta keepalives, overflow body, fixture sweep
2973c106d gateway: keyed sticky affinity for OpenAI client sessions
c3ebbb837 acceptance: openclaw live suite + launcher + e2e profile
```

`liu/pie-provider` (openclaw) is one commit: `e35577861f4`.

---

## 2. Setting up the new machine

### Pie

```bash
git clone <your pie remote> pie && cd pie
git fetch fork && git checkout -b liu/openclaw-integration fork/liu/openclaw-integration
```

Build (both flags matter — this cost time to discover):

```bash
# The bin package is named pie-bin, not pie; `-p pie` resolves to nothing.
# Without a driver feature the worker build script panics claiming
# --features driver-cuda is Linux-only (it is the default fan-out, not you).
CARGO_TARGET_DIR=$PWD/target cargo build --release -p pie-bin --features pie-bin/driver-metal

# Serving inferlet (its own workspace root):
cd inferlets/chat-completions && CARGO_TARGET_DIR=<repo>/target \
  cargo build --target wasm32-wasip2 --release
```

If you keep multiple worktrees, point them all at one `CARGO_TARGET_DIR`
(~24 GB of artifacts; the old machine ran out of disk otherwise).

### OpenClaw

The repo's only remote is upstream `openclaw/openclaw`, which you cannot push
to. Fork it once, then:

```bash
gh repo fork openclaw/openclaw --remote --remote-name fork   # once
git push fork liu/pie-provider
```

Running the published CLI needs **Node ≥ 24** (`nvm install 24`); the repo's
own checkout has no `node_modules` — per its `AGENTS.md`, heavy proof (tests,
typecheck, build) is meant to run remotely, so those gates are unrun here
(§5).

### Python bits

```bash
python3 -m venv .venv && .venv/bin/pip install transformers huggingface_hub jinja2
```
`jinja2` is required by `apply_chat_template` and is easy to forget — the
parity harness fails with an `ImportError` without it. No torch needed
(tokenizer-only).

---

## 3. What is done

**Phase 0 — complete.**

- **oc-P0.1** `tests/inferlets/fixtures/openclaw/AUDIT.md`: the full wire
  contract from four source sweeps of OpenClaw, every claim `file:line`-cited,
  with the **D-1…D-15** table of divergences from opencode and the pie-side
  action list in §7.
- **oc-P0.2** `wire/req-002..006.json`: real captures from
  `npx openclaw@2026.7.1-2` (plain / tool-call / history-replay / two lean
  turns) plus `record_server.py` and the capture profile. AUDIT §8 adds the
  **S-1…S-6** version-skew table (npm client vs repo HEAD).
- **oc-P0.3** renderer parity: **all 5 fixtures token-exact** vs HF
  `apply_chat_template`, Qwen3-0.6B, using the opencode harness unmodified
  (`--fixtures-dir`). Sizing result: full-surface prompts render **≈24.2k
  tokens**, lean ≈8.7k.

**Phase A — implemented, not yet proven live.**

- **oc-PA.0** audit fixes: keepalive is now an **empty-delta chunk** injected
  by the gateway after 15 s of inferlet silence (mirroring the stream's chunk
  id) — SSE comments are dropped by OpenClaw's sanitizer and never reset its
  watchdogs; `error::context_overflow_body()` with the wording OpenClaw's
  overflow tables match; serving crate sweeps the openclaw fixtures (45 tests).
- **oc-PA.1** `Affinity::Keyed(u64)` in the gateway + key extraction in the
  OpenAI ingress (`x-session-affinity` → `x-session-id` → `session_id` →
  body `prompt_cache_key`). Gateway suite 43+1+6 green.
- **oc-PA.2** `extensions/pie/` in OpenClaw + `docs/providers/pie.md` + core
  touch-ups (overlay-id allowlist, lean-mode auto-enable, docs nav).
- **oc-PA.3** `integrations/openclaw/`: 29-test acceptance suite (imports the
  opencode suite's plumbing, swaps in OpenClaw fixtures and policy),
  `run_pie_openclaw.sh`, e2e `openclaw.json`.

---

## 4. Where I stopped — the live-run blocker

**Nothing has been exercised against a running engine yet.** The last thing
in flight was `integrations/openclaw/run_pie_openclaw.sh`, and it never got
past server boot.

**Metal cannot be admitted on the old machine.** `driver/metal/src/batch/
forward.cpp:855-900` refuses when `want + min(transient, 2 GiB) + 2 GiB
margin > host_reclaimable`. Observed:

| profile | needs | reclaimable |
|---|---|---|
| 32k ctx, 1024 pages, default row budget | 4.245 GiB | 2.501 GiB |
| 12k ctx, 384 pages, `PIE_METAL_ROW_BUDGET_MB=256` | 2.058 GiB | 2.398 GiB |
| 10k ctx, 320 pages, `PIE_METAL_ROW_BUDGET_MB=192` | 1.839 GiB | 2.486 GiB |

The flat 2 GiB margin means even the smallest profile needs ~3.8 GiB free on
an 8 GB machine. The driver's own message says a previously wedged run may be
holding pages and only a reboot clears it. **On the new machine: try the
default 32k profile first** (`./run_pie_openclaw.sh` with no argument); only
shrink if admission still refuses.

**Dummy-driver fallback** (no GPU, exercises every other path — ingress,
envelope, inferlet, SSE, keepalives, wire shapes). Doctor passes with this
config; the suite run itself was interrupted before finishing:

```toml
[server]
host = "127.0.0.1"
port = 8080

[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[driver]                      # NOT [model.driver] — the layout moved on dev
type = "dummy"
device = ["cpu"]
vocab_size = 151936           # flat keys; there is no [driver.options] table
arch_name = "qwen3"
```

Two config traps found the hard way: `[model.driver]`/`[model.scheduler]` are
now `[driver]`/`[runtime]`, and driver options are flat keys on `[driver]`,
not a nested `options` table. Without `vocab_size`/`arch_name` the dummy
driver tries to auto-discover them from a `config.json` inside the `.zt`
artifact and fails with `Not a directory (os error 20)`.

Run it as: `./run_pie_openclaw.sh /path/to/that.toml`

---

## 5. Verification gaps — be honest about these

Things that are **written but never executed**:

1. **The whole live acceptance suite** (29 tests). `--collect-only` passes and
   the helpers are exercised, but no assertion has met real server bytes.
   Expect first-run breakage; the opencode track's equivalent found a
   universal-500 bug on its first live run.
2. **The keepalive path specifically.** The 15 s timer, the chunk-identity
   mirroring, and `test_keepalives_are_empty_delta_chunks` are all unproven
   against a real slow prefill. Unit tests cover the JSON shape only.
3. **Keyed affinity end-to-end.** Unit-tested at both layers
   (extraction priority; `Keyed(k)` → dispatch key), but never observed
   routing two requests to the same worker — needs a two-worker deployment.
4. **Everything in the OpenClaw repo.** No `pnpm install` on that checkout, so
   the extension has not been compiled, `pnpm docs:check-config-examples` has
   not validated the JSON5 fences in `docs/providers/pie.md`, and no vitest
   ran. `compat.sendSessionAffinityHeaders` was verified to exist
   (`src/config/types.models.ts:46`); the rest of the doc's config block is
   unvalidated. Per OpenClaw's `AGENTS.md` these gates belong on CI/Testbox —
   opening a PR is the cheapest way to run them.
5. **`context_overflow_body()` is unwired.** The body and its veto-word test
   exist; nothing calls it, because the guest has no way to query context
   capacity (no WIT getter). Until then the guard is catalog `contextWindow`
   = engine `max_model_len`, which drives OpenClaw's own preflight.

---

## 6. Next steps, in order

1. **Get one live run green** (oc-PA.3). Dummy driver first — it proves the
   whole wire path without a GPU — then Metal if the new machine admits it.
   Fix what the suite finds; the AUDIT is the spec when behavior is disputed.
2. **Stock-OpenClaw e2e**: `./run_pie_openclaw.sh --serve-only`, then the
   `openclaw agent` command in `integrations/openclaw/README.md`. This is the
   first moment the two halves meet.
3. **A/B vs Ollama and llama.cpp** (OpenClaw's incumbent local backends):
   wall clock, prompt tokens, `cached_tokens`, trajectory match. Report
   honestly — the H200 lesson from the qwen-code work is that short contexts
   are not pie's regime.
4. **Open the OpenClaw PR** for `liu/pie-provider` to run its CI gates (§5.4).
5. **Then Phase B** — the actual reason for the project. `oc-PB.1` is the
   shared session dialect (design it against *both* harnesses' rewrite
   inventories before freezing v0); `oc-PB.2` the `openclaw-session` inferlet;
   `oc-PB.3` the `createStreamFn` WS transport. The plan doc has the details.
   Note the Strategy-B hazard the audit surfaced: OpenClaw **changes `tools[]`
   mid-session** (heartbeat turns add `heartbeat_respond`; memory-flush turns
   collapse to `read`+`write`), so the dialect needs a per-turn tools digest.

---

## 7. Sibling work on the same fork (checked 2026-08-12)

Everything below was verified present on `fork` = `github.com/YangLiuWillow/pie`,
so a fresh clone loses nothing. Recorded here because the working tree made
some of it look local-only:

- **OpenHands integration** — `openhands-integration` and
  `openhands-integration-updated` (105 files under `integrations/openhands/`,
  incl. `pie_openhands/{llm,editor_repair,__init__}.py`, the tests, and the
  benchmark harness). The working-tree `integrations/openhands/` directory is
  **not** a source of truth: its `.py` files are gone, leaving only
  `__pycache__` bytecode, `.pytest_cache`, packaging metadata, and a **930 MB
  `.venv`**. Recreate the venv from the branch checkout; never commit it.
- **RL rollout fixtures** — `liu/rl-completions` carries
  `tests/inferlets/fixtures/rl_completions/` and `rl_completions_1.7b/`
  (137 files, ~48 MB; the untracked working-tree copies were byte-identical).
  `liu/rl-rollout-05` is also pushed now. Both were local-only until this date.
- **Other integration branches**: `liu/opencode-integration` (this branch's
  base), `liu/codex-integration`, `npr-inferlet`, `liu/qwen-code-dev` (which
  now also carries the 11 previously-ignored design docs under `docs/`).

## 8. Facts worth carrying in your head

- **Keepalives must be empty-delta chunks, not `: ping` comments.** This is
  the single most load-bearing finding (D-1) and it changed the shared
  ingress for both clients.
- **Unknown `finish_reason` strings fail the whole turn** client-side (D-9).
  Only `stop|length|tool_calls|function_call|tool_call|content_filter|
  network_error|end|null` are safe.
- **Tool-call `arguments` must parse to a JSON object** — `"{}"` for no-arg
  tools — or OpenClaw discards every call in the turn.
- **`prompt_cache_key` is the only session signal** on the default wire
  (= `sessionId:boundaryCount`, requires `compat.supportsPromptCacheKey`).
  `compat.sendSessionAffinityHeaders` upgrades to real headers.
- **OpenClaw's prompts are big**: ~24.2k tokens full-surface, ~8.7k lean, with
  a 34-tool / 56 KB schema block. That is the tax Strategy B is meant to kill.
- The npm client and repo HEAD **disagree on the wire** (S-1…S-6): string vs
  parts user content, `strict: false` presence, tool surface, tool-call id
  normalization. Pie must tolerate both.
