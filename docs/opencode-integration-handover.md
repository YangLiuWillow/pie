# opencode ↔ Pie integration — handover

**For:** a fresh Claude Code session, on this machine or a new one.
**Written:** 2026-08-12, at the end of the first working stretch.
**Branch:** `liu/opencode-integration` on `https://github.com/YangLiuWillow/pie.git`
(the `fork` remote), based on pie `dev` @ `58cb77936`.

**Working tree** (if it came across with you): `~/Documents/Liszt_ai/pie-opencode`,
a git worktree of `~/Documents/Liszt_ai/pie`. The project tree moved from
`~/Desktop/Lin_startup` to `~/Documents/Liszt_ai` on 2026-08-12; git worktree
links survived the move, and every path in this repo's docs and scripts was
rewritten to match. If you are starting from nothing, §3 rebuilds it from the
remote.

Read this first, then `docs/opencode-integration.md` (the design), then
`docs/opencode-integration-progress.md` (the task-by-task log, newest first).
Those two are the durable record; this file is just the bridge.

---

## 1. What this project is

Make **pie** (a programmable LLM serving system — SOSP'25 paper: inferlets,
explicit KV-cache control) the serving backend for **opencode** (an
open-source TypeScript coding agent).

Two strategies were planned; they are phased, not competing:

- **Strategy A (in progress)** — serve stock opencode from an OpenAI-compatible
  `/v1/chat/completions` endpoint. Pie is a drop-in vLLM-style backend; KV reuse
  is content-addressed and best-effort. Zero opencode changes.
- **Strategy B (not started)** — a long-lived `opencode-session` inferlet holding
  the conversation's KV working set, fed turn *deltas* over the sticky
  WebSocket. This is where pie's unique programmability pays: in-place context
  editing instead of client-side compaction+re-prefill, KV forking for
  subagents, early tool-call push, prefill-during-tool-execution. Design in
  `opencode-integration.md` §2 with a feature-by-feature table (B-1…B-6).

**Honest performance frame** (from the earlier qwen-code H200 A/B and the
paper): at short contexts, batch-1, pie loses to vLLM. The win regime is long
contexts, subagents, and multi-tenant load. Don't claim wins outside it.

---

## 2. State at handover

| id | task | status |
|---|---|---|
| P0.1 | Tool-history replay primitives (Instruct + WIT + host + SDK) | **done** |
| P0.2 | opencode wire audit + fixture capture | **done** |
| P0.3 | Shared `pie-openai-serving` crate | **done** |
| P0.4 | Renderer parity harness | **done — token-exact** |
| PA.1 | `chat-completions` inferlet | **milestone 1 done** (sessions/grammar/coder-dialect deferred) |
| PA.2 | Gateway OpenAI ingress | **done** |
| PA.3 | Acceptance suite + stock-opencode e2e | suite **authored**, **never run live** ← *you are here* |
| PB.1 | `opencode-session` inferlet + AI SDK provider | not started |
| PB.2 | Native `packages/llm` protocol in opencode V2 | not started (optional) |

**Everything is committed and pushed. Nothing was lost in the migration** —
verified by a fresh clone of the remote after the old working tree was deleted.

Test status (all green at handover, native/unit level):
`pie-openai-serving` 43/43 · `pie-model-qwen-3 --features chat` 24/24 ·
`pie-gateway` 40+1+6 · renderer parity **5/5 fixtures token-exact** vs HF ·
inferlet builds clean for `wasm32-wasip2` (606 KB; 35 s on Linux).

**The one thing that has never happened: a single real token served
end-to-end.** Everything is verified against unit tests, stub workers, and the
HF reference tokenizer — not against a running engine. Treat every "done" above
as "done pending first contact."

---

## 3. Set up the new machine

```sh
# 1. The repo. Upstream is pie-project/pie; the work lives on the fork.
git clone https://github.com/YangLiuWillow/pie.git ~/Documents/Liszt_ai/pie
cd ~/Documents/Liszt_ai/pie
git remote add fork https://github.com/YangLiuWillow/pie.git   # if `origin` is upstream
git fetch fork liu/opencode-integration

# 2. The convention here is one worktree per integration (pie-codex, pie-rl, …):
git worktree add ../pie-opencode liu/opencode-integration
cd ../pie-opencode
```

**Build discipline (matters):** the old machine ran out of disk with 24 GB of
cargo artifacts per checkout. Point worktree builds at one shared target dir:

```sh
export CARGO_TARGET_DIR=~/Documents/Liszt_ai/pie/target
```

Toolchain: `rust-toolchain.toml` pins **1.97.1** and declares the
`wasm32-wasip2` target. Materialize it once (`cargo --version` inside the repo)
**before** launching parallel builds — concurrent first use races rustup's
component install.

### Things that do NOT come across in git, and must be re-created

| What | Where it was | How to restore |
|---|---|---|
| Model weights | `~/.pie/models/*.zt` | `pie model` (see `pie model --help`). Metal needs an **MLX 4-bit** checkpoint (`mlx-community/Qwen3-0.6B-4bit` → `Qwen--Qwen3-0.6B-optimized`); CUDA takes raw bf16 `Qwen/Qwen3-0.6B`. |
| Inferlet program cache | `~/.pie/programs/chat-completions/0.1.0.{wasm,toml}` | Build the inferlet, copy the wasm + `Pie.toml` there. `run_pie_opencode.sh` does this automatically. |
| `~/.pie/config.toml` | local | `pie config init`; `run_pie_opencode.sh` generates a trimmed one if you pass no config arg. |
| HF tokenizer cache + venv | scratchpad | `check_render.py` re-downloads tokenizer files only (never weights); `pip install transformers huggingface_hub`. |
| **RunPod API key** | `Liszt_ai/pie-rl/.env` (`RUNPOD_API_KEY`) | **Secret — bring it across by hand.** Also `HF_TOKEN`, `GH_TOKEN` there. |
| Local memory files | `~/.claude/projects/…/memory/` | Their content is folded into this doc (§6, §7). |

---

## 4. Next step: PA.3, the live run

This is the whole remaining Phase A. Two paths; pick by what hardware the new
machine has.

### Path A — local Metal (Apple Silicon)

```sh
cd ~/Documents/Liszt_ai/pie-opencode
CARGO_TARGET_DIR=~/Documents/Liszt_ai/pie/target \
  cargo build -p pie-bin --release --features driver-metal   # ← the feature is REQUIRED
bash integrations/opencode/run_pie_opencode.sh               # boots, waits /health, runs the suite
```

**Known blocker on the old machine:** the Metal driver refuses admission unless
`needed (~1.2 GiB) + a flat 2 GiB host margin < reclaimable RAM` — it had only
~1.9 GiB free. That margin is deliberate (over-admitting on unified memory
produces an *unkillable* wedged process; only a reboot recovers), so **do not
try to bypass it** — free RAM instead (~1.5 GB was needed). Reboot before the
run if in doubt. `PIE_METAL_ROW_BUDGET_MB` shrinks the activation-row
reservation but does **not** move the flat margin.

### Path B — GPU box / RunPod

**Use a CUDA ≥ 12.8 devel image.** The stock `cuda-12.4.1-devel` image cannot
build pie's sm_90 kernels at all: its `ptxas` rejects the Hopper TMA bulk-copy
instructions the vendored XQA attention kernels emit
(`error : State space incorrect for instruction 'cp.async.bulk.tensor'`),
failing ~28 min into the build. It's a compiler limit, not a driver one. Full
recipe + the image's other potholes (no `nvcc` on `PATH`, no apt cmake, stale
NVIDIA apt mirror) are in the progress log's 2026-08-11 GPU entry.

RunPod, if used again (GraphQL at `https://api.runpod.io/graphql?api_key=…`,
mutation `podFindAndDeployOnDemand`, terminate with `podTerminate`): SSH via the
pod's **direct TCP mapping** (`root@<ip> -p <port>`), not the `ssh.runpod.io`
proxy, which failed PTY allocation. **Pods bill hourly — terminate them the
moment the results are banked.** Three were left running at $7.92/h before the
user caught it; all are now terminated.

### Then

```sh
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/test_acceptance.py
```

25 tests, stdlib-only. Wire-shape assertions are hard; 0.6B model-behavior ones
are soft `[WARN]`. `--collect-only` lists them, `--only <substr>` filters, exit
2 = server unreachable. Then the stock-opencode e2e per
`integrations/opencode/README.md` (`opencode run -m pie/qwen3-0.6b …` with the
committed `opencode.json`).

**Expect first-contact bugs.** Nothing below the HTTP layer has met a real
engine. Bank results to `integrations/opencode/results-<host>.md` and add a
progress-log entry either way — a failure list is a deliverable.

---

## 5. The code, and why each piece is shaped that way

Request path: **opencode → `POST /v1/chat/completions` (gateway ingress) →
`LaunchProcess` → `chat-completions` inferlet → engine → SSE back.**

- **`gateway/src/ingress/openai.rs`** — the HTTP surface (`/v1/chat/completions`,
  `/v1/models`, `/health`). Deliberately thin: it maps `Authorization: Bearer` →
  the trust-edge `Identity`, hands the request body **verbatim** to the
  inferlet, and re-frames the reply. All OpenAI wire logic lives in the
  inferlet. The **gateway⇄inferlet envelope contract is documented in that
  file's module docs and is authoritative**: first message `{"status": <u16>}`,
  then ready-made chunk JSON (one `data:` line each) or a single unary body;
  gateway appends `[DONE]`; launch acks and stdout/stderr never reach the wire.
- **`inferlets/chat-completions/`** — the guest wasm (own workspace root, like
  `tests/inferlets/*`). `lib.rs` envelope + 400-vs-500 discipline, `engine.rs`
  generation core (PTIR chunked prefill + device-carried decode, ported from
  `tests/inferlets/chat-completion`), `turn.rs` per-token state machine (tool
  decode, fencing, stops, salvage).
- **`inferlets/openai-serving/`** — pure logic, native tests, no WIT: wire types,
  chunk builders, `plan_render` → `RenderOp`, session canon (FNV-1a-64 ×2,
  resume-point splitting), `VisibleFilter`, salvage parsers, error bodies.
  **Put logic here, not in the inferlet** — it's the only part that's testable
  without an engine.
- **`model/qwen_3/src/chat.rs` + `model/common/src/instruct.rs` +
  `interface/inferlet/{chat,tools}.wit`** — the template layer. P0.1 restored
  `equip_after_system`, `assistant_with_tool_calls`, `answer_batch`; P0.4 added
  `cue_no_think`. After any WIT edit run `scripts/sync-wit.sh` (vendored copies
  must match; `--check` compares against *committed* state, so it fails on
  uncommitted edits — that's expected).
- **`integrations/opencode/`** — acceptance suite, launch script, opencode
  profile, and `parity/` (the renderer harness).
- **`tests/inferlets/fixtures/opencode/`** — five real captured opencode
  requests + `AUDIT.md`, the hazard table that the acceptance suite encodes.

---

## 6. Hard-won invariants — violate these and things break silently

1. **Never 500 on bad input.** opencode retries 5xx *without bound*. Malformed
   requests get 400 + an OpenAI error body. (500 is reserved for real faults.)
2. **The first tool-call delta per index must carry both `id` and
   `function.name`**, or opencode's SDK throws.
3. **Response content and snapshot-address content must be computed from the
   same string.** Any divergence (trimming, fallbacks) breaks every subsequent
   KV resume, because opencode echoes content back verbatim. This killed reuse
   silently in an earlier integration.
4. **Renderer parity is token-level, and it is a test, not a belief.** Two fixes
   came from it: tool schemas must serialize with Python `json.dumps` separators
   (`", "`/`": "`) because HF's Jinja `tojson` does (this string feeds the
   snapshot address too — change both together), and each replayed turn's inner
   text must be encoded in **one pass**, not as pre-tokenized fragments, so BPE
   merges the way HF does. Re-run `parity/check_render.py` after any template or
   envelope change.
5. **Inferlet ids must be `name@major.minor.patch`.** `ProgramName::parse`
   rejects bare names; a bare id turns every request into a 500. The ingress
   test now pins the format.
6. **`bind()` binds but does not serve.** Call `into_handle()`/`serve()`, or a
   client hangs forever against a listening-but-unserved socket. (~40 min lost.)
7. **SSE `: ping` comments are safe keepalives** — opencode's watchdog resets on
   raw bytes, upstream of the SSE parser. Verified from its source + captures.
8. **`cargo build -p pie` builds the runtime lib; the server binary is
   `-p pie-bin`**, and it needs an explicit `--features driver-metal|driver-cuda`
   or it builds with no GPU driver and fails at boot. `-c <config>` is a
   **global** flag: `pie -c cfg.toml serve`, not `pie serve -c cfg.toml`.

---

## 7. Working agreements from the user

- **Commit as we go**, at task boundaries, with a progress-log entry per task.
- **Never commit to pie's `dev` or `main`** — always a `liu/<topic>` branch.
- `docs/` is gitignored in this repo; design docs are added with `git add -f`
  (the convention the earlier integration branches used).
- Write a progress-log entry for completed work *and* for dead ends — the CUDA
  and RAM blockers are logged precisely so the next attempt is cheap.

---

## 8. If you're picking this up cold — the first three moves

1. `git clone` + worktree per §3; set `CARGO_TARGET_DIR`; run the test suites
   listed in §2 to confirm the machine reproduces green.
2. Restore model weights + build the binary with the right driver feature (§4),
   then run `integrations/opencode/run_pie_opencode.sh`. **Fix what first
   contact breaks** — that is the actual work now, and it is expected to be
   non-trivial.
3. Only once the suite and the opencode e2e are green, start **PA.1 milestone 2**
   (KV snapshot sessions — the seams are marked in-code: attach at
   `build_prompt` + a pre-finish save, then report real `cached_tokens`) and
   then **Strategy B**, which is where the interesting research claim lives.
