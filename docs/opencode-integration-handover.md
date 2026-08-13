# opencode ↔ Pie integration — handover

**For:** a fresh Claude Code session picking this up cold.
**Written:** 2026-08-12. **Rewritten same day** after the first live serving run
and a change of strategy — read §2 and §4 even if you have seen this file before.
**Branch:** `liu/opencode-integration` on `https://github.com/YangLiuWillow/pie.git`
(the `fork` remote), based on pie `dev` @ `58cb77936`.
**Working tree:** `~/Documents/Liszt_ai/pie-opencode` — a git worktree of
`~/Documents/Liszt_ai/pie`. (§3 rebuilds it if you are starting from nothing.)

Read this first, then `docs/opencode-integration.md` (the design), then
`docs/opencode-integration-progress.md` (the task-by-task log, newest first).
Those two are the durable record; this file is the bridge.

## 0. Orient yourself in one minute

Run this before anything else. It confirms you are on the right branch, up to
date, and that the machine reproduces the green test state.

```sh
cd ~/Documents/Liszt_ai/pie-opencode
export CARGO_TARGET_DIR=~/Documents/Liszt_ai/.cargo-target/pie-opencode  # see §3 — do NOT share this

git status -sb && git log --oneline -3          # expect: liu/opencode-integration, clean
cargo test -p pie-openai-serving                # expect 56 passed
cargo test -p pie-model                         # expect 16 + 3 + 4 + 23 passed
cargo test -p pie-model-qwen-3 --features chat  # expect 24 passed
cargo test -p pie-gateway                       # expect 40 + 1 + 6 passed
```

If those pass, the repo is healthy. **Strategy A is finished and frozen; the
next work is PB.1 (Strategy B) — see §4.** If they fail, something about the
environment differs from where this was written; fix that before building on it.

---

## 1. What this project is

Make **pie** (a programmable LLM serving system — SOSP'25 paper: inferlets,
explicit KV-cache control) the serving backend for **opencode** (an
open-source TypeScript coding agent).

Two strategies were planned; they are phased, not competing:

- **Strategy A (DONE, frozen green)** — serve stock opencode from an OpenAI-compatible
  `/v1/chat/completions` endpoint. Pie is a drop-in vLLM-style backend; KV reuse
  is content-addressed and best-effort. Zero opencode changes.
- **Strategy B (THE MAIN LINE, 2026-08-12)** — a long-lived `opencode-session` inferlet holding
  the conversation's KV working set, fed turn *deltas* over the sticky
  WebSocket. This is where pie's unique programmability pays: in-place context
  editing instead of client-side compaction+re-prefill, KV forking for
  subagents, early tool-call push, prefill-during-tool-execution. Design in
  `opencode-integration.md` §2 with a feature-by-feature table (B-1…B-6).

**Honest performance frame** (from the earlier qwen-code H200 A/B and the
paper): at short contexts, batch-1, pie loses to vLLM. The win regime is long
contexts, subagents, and multi-tenant load. Don't claim wins outside it.

---

## 2. State — Strategy A is DONE and LIVE; Strategy B is the main line

| id | task | status |
|---|---|---|
| P0.1 | Tool-history replay primitives | **done** |
| P0.2 | opencode wire audit + fixture capture | **done** |
| P0.3 | Shared `pie-openai-serving` crate | **done** |
| P0.4 | Renderer parity harness | **done — token-exact** |
| PA.1 | `chat-completions` inferlet | **m1 done, FROZEN**; m2 (KV sessions) **CANCELLED** — see §4 |
| PA.2 | Gateway OpenAI ingress | **done** |
| PA.3 | Acceptance suite + stock-opencode e2e | **DONE, LIVE** |
| PB.1 | `opencode-session` inferlet + AI SDK provider | **ACTIVE ← you are here** |
| PB.2 | Native `packages/llm` protocol in opencode V2 | optional |

**The line that used to be here — "a single real token has never been served"
— is obsolete.** It has now been served, a great many of them:

- acceptance **25/25, 0 warnings** on Qwen3-0.6B *and* Qwen3.6-35B-A3B
  (hybrid GDN MoE, MLX 4-bit, Metal);
- **stock opencode 1.18.17, unmodified**, on the 35B: `Read notes.txt` →
  `Write summary.md` — two tool calls, correct order, correct file on disk;
- single-tenant, warm: prefill **421 tok/s**, decode **90 tok/s**, TTFB
  0.003 s (the role chunk, *not* first token — first content is 12.4 s on a
  5.2k prompt), agentic task end-to-end **81.2 s**.

**The measurement that decided the strategy:** ~3 turns re-prefilling ~24k
tokens of history ≈ 57 s of that 81.2 s, while generating the ~200 tokens of
actual output ≈ 2 s. **~70% of an agentic task is re-prefill** of a history
that changed by a few hundred tokens. With the KV working set resumed across
turns: **~81 s → ~21 s**, same model, same hardware. Full write-up in
`integrations/opencode/results-Lius-MacBook-Pro.md`.

Live serving found five real defects, all fixed and all *below* the wire:
hybrid pass unsupported in the inferlet; tool schemas silently dropped for
every Qwen MoE/VL model; `</think>` leaking into content; reasoning served as
the answer; and a decode ring sized at exactly `channel_capacity()`. Two are
recorded and NOT fixed, both upstream: the device-geometry batch limit (§4) and
the missing RS index surface (§4).

---

## 3. Set up the new machine

```sh
# 1. The repo. Clone UPSTREAM as `origin`, add the fork as `fork` — that is the
#    remote layout every doc, script and commit note here assumes
#    (`git fetch origin dev` for upstream, `git push fork …` to publish).
git clone https://github.com/pie-project/pie.git ~/Documents/Liszt_ai/pie
cd ~/Documents/Liszt_ai/pie
git remote add fork https://github.com/YangLiuWillow/pie.git
git fetch fork liu/opencode-integration

# 2. One worktree per integration (pie-codex, pie-rl, …). `-b` is REQUIRED on a
#    fresh clone: the fetch above created only the remote-tracking ref
#    `fork/liu/opencode-integration`, and `worktree add` needs a LOCAL branch —
#    without it you get `fatal: invalid reference: liu/opencode-integration`.
git worktree add -b liu/opencode-integration ../pie-opencode fork/liu/opencode-integration
cd ../pie-opencode
```

### Build discipline — read this, it has already bitten twice

**Use a target dir private to THIS worktree:**

```sh
export CARGO_TARGET_DIR=~/Documents/Liszt_ai/.cargo-target/pie-opencode
```

**Do not share one target dir across the sibling worktrees.** The earlier advice
here was to share `~/Documents/Liszt_ai/pie/target` to save disk, and it is
wrong: `pie-openclaw` (and potentially other integration worktrees) contains a
crate with the **identical package name** `pie-openai-serving` but different
content, and sharing a target dir makes their build artifacts collide. The
symptom is baffling — in *this* worktree, `cargo test -p pie-openai-serving`
reports **39 passed / 6 failed** with failures like
`every_openclaw_fixture_parses_and_renders` (a test that does not exist in this
source tree) panicking on a missing fixture file. With a private target dir the
same command is **43/43**. If you ever see a test name you cannot find with
grep, suspect this first.

The opposite failure mode is also real: the old machine hit **ENOSPC** with
~24 GB of artifacts per checkout, which killed a build mid-run. Keep an eye on
`du -sh ~/Documents/Liszt_ai/.cargo-target/*` and `cargo clean` the worktrees
you are not using. Correctness first, disk second.

`run_pie_opencode.sh` honors `CARGO_TARGET_DIR` for both the `pie` binary and
the inferlet wasm, falling back to the crate-local and legacy shared paths.

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

## 4. Next step: PB.1 — the `opencode-session` inferlet

**The shape.** One long-lived inferlet per opencode session: launched once,
talked to via `send`/`receive` over the sticky WebSocket, holding one
`WorkingSet` (plus one `RsWorkingSet` on hybrid models) for its whole life.
The client sends turn **deltas**; the inferlet appends and generates. Design in
`opencode-integration.md` §2, which also carries the paper's own argument for
this shape.

### Three blockers dissolve because of that shape — do NOT re-solve them

| blocker under Strategy A | why B removes it |
|---|---|
| Metal refuses **2 device-geometry programs in one batch** (`driver/metal/src/context.cpp:997`) — N concurrent HTTP requests are N such programs, so N≥2 degrades every turn | a session inferlet is **one** program |
| a **fold cannot be published across processes** — `rs-working-set` has `fork` and no `update-index`/`from-index` | the fold never leaves its process |
| the control layer's FCFS policy *"terminating the most recently created inferlets"* kills the newest turn | N turns are N rows in one inferlet, not N tenants |

The middle one reframes the whole week: PA.1 m2 was blocked because a fold has
no cross-process identity, and **B never needs one**. That gap is a consequence
of the request-per-process contract, not a property of hybrid models.

> **Corrected 2026-08-12 by the qwen-code session, which has been running this
> shape since the start.** "The fold never leaves its process" removes the
> *persistence* half and **not** the pipeline-binding half. They hit
> `fork`-mints-a-new-sequence (`2^63` → `2^63+1`, rejected as a continuation)
> **inside a single process**, because their seal was a fire on a FRESH
> pipeline — `run_ahead` having closed the generation one. A long-lived
> inferlet does not fix that by itself.
>
> **The fix, and it is why sequential decode pays twice:** with a hand-written
> submission loop you control when the last fire goes out, so **the seal can be
> the final fire on the generation pipeline** rather than the first fire on a
> new one. That deletes the cross-pipeline binding problem outright instead of
> working around it. Design the session inferlet this way from the start.

### One blocker does NOT dissolve, and becomes load-bearing

`run_ahead` speculates past the stop token — the SDK says *"up to one window of
fires may still be in flight … their cells are simply never taken"* — but they
**execute, and therefore fold**. Measured driver-side as
`recurrent slot 0 is at position 165, this fire starts at 160`.

Under A this was harmless: the working set died with the request. **Under B the
state is reused every turn, so an over-advanced fold compounds across a session
and yields fluent, wrong output with nothing to catch it.**

**Use sequential decode from day one.** `engine.rs` carries the switch
(`SEQUENTIAL_DECODE`, defaulted off) and the cost is already measured at
**0.4%** — 91.0 → 90.6 tok/s single-stream on the 35B. Write it as a
hand-rolled `submit_frame` + take loop with ONE `close()` at the end:
`run_ahead` calls `on.close()` when its budget is spent, so a loop of repeated
`run_ahead(.., 1, ..)` submits into a closed pipeline and **hangs** rather than
erroring (a 300 s timeout to discover).

### Keep Strategy A alive

It is the compatibility path for anything that only speaks OpenAI, **and it is
the control for the `~81 s → ~21 s` claim.** Without the baseline there is no
A/B, only an assertion. Freeze it; do not delete it.

### There is a working reference for this shape — read it first

**The qwen-code port (`~/Documents/Liszt_ai/pie-qwen`, `liu/qwen-code-dev`) has
been Strategy B from the start**, not by choice: the `dev` rewrite removed
in-guest HTTP, so their OpenAI surface moved client-side and the inferlet had to
become long-lived. Their shape is what we are about to build:

- one long-lived `Daemon` over a `session::receive()` loop;
- a client-side shim multiplexing HTTP onto it by `req_id`;
- `sessions: HashMap<String, (SessionState, u32)>` held **in-process across
  turns**.

Read it before designing ours. They have already paid for several lessons in
it, including the seal/pipeline one above.

### Reusable from Phase A

`pie_openai_serving::session` (canon, `snapshot_address`,
`split_resume_point`) still decides what a "delta" *is*, even with nothing
being hashed into an index. The renderer, filters and salvage are all
model-side and dialect-agnostic.

### Running Strategy A (still works, for the baseline)

```sh
PIE_MODEL=qwen3.6-35b-a3b integrations/opencode/run_pie_opencode.sh
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/test_acceptance.py
```

`PIE_MODEL` accepts `qwen3-0.6b`, `qwen3.6-35b-a3b`, or a raw artifact name.

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

9. **The chat registry sees the driver's arch STEM, not the HF model type.**
   `architectures[0]` lowercased with the task suffix stripped, so
   `Qwen3_5MoeForConditionalGeneration` → `qwen3_5moe`, NOT `qwen3_5_moe`. A
   miss lands on the `_` arm with `has_tools:false` and **every tool schema is
   dropped silently** — chat renders, the model answers fluently, tool calling
   is simply gone. `model/src/instruct.rs` now carries both spellings with a
   test. Cheap check: same request with and without `tools`, compare
   `usage.prompt_tokens`; identical means dropped.
10. **Never fall back to raw generation for content.** Stripping `<think>` tags
   and keeping the body serves the model's private reasoning as its answer —
   fluent, on-topic, wrong in kind, invisible to every test. Use
   `answer_after_reasoning`.
11. **`$PIE_HOME/programs/<name>/<version>.wasm` is a GLOBAL path.** Every
   worktree building `chat-completions@0.1.0` overwrites it, silently, and the
   loser's server runs the winner's code. Use a private `PIE_HOME` (models and
   `py-runtime` symlinked back) for anything you intend to measure.
12. **Install a wasm atomically** — temp name, check the `\0asm` magic, `mv`.
   A torn or foreign wasm does **not** error: it hangs. No launch ack, nothing
   logged, `/health` still answering 200, every completion blocked forever.
13. **Size the decode ring ABOVE `channel_capacity()`**, not at it
   (`+ 7 * live_slots()`). At exactly `cap` the engine's ticket check silently
   skips continuations — upstream measured 12% of frames lost with no error.

---

## 6b. Operating this machine

- **One `pie serve` at a time**, stopped with **SIGTERM**. `kill -9` mid-fire is
  what actually leaks a wedged Metal context. Verify with `pgrep -fl` *after*
  killing — `pkill` exits 1 silently when its pattern matches nothing, so
  "I cleaned up" and "my pattern is broken" look identical.
- The driver's *"wired pages … only cleared by reboot"* warning is
  **ambiguous**: it reads identically when another `pie serve` is simply
  holding its heap. Check for a second server before believing the leak
  reading. (24.17 GiB of "leaked" memory turned out to be a live peer process;
  a SIGTERM took it to 2.85 GiB.)
- The 35B wants ~22.6 GiB at `total_pages 512 / max_forward_requests 8 /
  max_model_len 16384`. **32768 is NOT a ceiling** — it booted fine on a quiet
  machine. Admission is `want + min(transient,2GiB) + 2GiB > reclaimable`, so
  it tracks what else is resident, not the model.
- **The first request after a boot pays wasm JIT** and reads as a hang. Warm
  with a throwaway 4-token request before timing anything.
- **Backgrounding `run_pie_opencode.sh --serve-only` kills the server** when the
  wrapper shell is reaped (its EXIT trap). Launch `pie` directly with
  `nohup … & disown`.
- **CUDA**: `ptxas` needs **≥12.8**, cuBLASLt needs **≥12.9**
  (`driver/cuda/src/ops/gemm.cpp:1886`). Build on 12.9+; installing
  `cuda-toolkit-12-9` over a 12.8 image preserves the compiled Rust crates if
  you delete only `target/release/build/pie-worker-*/out/cuda`. RunPod H100
  PCIe ≈ $2.89/h — **terminate the moment results are banked.**

---

## 7. Working agreements from the user

- **Commit as we go**, at task boundaries, with a progress-log entry per task.
- **Never commit to pie's `dev` or `main`** — always a `liu/<topic>` branch.
- `docs/` is gitignored in this repo; design docs are added with `git add -f`
  (the convention the earlier integration branches used).
- Write a progress-log entry for completed work *and* for dead ends — the CUDA
  and RAM blockers are logged precisely so the next attempt is cheap.

---

## 8. The order of work from here

1. **Orient** — §0. Right branch, four suites green.
2. **PB.1** — §4. Build the `opencode-session` inferlet with **sequential
   decode from the start**, and the AI SDK provider package on the client side.
3. **A/B it** against frozen Strategy A on the same model and machine. The
   claim to test is `~81 s → ~21 s`; A is the control, which is why it stays.
   **The two arms need identical PROMPTS as well as identical models**, or the
   measurement is of the renderer rather than the servers (qwen-code's run-2
   failure mode). `parity/capture_vllm_render.py` on `liu/qwen-code-dev`
   captures vLLM's `/v1/chat/completions/render` token ids to disk, so the two
   ~20 GB stacks never need to be co-resident and neither side re-derives the
   reference.
4. Then PB.2 (native `packages/llm`) if the numbers justify the deeper change.

### The lesson that cost the most time, four separate times

**The answer was already in the output.** `ttfb == max_gap` was CPython's
buffering, not the server. "8/8 completed" sat on the same line as 13 completion
tokens. An 824 KB wasm was another branch's build, not a torn copy. And the
concurrency root cause — `2 device-geometry programs in one batch` — was in
**every** log collected all session, unseen because every grep was for a string
already expected (`pie_metal_launch failed`, `pie::inferlet`) rather than for
the driver's own output. It cost two rounds of source-diving, a rebuilt binary
with two diagnostics that found nothing, and a rented H100.

**Read the whole log before theorising about it.**

### Correct the durable record, not just the conversation

Three claims made this session were wrong — "32768 is refused", "no operator
knob admits an oversized model" (`[model].expert_slab_bytes` does), and an
8-way concurrency row reported as "8/8 completed". All three are corrected in
place, with the reasoning, where someone would actually read them. The pattern
in each was the same: **an observation made once, under conditions that were
not controlled, written down as a property of the system.**

---

## 9. Where things live (quick map)

| Path | What |
|---|---|
| `docs/opencode-integration.md` | The design: both strategies, the B-1…B-6 table, phasing, measurement |
| `docs/opencode-integration-progress.md` | Task-by-task log, newest first — read the top three entries |
| `gateway/src/ingress/openai.rs` | HTTP surface + **the authoritative envelope contract** (module docs) |
| `gateway/tests/openai_ingress.rs` | 6 ingress tests against a stub envelope worker |
| `inferlets/chat-completions/` | The serving guest wasm (`lib.rs` envelope, `engine.rs` generation, `turn.rs` state machine) |
| `inferlets/openai-serving/` | All engine-free logic + 43 native tests — **put new logic here** |
| `model/qwen_3/src/chat.rs` | The Qwen chat template (replay primitives, no-think cue) |
| `integrations/opencode/` | Acceptance suite, launch script, opencode profile, parity harness |
| `tests/inferlets/fixtures/opencode/` | 5 real captured opencode requests + `AUDIT.md` (the hazard table) |

## 10. Other sessions on this machine

Two peer Claude sessions share this codebase and hardware. Both caught real
bugs in our code today; both are worth talking to.

- **qwen-code** — `~/Documents/Liszt_ai/pie-qwen`, `liu/qwen-code-dev`.
  **Already a Strategy B implementation** (§4) — read it before designing ours.
  Two artifacts offered to us at `1e99c2567`:
  - **lineage-aware open-block cue** — `render_text::{lineage_opens_think,
    THINK_OPEN}` + `VisibleFilter::starting(bool)`, with three filter tests
    pinning untagged reasoning, truncation-with-no-closer, and an unchanged
    text-mode start. Verified live at both temperatures: a truncated turn
    yields `'…'` rather than leaked reasoning. Strictly better than our
    closed-empty-block cue.
    **Caveat from them, and it is ours to resolve:** verified on SINGLE turns
    only. Under a session-long working set the interesting question is whether
    think-channel hygiene holds when the RETAINED state already contains
    earlier turns' reasoning. Nobody has tested that shape; we will reach it
    first.
  - **Qwen3.6 dialect** — `Dialect::Qwen36` in `render_text.rs`, golden-tested
    byte-for-byte against the real `chat_template.jinja`
    (`tests/qwen36_template_golden.json`, regenerable with transformers). For
    prompt-parity work, not for replacing our WIT template path.
- **openclaw** — `liu/openclaw-integration`, forks this branch. Implemented KV
  snapshot sessions in the SHARED `chat-completions` inferlet, and added
  `test_no_placeholder_content` and
  `test_length_finish_actually_reached_the_budget` to the shared base suite.

Reach them via `SendMessage` (find addresses with `ListAgents`). Coordinate
before touching `inferlets/chat-completions`, `integrations/opencode/test_acceptance.py`
or `gateway/src/ingress/openai.rs` — all three are shared, and we have already
collided on two of them.

---

