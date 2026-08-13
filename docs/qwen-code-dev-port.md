# qwen-code ↔ Pie: port to the rewritten engine (`dev`)

**Date:** 2026-08-11. Successor to `qwen-code-integration-plan.md`, whose Option-A build
(commits `a2b0e2243`..`f380e9904` on `openhands-integration-updated`) ran against the
pre-rewrite engine. The rewrite (`dev`, now branched as `liu/qwen-code-dev`) removes both
hosts that build stood on: the in-guest HTTP daemon (`runtime/src/daemon.rs`,
`#[wstd::http_server]`) and the named-Context snapshot API. This doc maps every old
component to its new home. The wire contract (audit §1), the fixtures, and the acceptance
suite carry over unchanged — the OpenAI-facing surface is identical by design.

## 1. Shape: external shim + long-lived session inferlet

The new WIT world (`interface/inferlet/world.wit`) imports only `wasi:http/client`
(outbound); the incoming-server side was deliberately excluded. Client interaction is a
gateway WebSocket (`/v1/ws`, msgpack frames) carrying process events, and an inferlet is
one `run(input) -> result<string>` call that may loop forever on
`session::receive().await`.

So the integration becomes two pieces:

```
qwen-code ──HTTP/SSE──> shim.py (owns /v1/chat/completions)
                          │  one WS session, x-pie-identity header
                          ▼
                    pie gateway (pie serve)
                          │  signal_process / ProcessEvent::Message
                          ▼
              chat-completions inferlet (run = receive/send loop)
```

- **`integrations/qwen-code/shim.py`** — asyncio HTTP server + `pie_client.PieClient`.
  On boot: connect (identity header), authenticate, `install_program`,
  `launch_process("chat-completions", …)`, hold the `Process`. Per HTTP request:
  `proc.signal(json{req_id, body})`; drain `proc.recv()` events, demux on `req_id`,
  wrap each payload as one SSE `data:` line. The shim owns SSE keepalives (`: ping`)
  and HTTP error mapping (400 with OpenAI error JSON; never 500 for client faults —
  audit §1 rows 6/9). It is pure transport: all OpenAI semantics stay in the inferlet.
- **`tests/inferlets/chat-completions/`** — the ported inferlet. `run` loops:
  `receive()` → parse request → render → resume-or-prefill → decode loop → stream
  chunk JSONs (tagged `req_id`) via `session::send` → publish KV index → next request.
  One request at a time; the shim serializes (qwen-code is sequential anyway).

Lifecycle traps (from the gateway map): the client's event queue is deleted on
`return`/`error`, so if `run` ever returns the shim must **relaunch**, not reattach;
`launch_process` is the one long-lived gateway turn and each `signal` is a short
side-turn, all sticky to one worker.

## 2. Old inferlet → new SDK, file by file

Old sources: `git show f380e9904:inferlets/chat-completions/src/…` (2,060 lines).

| Old | Fate |
|---|---|
| `types.rs` (wire types, unknown-field tolerant) | port unchanged |
| `filter.rs` (hermes + bare Coder-XML salvage parsers) | port unchanged (pure text; the engine's `tools::Decoder` is hermes-only, Coder-XML still has no engine decoder) |
| `render.rs` | port with the same *strings*, new APIs (see §3) |
| `session.rs` (canon items, FNV-1a-64 ×2 addressing, resume-point strip) | port hashing + strip logic; storage moves to `WorkingSet::{update_index, from_index, remove_index}` (see §4) |
| `streaming.rs` (SSE writer) | rewrite: chunk JSON assembly stays, transport becomes `session::send` |
| `handler.rs` (HTTP orchestration) | rewrite around the receive loop; generation moves to PTIR (see §5) |
| `lib.rs` (`#[wstd::http_server]` routing) | gone; replaced by `#[inferlet::main]` + loop |

## 3. Rendering (the parity-critical part)

The chat template is now engine-side Rust (`model/qwen_3/src/chat.rs`, ChatML for all
Qwen variants; no Jinja, no `apply_chat_template`). Three engine helpers diverge from
the HF template and must NOT be used for history replay:

1. **`tools::equip()` makes its own system turn.** HF folds tools into the same
   `<|im_start|>system` turn as the user's system prompt. Render the merged turn by
   hand: `chat::system(system_text + tool_prompt)` (the engine's tool prompt begins
   with a leading space — that is the concatenation seam).
2. **No `assistant_with_tool_calls` renderer.** Hand-emit
   `chat::assistant(content + "\n<tool_call>\n{json}\n</tool_call>" …)` — byte-identical
   between generation replay and history replay, or index keys self-invalidate
   (finding 1 of §5a in the old plan; it still binds).
3. **`tools::answer()` renders one user turn per call.** HF batches consecutive tool
   results into one turn. Batch by hand into a single `chat::user`-shaped turn with
   N `<tool_response>` blocks.

Also carried over: the `/no_think` soft switch (there is still no
`enable_thinking` control anywhere in the new ABI); reasoning stripped from replay
(engine's `assistant()` already strips `<think>…</think>`); `chat::stop_tokens()` for
the stop set (`seal()` is a duplicate of it, not a single-token close);
`chat::cue()` for the generation header. `tools::Decoder` events: `Start` fires on
every non-call feed — only `Call` is meaningful.

## 4. KV reuse: named snapshots → indexed working sets

The prefix trie does not do automatic cross-process matching (that path is
`#[cfg(test)]`-gated and unwired). The discipline is the old one, promoted into the
engine — reference implementation:
`runtime/engine/tests/inferlets/prefix-cache-e2e/src/lib.rs`.

- Keep the old canon/hash scheme; the key (≤256 bytes) is the old
  `qwenchat/{hash32}` string.
- **Save:** after a turn, `slice` the working set to the full-page prefix
  (`page_len` must equal `mapped_len` — no unmapped tail) and `update_index(key)`.
  Insert-or-replace is atomic; replacing frees the old entry (this reproduces the old
  take-on-hit ≤1-snapshot-per-branch bound if we also `remove_index` the parent key
  on extension, as old `session.rs` did via open+delete).
- **Resume:** strip trailing tool/user suffix → hash → `from_index(key)`. Hit: fresh
  working set already holding the prefix; `reserve` only the shortfall; build the
  forward pass over **suffix rows only** with `writable_pages: (cached_pages)..`
  (declaring the whole range would CoW-copy the entire cached prefix). Miss: full
  rebuild, always safe. Sub-page remainder always recomputes — reuse granularity is
  one KV page.
- **`cached_tokens`** = `ws.page_len() * model::kv_page_size()` at resume.
- **Retention:** there is no LRU — index entries survive until the first KV-pool
  pressure event, then *all* non-open entries are dropped wholesale. Same
  "rebuild-on-miss is correct, just slower" posture as before; note in README.
- Hybrid models (qwen3_5 GDN): `rs-working-set` has **no index surface** — no
  cross-request KV reuse at all. Gate: if `model::pass_kind() != attention`, run
  reuse-off (rebuild every turn) rather than erroring.

## 5. Generation: PTIR decode loop

Template: `tests/inferlets/chat-completion/src/lib.rs` (same-tree, 294 lines).
`ptir::attention::prelude`, `run_ahead` driving a 1-wide decode pass whose epilogue
samples (`reduce_argmax` at t=0, `nucleus_sample` otherwise) and re-feeds geometry
channels; chunked prefill via `prefill_chunks(n, None)` respecting
`model::max_embed_length()`; `chat::Decoder` for incremental detok;
`reasoning::Decoder` for the `<think>` channel if it ever appears. Grammar-constrained
tool calls stay **off** (old finding 2: the portable-driver trap; unverified on the
new engine — re-evaluate behind a flag later).

## 6. Scaffolding + config deltas

`integrations/qwen-code/` (recreate; old copy recoverable from `f380e9904`):

- `shim.py` (new), `run_pie_qwen.sh` (update: new `pie serve` + shim instead of
  `launch_daemon.py`), `test_acceptance.py` (port, near-unchanged — it speaks raw
  HTTP), `README.md`, `pie_config.toml` (rewrite: `[model]` singular,
  `type = "metal"`, `device = ["metal:0"]`).
- **Metal is 4-bit-only now** — every matvec binds `.weight/.scales/.biases`. Local
  model becomes `mlx-community/Qwen3-0.6B-4bit`; a bf16 repo imports fine and then
  fails at bind time.
- Server: `cargo build --release -p pie-bin --features driver-metal`; `pie serve`
  boots embedded controller+gateway+worker; clients need the `x-pie-identity` header
  or the WS upgrade 401s before the socket opens (python client defaults it).

## 7. Execution order

- **A. Inferlet skeleton** — crate in `tests/inferlets/chat-completions/` building for
  `wasm32-wasip2`; `types.rs`/`filter.rs` ported; C1 unit tests green against the
  wire fixtures (`tests/inferlets/fixtures/`, to be committed on this branch).
- **B. Render + generation** — §3 rendering + §5 decode loop; single-request
  end-to-end via a throwaway driver script against `pie serve`.
- **C. Shim + acceptance** — `shim.py`; port `test_acceptance.py`; target the same
  33-row bar as §5a of the old plan.
- **D. KV sessions** — §4 resume/save; two-turn echo-back with `cached_tokens > 0`;
  then e2e with stock qwen-code v0.21.6 on M2 / Qwen3-0.6B-4bit.
- **E. Docs + results** — README, results appended here; memory updated.

Risks, front-loaded: (1) renderer parity — same C3 gap as before, now with three
known engine-template divergences to hand-render around; (2) PTIR is a full rewrite
of the generation core — the old integration reused `openhands-completion` idioms
that no longer exist; (3) 4-bit Metal weights change the local model artifact; (4)
first run of the new engine on this M2 is itself unproven (build in progress).

## 8. Results (2026-08-11, first bring-up on the rewritten engine)

Phases A–D executed same-day. Inferlet: 21/21 native unit tests; wasm32-wasip2
builds clean. One design change vs §4: KV sessions are retained **in-process**
(`HashMap<key, WorkingSet>`, fork-on-hit CoW, LRU 8) instead of the engine index —
`update_index` requires a full-page mapped slice and sealed chat turns essentially
never land on page boundaries; the daemon is long-lived so in-process retention
preserves exact-token resume, and a restart degrades to a clean rebuild miss.
Addressing (`qwenchat/{hash32}`) unchanged, so moving to the engine index later is
storage-only.

**Dummy driver (M2):** stack boots, WS client path green, acceptance 25/27 — the
two failures are the tool-call rows, which need a real model.

**RTX 3090 (RunPod, sm_86, CUDA 12.9 toolkit):** acceptance **33/33** — full
parity with the old integration's bar, including native tool calls with unique
ids and `finish_reason:"tool_calls"`, and turn-2 KV resume with
`cached_tokens > 0`. E2E with stock qwen-code v0.21.6 (audited §6 profile):
native `run_shell_command` decoded and executed, file created, clean exit, zero
retries. Session log shows resume hits at **cached 8,841–8,968 tokens** on the
follow-up agent turns (the full system+tools+history prefix) — same reuse
behavior as the old engine's 99.7% result.

Bring-up potholes for the next pod: the CUDA driver needs toolkit **≥ 12.9**
(`cublasGemmGroupedBatchedEx` 12.5+, cublasLt `MATRIX_SCALE_BLK128x128/VEC128`
12.9+) — RunPod's cuda-12.4 images fail at `gemm.cpp`; install
`cuda-toolkit-12-9` and build with `CUDACXX=/usr/local/cuda-12.9/bin/nvcc
CMAKE_CUDA_ARCHITECTURES=<sm>`. The dev-branch python client needed the
"Already authenticated" sentinel fix (committed here). Metal on a shared 8 GB
M2 is gated by the host-reclaimable guard (§6 of the README) — the GPU pod is
the practical e2e environment.

qwen-code behavior notes: `--safe-mode` blocks `write_file` in headless yolo
mode (tool call round-trips correctly, execution refused) — drop it for e2e
tasks or phrase tasks as shell commands; sub-page conversations (<32 tokens)
report `cached_tokens: 0` by design (page-granular reuse).

## 9. C3 renderer parity (2026-08-11) — 23/23 exact

The load-bearing check both plans deferred is built and green:
`echo_tokens: true` on a request makes the inferlet return its full rendered
token stream (render-only, non-stream, session-free), and
`integrations/qwen-code/parity/check_render.py` byte-compares that against
`AutoTokenizer.apply_chat_template(...)` (transformers 5.8.1, Qwen/Qwen3-0.6B)
for every checked-in wire capture. Runs entirely on the dummy-driver stack —
rendering needs only the tokenizer.

First run: 23/23 MISMATCH, from two real renderer bugs — both inherited from
the old "verified" renderer, and both plausibly implicated in the old H200
trajectory divergences (§5b of the old plan):

1. **Tools seam.** Old `equip_after_system` produced `content\n\n\n# Tools`
   (its tool block carried a leading `\n` *and* the merge added `\n\n`); the
   HF template renders `content\n\n# Tools`. Fixed: the block starts at
   `# Tools`, the seam belongs to the merge.
2. **jinja `tojson` re-serialization.** The HF template runs both the
   `<tools>` schemas and replayed `tool_calls[].function.arguments` through
   transformers' `tojson` (spaced `", "`/`": "` separators, insertion-order
   keys, raw non-ASCII) — a vLLM-served model saw that form, while we
   replayed qwen-code's compact echo strings verbatim. Fixed with
   `render_text::tojson` (+ serde_json `preserve_order`), applied to both
   sites.

After the fixes: **23 exact, 0 known-divergence, 0 mismatched** — including
the generation cue. Caveat kept honest: no fixture sets
`enable_thinking:false`, so the one *deliberate* divergence — our
position-independent `/no_think` user-turn decoration vs HF's empty
`<think>\n\n</think>` block after the cue — never fires in this corpus. It
remains normalized-and-reported by the checker (`[KNOWN-DIV]`), not silently
accepted. Acceptance stayed 25/27 on dummy (the 2 are real-model rows).

## 10. 30B A/B rerun (2026-08-12) — INCOMPLETE, pie arm only

Attempted on H200 (secure pod, sm_90, CUDA 12.9 toolkit, driver 575.57.08),
Qwen3-Coder-30B-A3B-Instruct, same 5-task battery as the old §5b run. **The
run did not complete**: the RunPod account's credit ran out mid-benchmark
(four pods across parallel sessions burning $7.92–11.51/hr), RunPod
terminated every pod, and the results directory — wire captures, per-task
logs — went with it. Only the console summary survives:

| task | wall | rc | check |
|---|---|---|---|
| shell-create | 20.50 s | 0 | PASS |
| fix-off-by-one | 11.58 s | 0 | FAIL |
| grep-count | 16.91 s | 0 | PASS |
| add-function | 8.92 s | 0 | FAIL |
| rename-refactor | 3.58 s | 0 | FAIL |

Total wall 61.5 s (old-engine pie arm: 45.1 s, 5/5). **Do not read this as a
regression**: different pod and driver, no token/cache statistics were
collected (`summarize.py` never ran), and the vLLM arm never started, so
there is no baseline for this hardware. The one substantive observation,
from the single surviving capture I inspected before the pod died:
`rename-refactor` ended after one model turn that returned prose and
`finish_reason:"stop"` with no tool call — i.e. a no-call turn, not a
decode failure. Temperature was the inferlet default (0.7): qwen-code
v0.21.6 still has no settings path that puts `temperature` on the wire, so
t=0 equivalence remains impossible without a fork patch (same constraint as
§5b).

**vLLM arm blocker (fixed for next time).** vllm 0.25.1 ships CUDA-13-linked
wheels (torch cu130 + extensions against `libcudart.so.13`) and requires
driver ≥ 580 — the old §5b pod had 580.159.04, this one had 575.57.08.
Torch refuses to initialize; force-swapping torch to cu128 gets torch
working but leaves vllm's own extensions unloadable. `start_vllm_arm.sh`
now gates on driver major ≥ 580 and fails fast with that explanation.

To finish this benchmark: top up RunPod, provision an H200 whose driver is
≥ 580, run `pod_bootstrap.sh`, then the two arm scripts and `summarize.py`.
Budget ~1.5 h at ~$4.60/hr, and watch for other sessions' pods on the same
account.

## 11. 30B A/B, run 2 (2026-08-12) — INVALID, and the bug it exposed

Ran cleanly end to end on an H200 (secure, driver 580.126.09, CUDA 12.9
toolkit) with both arms on one machine: pie (serve + shim + inferlet) and
vLLM 0.25.1 fair tier, Qwen3-Coder-30B-A3B-Instruct, the 5-task battery.
Raw result, archived under `bench/results-2026-08-12-run1/`:

| arm | ok | wall | prompt tok | cached | completion |
|---|---|---|---|---|---|
| pie | 3/5 | 123.7 s | 187,333 | 144,035 (76.9%) | 2,091 |
| vLLM | 5/5 | 30.3 s | 226,877 | (APC internal) | 2,956 |

**These speed numbers mean nothing, and must not be quoted.** All five
trajectories diverged, and the cause was ours: the port renders the
hermes/JSON tool preamble for every model, but Qwen3-Coder is tuned on the
`<function=…>` XML dialect. Served the wrong dialect, the 30B answered two
tasks with a sentence of intent and `finish_reason:"stop"` — no tool call
at all — and took different paths on the rest. vLLM, driving the model's
own template, produced the correct trajectories. So the A/B measured a
prompt bug in pie's client, not the serving stacks.

The bug is not a port regression in the strict sense — the old engine had
`ToolFormat::{Json,Coder}` and picked Coder for these models — but the port
carried over only the *parser* (`salvage.rs` handles bare Coder-XML) and
not the *renderer*. The C3 parity check missed it because every wire
fixture came from a hermes-dialect model, so 23/23 exact said nothing about
the Coder path.

Fixed in `render_text.rs` (`Dialect::{Hermes,Coder}`), verified against the
model's real `chat_template.jinja` through a checked-in golden. Two details
that golden caught, both of which hand-porting would have gotten wrong:
`| string` in the template's `render_extra_keys` is Python's `str()`, so
booleans render `True`/`False`, not `true`/`false`; and a request carrying
a system message but no tools must not get an empty `<tools>` preamble.
Dialect selection reads the config's `[model] name` — pie exposes no HF id,
and `architecture()` is `qwen3_moe` for Coder and non-Coder alike — so a
Coder deployment must carry "coder" in that name; the bench config does.

Also worth recording from this run, independent of the bug: pie's KV
sessions worked exactly as designed on a real agent workload — 76.9% of
prompt tokens served from cache, resume hits of ~8.7K tokens per follow-up
turn. And vLLM reports no `cached_tokens`, so its APC savings are invisible
to this harness; reuse cannot be compared arm-to-arm, only pie's measured
against its own prompt volume.

Run 3 (dialect fixed) is the first run whose speed numbers will be worth
reading. Gate it on a Coder-dialect C3 parity check before benchmarking.

Note for the record: generation-time turns save KV containing the model's own
compact-JSON tool-call bytes, while a rebuild-from-history renders the
tojson-spaced form; addresses hash canon strings (not bytes), and each path
is self-consistent, so reuse is unaffected — but extend-vs-rebuild token
streams for the same conversation differ in those bytes by design.

## 12. Coder-dialect gate (2026-08-12) — CLOSED, 23/23 exact on a served Coder model

The §11 fix was implemented and golden-tested but had never faced a served
Coder model. It has now, locally on the 48 GB machine:

```
pie serve  -c integrations/qwen-code/pie_config_metal_coder.toml   # 17.18 GB bound
parity/check_render.py --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct
→ 23 exact, 0 known-divergence, 0 mismatched of 23 fixtures
```

**No `render_text.rs` changes were needed** — the golden-driven port was
already byte-exact against the real template. The three divergence classes
a static audit of `chat_template.jinja` flagged as reachable in principle
(`| string` on a list-valued `type`; `| string` on boolean/null *tool-call
argument values*, where `coder_assistant_calls_text` uses
`serde_json::to_string` and would emit `true`, not Python's `True`; and the
template's `loop.previtem` guard suppressing the `<|im_start|>user` header
when a tool message opens the conversation) are all **unreachable in these
23 fixtures** — verified by scanning the corpus: 0 boolean/null tool-call
args, 0 list-valued param types, no tool-first conversation. They remain
latent bugs, not fixed ones, and a wider corpus can still trip them.

Because "23 exact" alone could pass for the wrong reason, the run was
checked against eight discriminating markers: the prompt carries the Coder
`# Tools` header, `<function>/<name>` schema form, `<function=…>` call
format and `<parameter=…>` replay, and carries *neither* hermes preamble
nor the hermes JSON call form, with no stray `/no_think`. All eight hold —
the Coder dialect is genuinely what a served Coder deployment now receives.

Two facts about the reference, both load-bearing:

- **The MLX 4-bit build ships an older chat template than the official
  repo** — no `# Tools` header, and `render_item_list` in place of
  `render_extra_keys`. pie renders the *official* template (hardcoded in
  `render_text.rs`), which is also what vLLM serves from
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`, so the A/B stays apples-to-apples.
  But regenerating the golden from the MLX snapshot would silently install
  the wrong reference.
- The two repos' **tokenizers are byte-identical** (vocab, added/special
  tokens, encode and decode round-trip), so decoding `echo_tokens` through
  the MLX artifact and comparing against the official tokenizer is sound.

### Still NOT verified: Coder tool-calling behavior end-to-end

The gate covers **prompt bytes only**. `echo_tokens` short-circuits before
the forward pass (the 23-fixture run takes 2.1 s), so it never touched the
model's numerics — and the numerics are broken here. See §13. The
handover's second half of step 1, "a tool-using request should produce
`tool_calls`, not prose", remains **unverified** and needs a CUDA pod.

## 13. Blocker: the 30B MoE produces garbage on Metal (2026-08-12)

`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` loads, admits, and
generates on Metal — incoherently:

```
"say hi"              → ".isoizonoczoczbar2andrardard2andr-dimensionalood people illiol"
"What is 2+2?" (t=0)  → "/\n/\n/</</</</</</<_bad/</</</</<<|fim_middle|>/<<//<"
```

Greedy output is **deterministic across repeats**, so this is numerics, not
sampling. Controlled against a dense model on the same driver, same binary,
same day: `mlx-community/Qwen3-0.6B-4bit` answers all three probe prompts
coherently. **The Metal driver is fine; the MoE path is not.**

**Root-cause hypothesis (evidenced, NOT confirmed).** The Coder MLX build
is mixed-precision: `quantization` is globally `{group_size: 64, bits: 4}`,
but every MoE router — `model.layers.N.mlp.gate`, all 48 — carries a
per-tensor override to **8 bits**. `driver/metal/src/model_facts.cpp:74-82`
reads only the top-level `bits`/`group_size` and has no per-tensor override
path. An 8-bit router decoded as 4-bit is noise, so top-8-of-128 routing
picks the wrong experts every token — which is exactly the observed failure
shape (locally fluent fragments, globally meaningless). The dense 0.6B has
a single uniform quant block and no overrides, consistent with it working.

Not yet checked: whether the `.zt` artifact carries per-tensor quant
metadata that the loader honors independently of these config facts, which
would refute the hypothesis. `pie model info` does not surface it.

This does not affect §12 — rendering never runs the model — but it blocks
local Coder e2e work, and it is a driver bug worth fixing or filing
regardless of this integration.

## 14. Retarget to Qwen3.6-35B-A3B + vllm-metal (2026-08-12) — plan and cost

Decision: run 3 switches the benchmark model to `Qwen3.6-35B-A3B` and
replaces the CUDA-pod A/B with a fully local one against
`vllm-project/vllm-metal`. Both arms then run on this 48 GB Mac.

**HANDOVER §5's "vLLM does not run on Apple Silicon GPUs" is now stale.**
`vllm-project/vllm-metal` (community-maintained, MLX-backed) installs
cleanly here: plugin 0.3.0.dev + vLLM **0.27.0** into `~/.venv-vllm-metal`,
native arm64 Python 3.12 required. Note the baseline moves on *two* axes at
once — 0.25.1→0.27.0 and CUDA→MLX — so run-3 numbers are not comparable to
run 2's, which were invalid anyway.

### The renderer cost: Qwen3.6 is a THIRD dialect

Because pie's renderer is now byte-exact to the Qwen3-Coder template (§12),
diffing Qwen3.6's template against Qwen3-Coder's over the same 23 fixtures
measures the renderer delta directly. Result: **0/23 identical, 23/23
differ**, first divergence at byte 19. It is neither `Hermes` nor `Coder`:

| | Coder (implemented) | Qwen3.6 (not implemented) |
|---|---|---|
| system turn | system content, then tools | **tools first**, then `\n\n` + *trimmed* system content |
| `<tools>` body | Coder XML `<function><name>…` | **`tool \| tojson`** — hermes-style JSON, envelope kept whole |
| call format | `<function=…>/<parameter=…>` | same |
| arg values | `\| string` → `True` for bools | **inverted**: strings raw, all else `tojson` → `true` |
| first call spacing | always `\n<tool_call>` | `\n\n<tool_call>` if content, else `<tool_call>` |
| tool responses | `<tool_response>\n…\n</tool_response>\n` each | `<\|im_start\|>user` + `\n<tool_response>\n…\n</tool_response>` |
| assistant content | raw | `\| trim` |
| thinking | none | `<think>` state machine (see below) |
| generation prompt | `<\|im_start\|>assistant\n` | + **`<think>\n`**, or `<think>\n\n</think>\n\n` when `enable_thinking:false` |
| multi-part text | joined with `\n` | **concatenated with no separator** |

So this is a new `Dialect` variant in `render_text.rs`/`render.rs`, not a
tweak. Two parts carry real risk:

1. **The thinking state machine.** Assistant turns keep their
   `<think>…</think>` block only when they fall *after* the last genuine
   user query (`ns.last_query_index`, computed by scanning messages in
   reverse and skipping user turns that are wholly `<tool_response>`);
   earlier ones are stripped to content alone. `reasoning_content` is taken
   from the message field when present, else split out of the content. No
   existing pie code path models any of this, and it is *position
   dependent* — which is exactly the property `no_think_decorate` was
   written to avoid, because it interacts with KV-prefix reuse: a turn's
   rendering changes as later turns arrive, so retained prefixes can be
   invalidated. This needs checking against `session.rs` addressing.
2. **The `/no_think` class becomes load-bearing.** Qwen3.6 is a thinking
   model, so `enable_thinking` now changes the generation prompt itself.
   HANDOVER §3 lists that path as untested, with no fixtures.

Also note the fixture corpus was captured from a *hermes* model; retargeting
to Qwen3.6 means the 23 wire captures no longer represent what qwen-code
would send this model, so a fresh capture is needed for a fair gate.

### The engine cost

`Qwen3.6-35B-A3B` is `qwen3_5_moe`: 40 layers in a 3:1
`linear_attention`/`full_attention` GDN hybrid, **256 experts** (8 active),
2 KV heads, head_dim 256 — and multimodal (vision tower present). pie's
Metal driver cannot correctly run *plain* `qwen3_moe` today (§13), so this
is a larger ask than the bug already open. `mlx-community/Qwen3.6-35B-A3B-4bit`
(~20 GB) fits 48 GB; bf16 (71.9 GB) does not.

### The fairness constraint, carried forward

vllm-metal's support matrix marks **Automatic Prefix Cache ❌ for the
Qwen3.5/3.6 family** (✅ for dense Qwen3, Qwen2.5, Llama 3, …). pie reuses
KV across agent turns; on this model the vLLM arm cannot. With ~9–15K-token
prompts resent every turn, that difference dominates wall-clock. Any run-3
report must therefore state plainly that the vLLM arm ran without prefix
caching, and must not present the gap as an engine-vs-engine result. This
strengthens, not replaces, §4's existing rule about `cached_tokens`.

### A better reference for the new gate: vLLM's `/render`

vLLM 0.27 exposes `POST /v1/chat/completions/render`, which returns the
exact `token_ids` it would feed the model (9,626 for fixture
`episode0/…eff10f97`). That is a strictly better parity reference than
`apply_chat_template`: it *is* the baseline arm's behavior, so it removes
the "is our reference even right?" question that §12 had to answer by hand.
The new gate should compare pie's `echo_tokens` against this endpoint.

It requires the server to run with `--enable-auto-tool-choice
--tool-call-parser qwen3_xml` (alias `qwen3_coder`; both map to
`Qwen3EngineToolParser`, `structural_tag_model = "qwen_3_coder"`). Without
those flags a tools-bearing request is rejected with HTTP 400 — and, more
importantly, **the benchmark arm would not emit `tool_calls` at all**,
reproducing run 2's failure mode on the vLLM side. `start_vllm_arm.sh`
must set them.

Decoding that stream against `hf_reference()` already caught a real
divergence, 3 bytes over 43 KB:

```
vllm: …</system-reminder><system-reminder>\nThis is the Qwen Code…
hf  : …</system-reminder>\n<system-reminder>\nThis is the Qwen Code…
```

**Multi-part text content concatenates with no separator.** The template's
`render_content` macro emits `item.text` per part with nothing between,
while both `check_render.py`'s `hf_reference` (`"\n".join(...)`) and pie's
`MessageContent::as_text()` join with `\n`. The corpus has 146 multi-part
messages, so under Qwen3.6 pie would diverge on nearly every turn. Fixing
this means `as_text()` needs to be dialect-aware — the `\n` join is correct
for the hermes captures it was written against and wrong here.

Confirmed from the same decode: the generation prompt really does end
`<|im_start|>assistant\n<think>\n`, so the thinking seed is unavoidable.

## 15. Scoping the pie arm for Qwen3.6 (2026-08-12) — the driver was never the problem

Tested rather than reasoned about: imported `mlx-community/Qwen3.6-35B-A3B-4bit`
(20.5 GiB `.zt`) and served it on Metal. **It boots and serves.**

```
[pie-metal] 19.51 GB of weights bound where they lie, and out of the heap
pie standalone serving listen=127.0.0.1:18080
```

So the §14 worry that a 3:1 GDN hybrid with 256 experts would need new
driver work was **wrong**. `driver/metal/src/model/qwen3_5/` already
implements the routed decode end to end — router, top-k, sort, gather,
expert gate/up/silu/down, combine, shared expert — and `facts.hpp` already
maps `qwen3_5_moe`, `qwen3_5_moe_text` and even `qwen3_6` to
`ModelFamily::Qwen35`. A comment in `decode_step_mb.hpp` sizes its dispatch
tables against "the 40-layer 256-expert MoE", i.e. this exact model.

**This also localizes §13.** `qwen3_moe` (the Coder 30B) maps to
`ModelFamily::Llama`; `qwen3_5_moe` maps to `ModelFamily::Qwen35`. They are
*different decode paths*. The garbage output is confined to the llama-family
MoE path, and says nothing about the Qwen35 one.

Two boot notes: `max_model_len = 32768` is refused here (23.19 GiB wanted +
the 2 GiB margin against 24.85 GiB reclaimable); **16384 boots**, and still
covers the 15,396-token longest fixture. The engine also caps lanes on this
model — `requested=8 seated=4`, because each lane holds a recurrent-state
slot per posted frame — so `max_forward_requests` is effectively 4.

### The actual blocker is the inferlet, not the engine

```
HTTP 500: unsupported model architecture ForwardKind::Hybrid:
          this daemon requires an attention-only model
```

`handler.rs:112` sets `attention_model = pass_kind() == ForwardKind::Attention`
and `handler.rs:159` refuses everything else. `generation.rs:16` imports
`inferlet::ptir::attention::prelude::*` and calls `fwd.attention(...)` at
three sites (125-127, 197-199, 316-318).

The SDK already has what is needed: `ptir::hybrid`
(`pie:inferlet/forward-hybrid`, documented as "attention layers and
recurrent layers in ONE forward (Qwen3.5 GDN, Nemotron-H Mamba2)"), plus
`RsWorkingSet` for the folded recurrent state — with `state_size`,
`buffer_page_size`, `discard_buffered`, `reorder_buffer`, and **`fork`**, a
"copy-on-write child sharing the current folded state and buffered suffix".

So the port is bounded: swap the prelude, bind an `RsWorkingSet` per request
alongside the KV `WorkingSet`, retain and fork it per session the way
`session.rs` already retains working sets, and delete the gate.

### The one thing to design carefully: reuse semantics

The handler's comment says hybrid "has no fork/retention surface". Given
`RsWorkingSet::fork` exists, that reads as over-conservative — but the
`ptir` docs draw a sharper line: **"KV eviction algorithms are NOT valid
here: dropping a KV page does not undo the fold that already consumed those
tokens."** Append-only growth (fork the state at the end of turn N, extend
with turn N+1) matches what `split_resume_point` already does, so agent-turn
reuse should survive. What does not survive is dropping or rewinding pages
inside a conversation — the fold is irreversible.

That needs verifying before it is claimed, because **KV reuse is the pie
result this benchmark exists to show** (76.9% on the run-2 30B). Combined
with vllm-metal's APC ❌ for the 3.5/3.6 family (§14), there is a real risk
that run 3 on Qwen3.6 measures neither arm's prefix reuse. Worth deciding
deliberately rather than discovering after the run.

## 16. What `dev` already has for Qwen3.6 (2026-08-12)

Searched `fork/dev` for a Qwen3.6-35B-A3B benchmark against vLLM / SGLang /
llama.cpp / mlx-lm. **No such result set is checked in.** What exists:

| | |
|---|---|
| `benches/` | runners for Pie, vLLM, SGLang, **llama.cpp**, **TensorRT-LLM** — no mlx-lm anywhere on `dev` (the only "mlx" hit is a substring in `website/package-lock.json`) |
| Pie's bench arm | `inferlets/text-completion-bench/src/lib.rs` (293 lines), selected by `benches/pie_bench.py:37` |
| Qwen3.6-35B-A3B | appears only in `benches/smoke_deterministic.py` as `qwen3_6_moe` — a **determinism** check (temp 0, sha ledger, TP2), not a perf comparison |
| checked-in results | `results_pie/vllm/sglang.json` are all **Qwen3-0.6B** |
| published figures | `website/docs/overview/benchmarks.mdx` — Llama 3 1B on an L4 vs vLLM 0.6.0 / SGLang 0.4.4, from the SOSP '25 paper |
| llama.cpp link | `driver/portable` (GGUF/ggml); `driver/portable/dev_qwen3_5_35b.toml` targets this exact model |

### The part that changes our plan

`smoke_deterministic.py` drives `pie_bench.py`, which runs the
`text-completion-bench` inferlet — so **Qwen3.6-35B-A3B already runs
through pie on `dev`**. That inferlet uses the **high-level
`Context`/`Model::generate` API** (`Context::new`, `ctx.generate(sampler)`),
not raw PTIR, and carries no `ForwardKind` gate at all.

So §15's blocker is narrower than stated: it is not "the inferlet layer
cannot do hybrid", it is "**our** inferlet hand-rolls a raw-PTIR
`ptir::attention` loop". Two routes now:

1. Port `generation.rs` to `ptir::hybrid` + `RsWorkingSet` (keeps chunked
   prefill and the device-carried decode loop, and keeps explicit KV
   control — which is the whole point of the KV-reuse story).
2. Drop to the high-level `Context::generate` path for hybrid models, as
   `text-completion-bench` does. Cheaper, and there is a working reference —
   but it likely surrenders the explicit KV retention that
   `session.rs` is built on, so it may cost the reuse result.

Route 1 still looks right for this integration, but `text-completion-bench`
is now a working reference for how the SDK drives this model, and worth
reading before writing the hybrid pass.

**Caveat:** that path is verified on **CUDA TP2** (`smoke_deterministic`
pins `devices="cuda:0,cuda:1"`), not on Metal.

Also useful — In Gim's portable-driver bring-up (`a6f710a32`, "Qwen3.5-35B-A3B
verified") documents three Qwen3.6 config traps, the nastiest being
`tie_word_embeddings` living at the **outer** level of a multimodal-wrapped
checkpoint: reading only `text_config` silently ties `lm_head`↔`tok_embd`
and flips argmax on the first token while per-layer activations still match
HF at cosine 0.9985. Checked: `driver/metal/src/model_facts.cpp:303-313`
already handles this (tries `text_config`, falls back to top level, citing
Qwen3.5-35B-A3B), so Metal is not exposed to it.

## 17. The pie arm is live on Qwen3.6-35B-A3B, Metal (2026-08-12) — VERIFIED

The hybrid port of §15/§16 has now served real tokens through *our* inferlet:

```
pie serve -c <metal qwen3.6, max_model_len 16384>   → 19.51 GB bound, seated=4 of 8 lanes
POST /v1/chat/completions  t=0
  "What is 2+2? Answer with just the number."      → "4"          (139 tok, 6.0 s)
  "Write a one-line Python function…"              → coherent      (48 tok, 0.6 s)
```

~80 tok/s after warm-up. This closes the existential question for the pie
arm: **Metal + `qwen3_5_moe` + our chat-completions inferlet generates
correctly.** The §13 garbage is confined to the llama-family `qwen3_moe`
path and does not touch this one.

Cross-check against the oracle: vLLM-metal on the same machine and model at
t=0 opens both answers identically ("Here's a thinking process:\n\n1.
**Analyze User Input:**…") and diverges only later — expected, because the
two prompts still differ (§14). Under identical prompts they should agree
token-for-token; that is the parity gate, not this smoke test.

### The `</think>` hazard, and why our renderer misses it

The opencode branch found Qwen3.6 emitting a `</think>` it never opened,
leaking reasoning text into content and the bare tag into replayed history.
We do **not** reproduce it, and the reason is instructive: their `cue_no_think`
renders a *closed* empty block (`<think>\n\n</think>\n\n`, the Qwen3-0.6B
convention their parity fixtures pin), so the model appends its reasoning
*after* an already-closed block and its own closer has no opener. Our cue
emits no think block at all, so the model opens its own `<think>`, and
`filter.rs` — which enters `Mode::Think` on the opener and drops to
`</think>` — strips the whole block cleanly. Full content for the arithmetic
prompt is exactly `"4"`.

**Correction (same day):** the causal half of that reading is withdrawn. The
opencode branch ran the control — before their `instruct.rs` fix,
`has_thinking` was false, so their cue fell through to plain `cue()`, i.e.
no think block, exactly our configuration — and Qwen3.6 emitted *untagged*
reasoning prose, no tags anywhere. So omitting the block does not reliably
make the model tag, and the closed block is not established as the cause.

The two observations do still conflict, and it is unresolved. Ours was
`finish_reason:"stop"`, **139 completion tokens, content exactly `"4"`** —
which untagged prose cannot produce through a marker-based filter, so a
tagged block was present in our run. The `/no_think` confound is ruled out:
`no_think()` fires only on an explicit `chat_template_kwargs.enable_thinking
== false`, which these requests never sent. Same cue, different tagging, so
the deciding factor is elsewhere in the prompt — our hermes tool preamble
and system-turn handling versus theirs.

That unresolved dependence argues *for* the open-block cue rather than
against it: if tagging varies with prompt context neither side controls, no
marker-based filter is safe in either direction.

This is luck, not design, and it is fragile in one direction: a turn cut off
by `max_tokens` *before* the closer arrives still emits the reasoning
preamble as content (seen here at 48 tokens). The durable fix is the same
one the opencode branch identified from the other side — a lineage-aware cue
that leaves the block OPEN for the 3.5/3.6 lineage, matching that template's
`<|im_start|>assistant\n<think>\n`, so the filter can start in think-mode
and suppress deterministically with no buffering and no heuristics. That is
renderer work, and it belongs with the third-dialect implementation (§14).

### Also confirmed: KV fork is Metal-unsupported off this geometry

Acceptance on the dense 0.6B is 31/1, and the one failure is `turn 2
(echo-back history)` — the resume path — with:

```
[pie-driver-metal] copy_kv: UNSUPPORTED — this increment only supports
                   the qwen3.6 (GDN-hybrid) checkpoint geometry
```

The gate is `driver/metal/src/context.cpp:1978`, `if (!facts_.has_linear_attn)`
— so hybrid is the *supported* case and dense checkpoints are the refused
one, not the other way round. Proven pre-existing by re-running the identical suite against stashed,
unmodified inferlet code: same 31/1. So `WorkingSet::fork` — the entire
KV-reuse resume path, and pie's headline benchmark result — is currently
implemented on Metal *only* for the qwen3.6 geometry. That explains why the
handover's 33/33 was measured on an RTX 3090, and it means the KV-reuse half
of run 3 is only testable locally on the model we happen to have chosen.
Reuse on Qwen3.6 itself is still unmeasured: `cached_tokens` was 0 on every
turn here because each probe was a fresh conversation.

## 18. Qwen3.6 dialect + open-think cue (2026-08-12) — and a seal bug the measurement found

### Implemented and live-verified

`Dialect::Qwen36` now renders the Qwen3.5/3.6 template, generated as a
golden from `Qwen/Qwen3.6-35B-A3B`'s real `chat_template.jinja`
(`tests/qwen36_template_golden.json`) and matched byte-for-byte:

- tools block FIRST, then `\n\n` + the trimmed system message (inverted vs Coder)
- `<tools>` carries `tool | tojson` — hermes-style JSON, envelope whole
- calls are Coder-style XML, but the argument rule is **inverted**: strings
  raw, everything else `tojson`, so a bool renders `true`, not `True`
- first-call spacing is `\n\n` with content, nothing without
- tool-result blocks joined by `\n`, last NOT terminated
- multi-part text concatenated with NO separator (`part_separator()`)
- think replay is POSITIONAL — kept only after the last genuine user query,
  so `render_messages_at` takes an absolute offset and the last-query index
  computed over the WHOLE list; a suffix rendered with suffix-relative
  indices would disagree with the prefix already in KV

The cue leaves the block OPEN (`<|im_start|>assistant\n<think>\n`) and
`VisibleFilter::starting(true)` begins in think-mode. **Verified live on the
35B, and the property that matters held at both temperatures:**

| max_tokens | t=0 | t=0.6 |
|---|---|---|
| 8 (truncated mid-reasoning) | `'…'` — no leak | `'…'` — no leak |
| 700 (allowed to finish) | `'Hello to you.'` (218 reasoning tokens suppressed) | `'Hello to you.'` (201 suppressed) |

Temperature-indifference is the point: it is what the marker-based
alternatives cannot offer while the tagging question stays open (§17).

Live verification also caught a bug unit tests could not. The empty-content
fallback stripped only the `<think>`/`</think>` *tags* from the raw
generation and kept the reasoning body — so exactly when the filter
correctly returned nothing, the handler substituted the model's private
reasoning as its answer. Under the open cue that is the common case, not a
rare one. Fixed to take the body after the last `</think>`, and to treat an
unterminated block under the open cue as having no content at all.

Regression: hermes parity still **23 exact / 0 known-div / 0 mismatched**,
37 native tests, acceptance 31/1 (the 1 being §17's `copy_kv` limit).

### NOT working: session retention on the hybrid path

KV reuse on Qwen3.6 measured **0%**, and the cause is not page granularity
or address mismatch — **the seal fire fails on every turn**:

```
seal failed, session dropped: seal take: sink_s take:
channel is poisoned: driver published poison epoch 1        (8 of 8 turns)
```

`handler.rs` only skips retention on `gen_error`, so every turn reaches
`seal`, and every seal fails, so every session is dropped and every
subsequent turn is a `resume miss`. The dense 0.6B attention path retains
normally (`retained qwenchat/… (seq N)`), so this is specific to the hybrid
port.

**The pipeline hypothesis was WRONG, and the fix built on it was too.** Both
this branch and `liu/opencode-integration` reasoned that `run_ahead` closing
the generation pipeline left the fold with no cross-pipeline identity, and
that forking it onto the seal's own pipeline would re-order it. That was
plausible, agreed by both branches, and false. Run with the fork in place:

```
[pie-driver-metal] paged continuation: recurrent slot 1 holds sequence
                   9223372036854775808, this fire is sequence 9223372036854775809
```

`fork` mints a NEW sequence (2^63 → 2^63+1) and the driver requires a
continuation to carry the slot's own. The fork did not fix the bug; it added
a second, different rejection on top. Reverted.

**The actual cause is position accounting, and it is structural.** The
pre-fork driver messages say it plainly:

```
recurrent slot 0 is at position 165, this fire starts at 160   ← fold 5 AHEAD
recurrent slot 0 is at position 14,  this fire starts at 15    ← fold 1 BEHIND
```

Two distinct mismatches, both fatal to the seal:

- **Fold ahead.** The decode loop uses `run_ahead`, which fires
  speculatively past the stop token. For KV that is harmless and
  deliberate — `generation.rs` says so: those fires "leave garbage KV beyond
  the accepted length; it is never referenced because every later fire's
  `kv_len`/page-CSR only ever cover valid tokens". **A fold has no
  `kv_len`.** It advances on every fire that executes and cannot be rewound,
  so the speculative overshoot is permanently folded in. The seal then tries
  to start at the *accepted* length and lands 5 behind the fold.
- **Fold behind.** The stop token is "truncated at, never written" for KV,
  so `total_len` can sit one past where the fold stopped.

So the KV-side accounting this loop is built on — tolerate overshoot, mask
it with `kv_len`, truncate at the stop token — is exactly what an
irreversible fold cannot support. This is the same property the SDK warns
about for eviction ("dropping a KV page does not undo the fold"), reaching
the seal by a different route.

Fixing it means making the fold position and `total_len` agree by
construction: either no speculation on a hybrid pass (each fire's tokens all
accepted before the next is submitted), or the seal starts at the fold's
position rather than the accepted length — and the second only works if
nothing was ever folded that KV does not contain. Neither is a small change,
and neither is attempted here.

Until this is fixed, **pie's KV-reuse result is unobtainable on Qwen3.6**,
which is the only geometry where Metal implements CoW fork at all (§17). It
is now the top of the queue: the reuse number is the headline pie result the
A/B exists to produce.

## 19. A parity gate referenced to vLLM, not to the template (2026-08-12)

`check_render.py` compares pie against `apply_chat_template`, which answers
"does our renderer match my reading of the template". That was the right
question while the template *was* the specification (§12). For the Qwen3.6
A/B it is the wrong one: what the benchmark compares against is what the
**vLLM arm actually feeds the model**, and vLLM 0.27 will just tell us —
`POST /v1/chat/completions/render` returns the exact `token_ids`.

The served reference also removes a class of error the HF path cannot avoid,
because `hf_reference()` has to re-implement vLLM's content normalization by
hand. It had the multi-part separator wrong until vLLM's own render caught
it — 3 bytes over 43 KB (§14).

Both stacks want ~20 GB and cannot be up at once on 48 GB, so the gate is
split in two:

```bash
# 1. capture the reference (vLLM up, pie down)
vllm serve mlx-community/Qwen3.6-35B-A3B-4bit --port 18000 \
    --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser qwen3_xml &
python3 parity/capture_vllm_render.py --base http://127.0.0.1:18000 \
    --model mlx-community/Qwen3.6-35B-A3B-4bit -o parity/vllm_render_qwen36.json

# 2. compare (vLLM down, pie up) — token ids, not text
python3 parity/check_render.py --base http://127.0.0.1:8123 \
    --reference parity/vllm_render_qwen36.json
```

Comparing **ids rather than text** drops decode round-tripping out of the
loop; text existed in the HF path only because `apply_chat_template` returns
a string. The capture script drops generation policy (`max_tokens`, `stream`,
…) and sends only the conversation fields, so `/render` cannot reject a
request over a field that has no effect on the prompt.

The `--enable-auto-tool-choice --tool-call-parser qwen3_xml` flags are not
optional: without them a tools-bearing request 400s, and a *benchmark* arm
launched without them emits no `tool_calls` at all — run 2's failure mode
reproduced on the vLLM side.

Written and syntax-checked; **not yet run** — the machine is with another
session measuring non-speculative hybrid decode throughput (§18). The
captured reference is reusable by any branch doing prompt-parity work
without needing both stacks up.

## 20. Tool calls on Qwen3.6, and a concurrency guard (2026-08-12)

**Acceptance on Qwen3.6-35B-A3B, Metal: 34 passed, 1 failed.** The single
failure is §18's seal/fold bug (`cached_tokens = 0`). Everything else —
streaming, tool-call atomicity, id uniqueness, degrade discipline, fixture
replays, concurrency — is green on the 35B.

### The parser was the mirror of run 2

Getting there needed one fix, and it is the exact inverse of the bug that
invalidated run 2. There, the renderer was wrong and the model never called
a tool. Here the renderer is right — the model emitted a textbook Qwen3.6
call —

```
<tool_call>\n<function=get_time>\n<parameter=timezone>\nAsia/Tokyo\n</parameter>\n</function>\n</tool_call>
```

— and the turn still returned `tool_calls: null` with that text sitting in
`content`. The **parser** dropped what the model correctly produced.

Cause: the Coder/XML salvage scanned only `visible_text`. A *well-formed*
call is wrapped in `<tool_call>…</tool_call>`, and `filter.rs` treats
`<tool_call>` as an opener and drops the whole block — so a correct call is
absent from `visible_text` **by construction**. The condition therefore
caught the malformed case (a bare `<function=` leaking into content) and
missed the correct one: exactly inverted. The engine `tools::Decoder`
understands only the hermes JSON form, so nothing else catches the XML
dialects. Now scans the visible text first and then the raw generation.

This would have invalidated run 3 the same way the dialect bug invalidated
run 2 — a Coder-dialect model producing zero parsed tool calls — and no unit
test could have caught it, because it needs a real model emitting a real
call through the real filter.

### Concurrency: ours is clean, and now guarded

`liu/opencode-integration` found their serving path degrades **every**
request at N≥2 on both a dense and a hybrid model — a rejected launch
(`pie_metal_launch failed with status -1`) turned by the degrade discipline
into `finish_reason:"length"` with a one-token answer, indistinguishable on
the wire from a model that stopped.

Ours does not reproduce it, measured on both models:

| N | Qwen3-0.6B (dense) | Qwen3.6-35B (hybrid) |
|---|---|---|
| 1 | 200 tok | 150 tok |
| 2 | 200, 200 | 150, 150 |
| 4 | 200 ×4 | 150 ×4 |

No `pie_metal_launch` rejections in either log; the 8 launch failures on the
35B run are all §18 seal-position mismatches. Concurrency is also *faster*
here — 4 requests in 2.1 s against 4.4 s for one — so batching works.

The suite now carries a guard for the class, because a sequential suite
cannot see it: N=2 and N=4 concurrent requests, asserting on **token count,
not status**. Every request is given a budget it should exhaust, so a turn
that stops at `"length"` after a handful of tokens is nonsense on its face —
`"length"` means the budget was hit. That shape also survives the `'…'`
placeholder check, since one *real* token is not the placeholder. Assertion
shape suggested by the opencode session.

## 21. Sequential hybrid decode + in-pipeline seal (2026-08-12) — built, hybrid unverified

Implements the fix §18 called for. Three changes, all confined to the
recurrent-state path; the attention path is untouched and still uses
`run_ahead`.

1. **Sequential decode when `state.rs` is non-empty.** `submit_frame` one
   fire, take it, then submit the next — no window, so no fire executes past
   the stop token and nothing is folded that the turn rejected. Written as a
   hand loop rather than repeated `run_ahead(.., 1, ..)`, which would submit
   into a closed pipeline and **hang**: `run_ahead` closes as soon as its
   budget is spent.
2. **`written` counted before the stop test**, not after. The fire that
   produces a stop token has already folded it, so KV must record it too or
   it ends a token short of the fold. This is the semantic change: on the
   hybrid path the retained context now contains the model's own stop token
   instead of excluding it and re-adding it in the seal.
3. **The seal is the last fire on the generation pipeline**, via
   `seal_in_pipe`. The free-standing `seal` cannot work here at all — it
   opens a fresh `Pipeline`, and a fold has no identity there: binding it
   directly is refused with a poison epoch, forking it mints a new sequence
   and is refused as a continuation mismatch. Only a hand-written loop makes
   this possible, since `run_ahead` closes the pipeline out from under the
   caller. If the model's stop token is already the first token of the turn
   suffix, the seal appends only the remainder rather than doubling it.

That the seal moves onto the generation pipeline came from the
`liu/opencode-integration` session's framing ("the fold never leaves its
process") — which they subsequently corrected, since it removes the
persistence half of the problem but not the pipeline-binding half. The
correction is right and the derived design still holds: it is the *hand
loop*, not the process lifetime, that makes an in-pipeline seal possible.

**Verified:** builds clean, 37 native tests, and the attention path is
unregressed — acceptance 33/1 on the dense 0.6B (the 1 being §17's
`copy_kv` limit, unrelated) with renderer parity still 23 exact / 0
known-div / 0 mismatched.

**NOT verified: the hybrid path itself.** The Metal guard prices the 35B at
22.55 GiB + a flat 2 GiB margin against 21.5 GiB reclaimable — another
session holds a ~3 GiB `pie serve` and lowering `max_model_len` cannot
recover it, since the weights alone are 18.16 GiB. So **KV reuse on Qwen3.6
remains 0% as measured**, and this is a candidate fix exactly like
`a07637621` was — which failed. It should not be described as fixed until a
turn retains and a later turn reports `cached_tokens > 0`.
