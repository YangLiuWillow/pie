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
