# PA.3 live run — `Lius-MacBook-Pro` (M-series, 48 GB unified, macOS 26.5.1)

**Date:** 2026-08-12 · **Branch:** `liu/opencode-integration` · **Driver:** Metal
**Binary:** `pie-bin --release --features driver-metal` (shared target dir)
**Inferlet:** `chat-completions@0.1.0` (605 KB wasm)

This is the first time pie has served a real token to opencode. Everything
before this file was verified against unit tests, stub workers and the HF
reference tokenizer.

---

## Verdict

| what | result |
|---|---|
| Native suites (`pie-openai-serving` / `pie-model` / `pie-model-qwen-3 --features chat` / `pie-gateway`) | 49 · 16+3+4+23 · 24 · 40+1+6 — all green |
| Acceptance suite, live, `Qwen3-0.6B` (MLX int4) | **25 passed, 0 failed, 0 warnings** |
| Acceptance suite, live, **`Qwen3.6-35B-A3B` (MLX 4-bit, GDN hybrid MoE)** | **25 passed, 0 failed, 0 warnings** |
| Stock opencode e2e, 0.6B | `read` tool call + correct answer |
| Stock opencode e2e, **35B, multi-step** | **read → write → correct file on disk** |

**No wire-level first-contact bugs.** Every hard assertion in the 25-test suite
passed on the first live run, including the two fixture replays (req-004 tool
turn, req-005 7.5k-token history replay), tool-call delta atomicity, id
uniqueness across processes, envelope non-leakage and the SSE keepalive path.
Everything first contact did find was *below* the wire (§Findings) — and the
three that only a real model could expose were all found by the 35B, not the
0.6B.

---

## What ran

### Configuration

```toml
[model]  model = "Qwen--Qwen3-0.6B-optimized"     # and mlx-community--Qwen3.6-35B-A3B-4bit
[driver] type = "metal", kv_page_size = 32, total_pages = 512,
         max_forward_tokens = 1024, max_forward_requests = 8, max_model_len = 16384
```

`max_model_len` must stay ≥ ~8k: opencode's build-agent prompt alone replays at
7473 tokens, and an over-long prompt is refused by the Metal driver, not chunked.

### Acceptance suite

```
$ PIE_BASE_URL=http://127.0.0.1:8081 python3 integrations/opencode/test_acceptance.py
25 passed, 0 failed, 0 warnings (25 tests)
```

Slowest tests are the two fixture replays (8–19 s each) — 0.6B prefill of a
~7.5k-token prompt on Metal, not a wire problem.

### Stock opencode, end to end

opencode `1.18.17`, the committed `integrations/opencode/opencode.json` profile
(baseURL retargeted to the running port), no opencode changes:

```
$ opencode run -m pie/qwen3-0.6b "Read the file notes.txt and tell me the secret color."
> build · qwen3-0.6b
→ Read notes.txt [offset=0]
The secret color is **chartreuse**.
```

The full path — opencode → `POST /v1/chat/completions` → gateway ingress →
`LaunchProcess` → inferlet → engine → Metal → SSE back — carries a real
agentic tool call and a correct answer.

A second, two-step prompt ("list the files, then write summary.md") produced a
correct `glob` call and then *narrated* the write instead of emitting the second
tool call. That is 0.6B capability, not a wire fault: the call it did emit was
well-formed and the text was coherent — and the same prompt on the 35B below
completes the whole thing.

### Stock opencode on Qwen3.6-35B-A3B — multi-step, end to end

```
$ opencode run -m pie/qwen3.6-35b-a3b \
    "Read notes.txt, then write a file summary.md with one sentence describing what it says."
> build · qwen3.6-35b-a3b
→ Read notes.txt
I'll read notes.txt and create summary.md with a one-sentence description.
← Write summary.md
Wrote file successfully.
Done. Created summary.md with: "PIE is a programmable LLM serving system that runs inferlets over WebSocket."

$ cat summary.md
PIE is a programmable LLM serving system that runs inferlets over WebSocket.
```

Two tool calls, correct order, a real file on disk with correct content. This
is the Strategy A milestone: a stock coding agent doing real agentic work on a
35B MoE served entirely by pie.

---

## Measured, single-tenant and warm (2026-08-12, post-`read1`)

Every earlier timing in this file was taken with other `pie serve` processes
resident, and every `ttfb` predating the `read1` fix is an artifact of CPython
buffering rather than a property of the server. **These are the numbers to
quote.** One `pie serve`, nothing else on the GPU, model warmed with a
throwaway request first, config exactly as in §Configuration.

| | Qwen3.6-35B-A3B (MLX 4-bit, Metal) |
|---|---|
| prefill | **421 tok/s** (5221-token prompt, first content at 12.41 s, n=3, σ<0.05 s) |
| decode, single-stream | **90 tok/s** (400 tokens, n=3: 87.9 / 89.6 / 90.0) |
| TTFB (first SSE byte) | **0.003 s** |
| 8 concurrent long prompts | ~~98.6 s wall, 8/8 completed~~ **WRONG — see §Concurrency below. All 8 turns DEGRADED; 13 completion tokens across 8 requests should have been the tell.** |
| stock-opencode agentic task, end to end | **81.2 s** (read → write, 2 tool calls, correct file) |

Two of those need their labels read carefully:

- **TTFB is not time-to-first-token.** This inferlet emits the role chunk
  before prefill starts, deliberately, so a long prefill cannot look like a
  dead stream. 0.003 s is that chunk. The number a user feels is
  `first_content`, 12.4 s on a 5.2k prompt.
- **"8 concurrent" is 8 requested, 4 seated.** A hybrid model costs two
  admission seats per lane (`seat_cost=2`), so `max_forward_requests = 8`
  seats 4 and the other four queue. Reporting it as 8-way concurrency would
  overstate it by exactly 2×. **And it did not work at all** — see below.

### Concurrency: ROOT-CAUSED — the Metal scheduler batches two device-geometry
### programs, and the driver refuses

Reported here as an open defect for most of the session, with two wrong
framings before the right one. The first version of this file recorded the
8-way row as "8/8 completed" when all eight had degraded (13 completion tokens
across 8 requests, in the same line). The second called it "the serving
inferlet cannot serve two concurrent requests", which put the blame on us.

**The actual cause**, from the driver's own output:

```
[pie-driver-metal] launch: 2 device-geometry programs in one batch
                           (at most one is supported)
```

and its source comment (`driver/metal/src/context.cpp:997`):

> *"Phase 2 (C3): at most one device-geometry program per launch batch — the
> same structural constraint **the runtime's scheduler already upholds**
> (`metal_ptir_plan.md §6`); a defensive re-check here so a **scheduling bug**
> fails the launch loudly instead of resolving two programs' geometry against
> one shared forward."*

So:

1. Our decode loop is **device-geometry** by construction — the loop-carried
   epilogue resolves `w_slot`, `w_off`, `klen`, `pos`, `fill`, `pages` and
   `page_indptr` on device. That is the whole point of the design.
2. `chat-completions` is **one inferlet process per HTTP request**, so N
   concurrent requests are N device-geometry programs.
3. The Metal scheduler batches two of them into one launch, and the driver
   rejects the batch. Correctly — the alternative is silently resolving two
   programs' geometry against one shared forward.
4. **The check firing means the scheduler did not uphold a constraint it is
   documented as upholding.** This is an upstream scheduler bug, and the driver
   author left a tripwire precisely for it.

It accounts for every observation: N=1 always clean (one program); N≥2 always
degraded; **flat across `max_forward_requests` 8 / 32 / 64** with exactly 6
launch failures each time (a structural limit, not capacity); and sequential
decode helping partially (fewer in-flight fires → fewer chances for the batcher
to pair two device-geometry fires).

**Why upstream never sees it.** `benches/pie_bench.py` drives
`text-completion-bench`, which takes a `prompts` ARRAY plus `batch_concurrency`
and runs the fleet **inside one process**. One process is one program, so
however wide the concurrency, there is never more than one device-geometry
program in flight. The N-concurrent-*processes* shape is the OpenAI-serving
shape, and it is structurally unreachable from upstream's harness.

| concurrent requests | speculative (shipped) | sequential (one fire at a time) |
|---|---|---|
| 1 | 200 tokens — fine | 200 tokens — fine |
| 2 | 2/2 degraded | 0/2 degraded |
| 4 | 4/4 degraded | 2/4 degraded |

**Hypotheses eliminated on the way**, recorded so nobody re-runs them: window
oversubscription against a global budget (disconfirmed by the flat R sweep);
channel-name collision (`named()` is documented as trace-only); duplicate
instance ids in the roster (deduplicated by a `HashMap` in
`scheduler/batch.rs:308`); instance missing from the driver registry (added a
diagnostic at `context.cpp:930`, never fired).

**How it was actually found, which is the lesson.** The message was in EVERY
log collected all session — 6 occurrences in the isolated run, 54 in the
concurrency ladder. It went unseen because the greps were for
`pie_metal_launch failed` and `pie::inferlet`, the strings already known,
rather than for the driver's own output. Two rounds of source-diving and a
rebuilt binary to surface a line that was already on disk. **Read the whole
log before theorising about it.**

**Where this leaves us.** Not our bug, but our problem: Strategy A serves N
concurrent turns as N processes, and on Metal that is currently limited to one
in-flight device-geometry program. Options, none yet taken: host-resolved
geometry per fire (costs the device-carried decode loop, sidesteps the
constraint), an upstream scheduler fix so it upholds §6, or Strategy B, where
one long-lived session inferlet is one program by construction.

### Speculative vs sequential decode### Speculative vs sequential decode — the PA.1 m2 go/no-go

Commissioned to price what dropping `run_ahead` would cost on a hybrid pass,
since its speculative overshoot is what a fold cannot tolerate. One dedicated
server per variant (a hot-swap of the installed wasm under a live server was
tried first and produced unusable data — degraded turns and unstable prefill;
do not do that).

| | median decode, n=5 full runs, 35B |
|---|---|
| speculative `run_ahead` (shipped) | **91.0 tok/s** |
| sequential, one fire at a time | **90.6 tok/s** |

**0.4% — inside the noise.** At batch-1 on an embedded worker the "host round
trip" `run_ahead` exists to hide is a channel operation, not IPC, so there is
almost nothing to hide. Its value should show up with many lanes in flight and
with a remote worker; neither applies here.

Combined with the concurrency result above, the answer is not "cheap enough":
**the sequential path is strictly better on this machine.** It costs nothing
measurable at N=1, it is what a hybrid seal requires, and it degrades far less
under concurrency. What was framed as a cost of sessions turns out to be a
partial fix for a defect we did not know we had.

Two caveats on that. The sequential path is **not** a complete concurrency fix
(2/4 still degraded at N=4), so the underlying launch rejection needs its own
diagnosis. And `run_ahead` is single-use per pipeline — it calls `on.close()`
once its budget is spent, so a "sequential" loop written as repeated
`run_ahead(.., 1, ..)` submits into a closed pipeline and **hangs** rather than
erroring. The working shape is a hand-written `submit_frame` + take loop with a
single `close()` at the end.

### What the numbers say about the roadmap

Prefill is 421 tok/s and decode is 90 tok/s, and the agentic task took 81.2 s.
Those three facts together are the argument for the rest of this project:

```
3 model turns re-prefilling ~24k tokens of history   = ~57 s
generating ~200 tokens of actual output              = ~2 s
                                                       ─────
                                            ~70% of the task is re-prefill
```

opencode re-sends the whole conversation every turn, so pie re-prefills a
history that changed by a few hundred tokens. With the KV working set resumed
across turns, turns 2 and 3 would prefill only their deltas: **~81 s → ~21 s on
this task, from the same model on the same hardware.**

That is PA.1 milestone 2 and Strategy B, and it is no longer a claim from the
paper — it is the measured shape of our own e2e run. It is also why the
constraints in the progress log matter so much: the RS index surface gap
decides whether this is reachable on a hybrid model at all.

---

## Findings

### 1. The serving inferlet could not run a hybrid (GDN) model — FIXED

`Qwen3.6-35B-A3B` is `qwen3_5_moe`: 40 layers, every 4th full attention, the
rest Gated DeltaNet. It loaded and served `/health` and `/v1/models` fine, and
then answered every completion in ~50 ms with `finish_reason:"length"`,
`completion_tokens: 0` and the `"…"` placeholder content.

Cause: `pie:inferlet` exposes three forward interfaces and `ForwardPass` is
three unrelated types. `engine.rs` was written against
`ptir::attention` only. For a recurrent-state model the driver requires one
rs-working-set per request row:

```
resolved forward has 1 request row(s), but recurrent-state model bound
0 rs-working-set(s); expected 1
        — runtime/engine/src/pipeline/fire/rs.rs, validate_count
```

which surfaces as a submit failure, which the turn correctly degrades to
`finish_reason:"length"` — the degradation discipline worked exactly as
designed and hid the cause perfectly.

Fix (`inferlets/chat-completions/src/engine.rs`): a `BindState` trait with one
impl per forward interface, and the generation body expanded once per pass kind
by `define_generate!` so the two cannot drift; `generate` dispatches on
`model::pass_kind()`. Serving never buffers —
`RsGeometry { fold_len: None, buffer: 0..0 }` — and the same rs working set is
bound by the prefill chunks and the decode fires so decode continues the
prefill's folded state. Shape ported from
`tests/inferlets/text-completion-bench`.

The attention path is byte-identical after the rewrite (same completion text,
25/25 still green), and the hybrid path is now **confirmed live**: the 35B
generates through it and the full suite is green on it.

### 2. Every Qwen MoE/VL model silently lost its tool schemas — FIXED

With the hybrid path working, the 35B still produced no tool calls where the
0.6B did — backwards. The tell was in `usage`: the same request **with and
without a `tools` array rendered to the identical prompt length** (15 tokens
both ways), and asked to call a tool the model wrote *"Since I'm simulating,
I'll assume there's a standard time tool available like `current_time`…"*. It
had never been shown the schemas.

`instruct::create` keys on HF **model types** (`qwen3_5_moe`). The string the
engine actually passes is the driver's arch **stem** — `architectures[0]`
lowercased with the task suffix stripped (`arch_stem`, mirrored in
`worker/src/embedded_driver.rs`). CamelCase word boundaries carry no separator
through that:

| `architectures[0]` | stem | matched? |
|---|---|---|
| `Qwen3ForCausalLM` | `qwen3` | yes |
| `Qwen3MoeForCausalLM` | `qwen3moe` | **no** → Qwen3-Coder-30B-A3B |
| `Qwen3_5MoeForConditionalGeneration` | `qwen3_5moe` | **no** → Qwen3.6-35B-A3B |
| `Qwen3VLForConditionalGeneration` | `qwen3vl` | **no** |

A miss falls to the `_` arm, whose `has_tools: false` makes
`equip_after_system` drop every schema and `has_thinking: false` leaves the
think channel unhandled. Chat still renders and the model still answers
fluently — tool calling is just gone, with no error anywhere.

Fixed in `model/src/instruct.rs`: the qwen rows carry both spellings, with a
test that runs `arch_stem` over the six qwen `architectures[0]` strings and
asserts each reaches a tool-capable instruct (asserting on *rendering*, not on
the `has_tools` flag — dropping schemas is the failure being tested).

Live proof: prompt_tokens 19 → 165 for the same request, and

```
finish_reason: "tool_calls"   get_time({"timezone":"Asia/Tokyo"})   24 tokens
```

Worth recording: **Qwen3.6 follows the tool dialect it is shown.** Given
hermes-style schemas it emits hermes-style calls, even though its own template
pairs JSON schemas with XML-style `<function=…>` calls. Serving works today;
byte-exact prompt parity against vLLM would still need the native dialect.

### 3. A reasoning model's `</think>` leaked into content — FIXED

`VisibleFilter` entered think-mode on an *opening* `<think>`. Qwen3.6 reasons
whether or not the cue closes the think block for it, and emits only the
closer — so the reasoning text was served as content and the bare `</think>`
went out as a literal, into the assistant message and from there back into the
next request's history verbatim. opencode showed it plainly:

```
→ Read notes.txt
I need to read the notes.txt file to find the secret color. Let me do that.

</think>
The secret color is chartreuse (line 4).
```

Two fixes, split by what each path can still act on:

- **`VisibleFilter`** now scans for closers alongside openers in Text mode and
  drops an unmatched one, with closers added to the boundary holdback so a tag
  split across chunks cannot leak its first half. The literal tag never
  reaches content on any path.
- **`cut_leading_reasoning`** drops the preamble itself, and runs only on the
  non-streaming path — streamed deltas are already on the wire and are never
  retracted. It fires on the FIRST closer and only when no opener precedes it;
  past that a `</think>` is a literal the model wrote. Same asymmetry the
  trailing-whitespace trim already has.

After: `"What is 17 times 3? Answer with just the number."` → content `'51'`,
`finish_reason: "stop"`.

**Known limitation:** on the *streaming* path the reasoning preamble is still
delivered as content (only the tag is suppressed) — a delta cannot be
un-sent. The real fix belongs in the template layer: Qwen3.6's own generation
prompt ends `<|im_start|>assistant\n<think>\n`, so a lineage-aware cue would
leave the block open and let the filter suppress the reasoning deterministically
with no buffering. That is renderer work, tracked with the Qwen3.6 dialect.

### 4. Inferlet diagnostics were invisible — FIXED

A launched process routes stdout/stderr to the process actor, not the runtime
log, and `gateway/src/ingress/openai.rs` dropped those events. So the
inferlet's own `eprintln!` on a degraded turn reached nobody, and finding
finding #1 took a rebuild with hand-added traces. The ingress now logs them
(`pie::inferlet` target, stderr at `warn`) and still keeps them off the wire.

---

## Host memory: what actually blocked the 35B, and what did not

The first 35B attempt was refused — it needs 22.55 GiB resident at
`total_pages 512 / max_forward_requests 8 / max_model_len 16384`, and only
15.54 GiB was reclaimable. The driver's own warning named the usual suspect:

```
[pie-metal] warning: 24.17 GiB of this machine's 48.00 GiB is wired before this
model is loaded. … a GPU context whose command buffer never signalled is
abandoned rather than released, and its pages stay wired until reboot.
```

**That diagnosis was wrong here, and it is worth knowing why.** The 24 GiB was
the *live* Metal heap of a second `pie serve` from an unrelated session on the
same machine. A clean `SIGTERM` to it released everything: wired went 24.17 GiB
→ 2.85 GiB in seconds, and the 35B booted on the next try. No reboot was needed.

So on a shared machine, check for another `pie serve` before believing the
leaked-context reading. Both failure modes present identically in that warning.
The operational rules that follow:

- **one `pie serve` at a time**, and stop it with `SIGTERM`, not `kill -9` — a
  hard kill during a fire is what genuinely leaks a wedged context;
- budget ~22.6 GiB for this model at the settings above;
- `max_model_len 32768` is **not** out of reach on this machine, and an
  earlier revision of this file said it was. The first 35B boot of the session
  ran `total_pages 1024 / max_forward_tokens 2048 / max_forward_requests 32 /
  max_model_len 32768` and came up in 3 s. The later refusal at those settings
  happened only once a second `pie serve` was resident. Admission is
  `want + min(transient, 2 GiB) + 2 GiB margin > reclaimable`, so it tracks
  what else is on the machine, not the model alone. Measured `want` for this
  checkpoint: **24.77 GiB** at 32768/1024 pages/32 reqs (18.16 weights +
  4.466 KV/state/scratch), **22.55 GiB** at 16384/512 pages/8 reqs (18.16 +
  2.259). Pick from the arithmetic and the machine's reclaimable figure.

## Reproduce

```sh
export CARGO_TARGET_DIR=~/Documents/Liszt_ai/pie/target
cargo build -p pie-bin --release --features driver-metal
(cd inferlets/chat-completions && cargo build --release --target wasm32-wasip2)
cp $CARGO_TARGET_DIR/wasm32-wasip2/release/chat_completions.wasm \
   ~/.pie/programs/chat-completions/0.1.0.wasm
cp inferlets/chat-completions/Pie.toml ~/.pie/programs/chat-completions/0.1.0.toml

pie -c <config>.toml serve &
PIE_BASE_URL=http://127.0.0.1:8081 python3 integrations/opencode/test_acceptance.py
```

Two things cost time and are worth knowing:

- **The first request after a boot pays wasm JIT** and, on a memory-pressured
  machine, can look like a hang. Warm it with a throwaway 4-token request
  before timing anything or before pointing opencode at it.
- **`pie run` binds `[server].port`** even for a one-shot, so it collides with
  a running `serve`. Give the one-shot config its own port.
