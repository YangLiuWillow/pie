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
| Native suites (`pie-openai-serving` / `pie-model-qwen-3 --features chat` / `pie-gateway`) | 45 · 24 · 40+1+6 — all green |
| Acceptance suite, live, `Qwen3-0.6B` (MLX int4) | **25 passed, 0 failed, 0 warnings** |
| Stock opencode e2e, tool call | **works** — `read` tool call + correct answer |
| Live run on `Qwen3.6-35B-A3B` (MLX 4-bit) | **blocked on host RAM**; the code gap it exposed is fixed (see below) |

**No wire-level first-contact bugs.** Every hard assertion in the 25-test suite
passed on the first live run, including the two fixture replays (req-004 tool
turn, req-005 7.5k-token history replay), tool-call delta atomicity, id
uniqueness across processes, envelope non-leakage and the SSE keepalive path.
The two bugs first contact did find were both *below* the wire (§Findings).

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
well-formed and the text was coherent.

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
25/25 still green). **The hybrid path has not been exercised live** — see below.

### 2. Inferlet diagnostics were invisible — FIXED

A launched process routes stdout/stderr to the process actor, not the runtime
log, and `gateway/src/ingress/openai.rs` dropped those events. So the
inferlet's own `eprintln!` on a degraded turn reached nobody, and finding
finding #1 took a rebuild with hand-added traces. The ingress now logs them
(`pie::inferlet` target, stderr at `warn`) and still keeps them off the wire.

---

## Blocked: the Qwen3.6-35B-A3B live run

The model imports and boots (`19.51 GB of weights bound where they lie`), but
re-admission now fails:

```
this model does not fit the memory this machine has left: it needs 24.77 GiB
resident (18.16 GiB of weights, 4.466 GiB of KV, state and scratch) … and only
21.83 GiB is reclaimable.
```

Two causes, both host-side, neither in this branch:

1. **A second `pie serve` on the same machine** (an unrelated `pie-npr` /
   NPR-4B session) holds its own Metal heap.
2. **Wired pages from abandoned GPU contexts.** The driver reported
   `26.22 GiB of this machine's 48.00 GiB is wired before this model is
   loaded` — a context whose command buffer never signalled is abandoned
   rather than released, survives `kill -9`, and is cleared only by a reboot.
   Every `pie serve` killed mid-flight during this session contributed.

To finish: **reboot**, keep one `pie serve` at a time, and re-run with
`total_pages`/`max_forward_requests` sized for ~21 GiB of headroom. The
first thing to check afterwards is a bare completion — the hybrid binding
above is the untested half.

---

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
