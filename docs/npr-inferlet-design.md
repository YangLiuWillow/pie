# Realizing Native Parallel Reasoning with Pie Inferlets

*Design study, 2026-08-11. Sources: the NPR paper (arXiv 2512.07461), the NPR code release (`Native-Parallel-Reasoner/`), and the `fork/dev` branch of `pie/`.*

> **Phase 1 status (2026-08-11): implemented and smoke-tested.** The inferlet
> lives at `inferlets/npr/` on the `npr-inferlet` branch (worktree
> `../pie-npr`, branched off `fork/dev`). Validated end-to-end against a dev
> `pie serve` build: dummy driver (control flow, 256 tok in 99 ms) and the
> portable CPU driver with stock `Qwen/Qwen3-0.6B` (real decode; fork of 2
> branches via a `primer` test hook, concurrent branch decode, ×degree budget
> charging stopping both branches at the shared cap, textual join +
> `<takeaway>` cue, clean stats/return). Stock 0.6B does not emit the NPR
> format unprompted (expected — that is what the NPR checkpoint is for); the
> `primer` input exercises the fork path structurally. See §9 for how to run
> and the upstream issues found along the way.

## TL;DR

NPR's inference-time execution model maps remarkably well onto pie inferlets — arguably better than onto the patched-SGLang "NPR Engine," because everything NPR had to hack into SGLang's scheduler (fork triggers, format validation, token ledgers, repetition-penalty scoping) becomes ordinary user-space inferlet code. There is exactly **one primitive pie lacks**: the zero-copy KV **join** (NPR stitches branches' KV pages into the merged sequence). On pie today, a mathematically-identical "refill join" can substitute, using explicit position IDs + per-token attention masks that pie's `forward-pass` already supports. A cross-lineage page-graft op is the one engine extension worth adding later.

---

## 1. What NPR actually does at runtime

Verified in the code, not just the paper's abstraction. The model is Qwen3-4B (Instruct-2507 or Non-thinking) with **8 added single-token tags**:

```
<guideline> </guideline> <plan> </plan> <step> </step> <takeaway> </takeaway>
```

added at Stage-2 SFT (`npr-beta/train/utils.py:19-60`; new embedding rows warm-started as the mean of each tag word's ordinary BPE sub-token embeddings). The output schema is a Map–Process–Reduce loop:

```
<guideline><plan>1: ...</plan><plan>2: ...</plan>...</guideline>   # Map: N strategies
<step>1: ...</step> <step>2: ...</step> ...                        # Process: N parallel branches
<takeaway>...</takeaway>                                           # Reduce: synthesis
... (repeat rounds, possibly nested) ...
final answer with \boxed{...}
```

All branching lives in a patched SGLang scheduler
(`npr-rl/verl/workers/rollout/sglang_rollout/sglang/python/sglang/srt/managers/{schedule_batch.py,scheduler.py}`):

1. **Generate with stop tokens** `</guideline>` and `</step>` (`no_stop_trim=True`). Each sequence is ordinary causal decoding; branching happens *between* decode steps at the scheduler level.
2. **Fork on `</guideline>`** (`schedule_batch.py:677-782`, `1600-1645`): a two-step "postponed finish" state machine decodes one extra token to disambiguate, then regex-extracts plan indices (`<plan>\s*([0-9]+...):`). If 1–5 plans: create K brand-new requests, each with prompt = `shared_prefix + "\n<step>\n{i}:"`. KV sharing is free via the radix cache (identical token prefix). Branch isolation needs **no attention mask at inference** — branches are physically separate sequences. Crucially, all siblings get the **same starting RoPE position** (`right_most_pos`, the position right after `</guideline>`; `forward_batch_info.py:333-367`) — positions are "parallel," not sequential.
3. **Merge when all siblings close `</step>`** (`scheduler.py:1562-1660`): the merged request's `prefix_indices` are built by **concatenating each branch's radix-cache KV indices** (`over_all_indices = torch.cat([over_all_indices, suffix_indices])`, line 1606) — i.e. **zero-recompute KV stitching**. Only `"<takeaway>\n"` is prefilled; decoding continues with position = `max(sibling end positions)` (the paper's Algorithm 2 alignment rule, realized as page stitching). The merged request reuses the parent's `rid`, so callers see one continuous stream.
4. **Guard rails**:
   - ≤ 5 plans per guideline (else treated as malformed → stop normally) — `schedule_batch.py:737`
   - nesting depth ≤ 5 (`parent_start_path_idx_stack`) — `schedule_batch.py:1587`
   - no fork if < 1024 tokens of budget remain → branch degrades to sequential — `schedule_batch.py:1586`
   - per-branch tokens charged `× parallel_degree` against the global `max_new_tokens` — `schedule_batch.py:693-697`
   - repetition penalty 1.02 inside `<step>` spans only, reset to 1.0 on merge — `schedule_batch.py:1614`, `scheduler.py:1475,1613`
   - sampling: temperature 1.0, top_p 0.7, `max_new_tokens` 30–40k.

**Training pipeline** (for context): Stage 1 (NPR-Zero) = format-follow RL with DAPO on stock SGLang, schema taught by system prompt (`evals/prompts/npr.txt`); Stage 2 (NPR-Beta) = rejection-sampled self-distillation + parallel SFT with custom block-diagonal attention masks and overlapped position IDs (`npr-beta/train/utils.py:314-463`); Stage 3 (NPR) = native-parallel RL (PAPO: no clip-masking on structural tokens, stop-gradient importance ratio, batch-level advantage normalization) with rollouts from the patched engine. The custom mask/position algorithm (`npr-rl/verl/workers/actor/mv_utils.py`) is used only for **training-side log-prob computation** over the flattened trace — never at inference.

Reported results: up to +24.5% accuracy, 4.6× decode speedup vs. sequential, 100% parallel-trigger rate across 8 benchmarks.

---

## 2. What pie's `dev` branch provides

(All paths as of `fork/dev` @ `94043eb12`; WIT source of truth is `interface/inferlet/`.)

- **`Context::fork()`** — O(1) copy-on-write over a content-addressed page trie (`runtime/src/context/{pagestore,snapshot}.rs`): committed pages refcount-shared in a Patricia trie, only working pages physically copied (GPU D2D). This *is* NPR's fork, minus the need for radix prefix re-matching.
- **Transparent batching** — concurrent decodes from forked contexts (driven by `futures::future::join_all` etc.) are coalesced into shared GPU batches by the per-driver `BatchScheduler` (`runtime/src/inference/scheduler.rs`). NPR's parallel wall-clock speedup comes for free.
- **`forward-pass.input-tokens(tokens, positions)`** — explicit per-token position IDs.
- **`forward-pass.attention-mask(list<brle>)`** — one BRLE visibility row per input token over the context's full KV (`interface/inferlet/core/wit/inference.wit:54-57`; plumbing in `runtime/src/inference/request.rs:348-465`). If omitted, the runtime synthesizes causal masks as `all_true(position + 1)`; pure single-token decode skips masks and attends to all existing KV via page metadata.
- **Tokenizer surface** — `tokenizer.encode/special-tokens()` to resolve the tag IDs; `Generator.stop(&[u32])` for token-ID stop sets; per-step probes (`Logits`, `Distribution{temp,k}`, `Logprobs`) for custom sampling; `ctx.truncate(n)` for rollback.
- **Structural references among shipped inferlets** — `inferlets/best-of-n` (fork ×N + consensus vote), `inferlets/skeleton-of-thought` (plan → parallel elaboration), `inferlets/graph-of-thought` (fan-out + pairwise **LLM-based** aggregation — structurally closest to NPR's Reduce), `inferlets/tree-of-thought`, `inferlets/parallel-generation`, `inferlets/demo-parallel-fork`.
- **Model support** — Qwen3 in both `cuda` and `portable` drivers; host-side chat templates per architecture (`runtime/src/model/instruct/qwen3.rs`).
- Scheduling market (`bid`/`idle`) — `Generator` auto-rebids; `ctx.idle()` around non-GPU waits.

---

## 3. The mapping

| NPR engine mechanism | Pie inferlet realization |
|---|---|
| Stop-token trigger + postponed-finish disambiguation | `Generator.stop([</guideline>, </step>])`; parse decoded text in the guest (regex over plans) — no one-token lookahead hack needed, the inferlet sees full text |
| Fork K requests sharing prefix KV via radix cache | `ctx.fork()` × K, fill `"\n<step>\n{i}:"` into each, decode concurrently with `join_all` |
| Sibling isolation | Automatic — separate contexts |
| Siblings share starting RoPE position | Automatic — every fork continues from the same prefix length |
| **Merge: KV stitching + `<takeaway>` continuation** | **The one gap.** Refill join (§4), or a future engine graft op |
| Takeaway positions continue from `max(branch ends)` | Manual decode loop via the raw `Forward` builder with explicit positions (`Generator` has no position-offset option) |
| Repetition penalty 1.02 inside steps | No repetition-penalty sampler in pie → probe a top-k `Distribution`, apply the penalty guest-side over the branch's seen tokens, sample, `g.accept(&[tok])`. Or skip in v0 — it is a mild anti-collapse tweak |
| Budget ledger (`× parallel_degree`), plan-count/depth caps, pre-branch format validator | Trivial guest code |
| NPR Engine stability fixes (radix double-free, global ledger, undefined states) | Moot — control plane lives in user space; the engine only does paged KV + CoW fork + batching, which pie already does robustly |

---

## 4. The refill join

Keep branch 1's context (prefix + step-1 already in KV). For each sibling *j* = 2..K, re-prefill its `"\n<step>\n{j}:" + step_j` tokens into that context in one `forward-pass` with:

- **positions** restarting at `prefix_end + 1` (overlapping step-1's positions, per the paper's Algorithm 2), and
- a **BRLE mask row per token** exposing only `[0 .. prefix_end] ∪ [own step's tokens so far]`.

Because K/V vectors depend only on token content, position IDs, and the visible attention set — and all three are identical to what branch *j* computed during its own decode — the resulting KV is equivalent to NPR's stitched pages (up to kernel nondeterminism). Then fill `"<takeaway>\n"` and decode with positions `max_end + t`; single-token decode attends to all KV by default, which is exactly the Reduce semantics.

**Cost vs. NPR Engine**: one extra chunked prefill of Σ(sibling step lengths) tokens per join — compute-bound and small relative to the decode wall-clock just saved. It disappears entirely if pie later adds a graft primitive.

**Proposed engine extension (phase 3)**: `context.adopt_pages(other: borrow<context>, page_range)` — refcount-share another lineage's *physical* pages under new chain entries. Sound because positions are already baked into KV; only the content-addressed chain hashing needs extending. This makes the join O(1) and closes the last efficiency gap with the NPR Engine.

---

## 5. Inferlet control-loop sketch (Rust)

```rust
const MAX_PLANS: usize = 5;
const MAX_DEPTH: usize = 5;
const MIN_FORK_BUDGET: usize = 1024;

async fn npr_round(ctx: &mut Context, depth: usize, ledger: &mut Budget) -> Result<Flow> {
    // Decode until </guideline>, </step> (shouldn't appear at this level), or EOS.
    let seg = ctx.generate(top_p(1.0, 0.7))
        .stop(&[GUIDELINE_END, STEP_END])
        .max_tokens(ledger.remaining())
        .collect_text().await?;

    let Some(plans) = parse_plans(&seg) else { return Ok(Flow::Terminal(seg)) };

    if plans.len() > MAX_PLANS || depth >= MAX_DEPTH || ledger.remaining() < MIN_FORK_BUDGET {
        return Ok(Flow::Sequential);   // NPR's degrade-to-sequential path
    }

    // Map: fork K branches off the shared prefix (post-</guideline>).
    let branches = plans.iter().map(|i| {
        let mut b = ctx.fork()?;
        b.fill_tokens(encode(&format!("\n<step>\n{i}:")));
        // Process: decode each branch to </step>; ledger charges tokens × K.
        // (Nested <guideline> inside a step => recurse with depth + 1.)
        decode_step(b, ledger, depth)
    });
    let steps: Vec<BranchResult> = future::join_all(branches).await;

    // Reduce: refill join into branch 1's context, then takeaway.
    let mut merged = steps[0].ctx;
    let max_end = refill_join(&mut merged, &steps[1..]).await?;   // explicit positions + BRLE masks
    fill_and_decode_takeaway(&mut merged, max_end, ledger).await?; // raw Forward loop, pos = max_end + t

    *ctx = merged;
    Ok(Flow::Continue)   // next segment may open another <guideline>
}
```

---

## 6. Gaps, risks, verification points

1. **No cross-lineage page graft** — refill join works today; the graft op is the one WIT/runtime extension for O(1) joins (§4).
2. **`Generator` stops at sequential positions** — the merged-context phase drops to the raw `Forward` builder. A `position_offset` option on `Generator` would be a small, clean SDK addition.
3. **Model loading** — the NPR checkpoint is Qwen3-4B with embeddings resized (+8 tags, padded to a multiple of 64). Verify pie's weight-loader accepts the enlarged vocab and that `tokenizer.special-tokens()` / `encode()` resolve the tags to single IDs. Prompting: Qwen chat template + the `evals/prompts/npr.txt` instruction as user-message preamble (NPR does not modify the chat template).
4. **BRLE mask subtlety** — the runtime's synthesized causal mask is `all_true(position + 1)`, i.e. it assumes position == KV index. After a join, positions overlap, so **every subsequent multi-token prefill into that context must pass explicit masks** sized to actual KV length (single-token decode is safe). This is the subtlest correctness point of the whole port. Unit-test on the dummy driver: refill-join KV/logits vs. a straight-line decode oracle.
5. **Fidelity of the join matters** — a "textual join" shortcut (refill concatenated steps causally, sequential positions, no masks) runs today with zero low-level code but is off-distribution for the RL'd model: NPR was trained (parallel SFT + PAPO) with overlapped positions and step isolation, and the paper's §4.2 pseudo-parallelism analysis is precisely about baselines silently degrading. Build the faithful version; A/B the shortcut on AIME25 avg@8 using the NPR eval harness.
6. **Nested guidelines** — a branch may open its own `<guideline>` inside a `<step>`; handle by recursion (depth-capped), mirroring NPR's `parent_start_path_idx_stack`.

---

## 7. Why this is a genuinely good fit

Section 2.5 of the paper is a catalog of pain from pushing branch/merge logic into a monolithic scheduler: radix-cache double-frees under branching, underestimated global token budgets, undefined states from illegal parallel schemas, scoped repetition penalties. Every one of those becomes ~10 lines of guest code in an inferlet, because the control plane (schema validation, budgeting, fork policy) moves to user space while the engine does only what it already does well (paged KV, CoW fork, continuous batching). Longer term, pie-as-rollout-engine (cf. the `liu/pie-rl` branch) would let PAPO's structural filtering and branch bookkeeping live in the same inferlet that serves inference, instead of in a vendored SGLang fork that must be rebased.

## 8. Suggested plan

- **Phase 1 — control loop + textual join** (~a day): fork/decode/join with plain causal refill; validates model loading, tag stop-sets, plan parsing, budget ledger end-to-end. Templates: `inferlets/best-of-n`, `inferlets/graph-of-thought`. **✅ Done — see §9.**
- **Phase 2 — faithful refill join**: explicit positions + BRLE masks, manual takeaway decode loop; dummy-driver unit test against a straight-line oracle; then evaluate with `evals/evaluate.py` (AIME25, avg@8) against the NPR-Engine numbers.
- **Phase 3 — engine extension**: propose/implement `context.adopt_pages` for O(1) joins; optionally add `Generator::position_offset` and a repetition-penalty sampler upstream.

---

## 9. Phase 1 implementation notes

### What was built

`inferlets/npr/` (branch `npr-inferlet`, worktree `../pie-npr`): a single Rust
inferlet (~600 lines) implementing the full NPR control loop —

- **Decode loop** driven token-by-token via `Generator::next_token()` with the
  chat stop set only; NPR tags are detected **byte-level** against a
  `tokenizer.vocabs()`/`special_tokens()` id→bytes table, so both single-token
  tags (NPR checkpoint) and plain-BPE tags (stock models) work, and the tag
  token stays in the stream/context (NPR `no_stop_trim` semantics — pie's
  Generator would otherwise swallow the stop token and hide which one fired).
- **Fork** on `</guideline>`: plan labels parsed like NPR's regex over the last
  guideline block; `Context::fork()` per plan; each branch fills
  `\n<step>\n{i}:` (tags spliced via special-token ids, never plain `encode`)
  and decodes concurrently under `futures::join_all`.
- **Guard rails** as in the NPR engine: ≤5 plans, depth ≤5 (nested guideline
  blocks recurse), fork only if ≥`min_fork_budget` remains, branch tokens
  charged ×degree against the global budget, sequential fallback otherwise.
- **Textual join (phase-1)**: adopt branch 1's context, append sibling branch
  tokens in plan order + `<takeaway>\n`, continue the outer loop. (Sequential
  positions/causal attention — the phase-2 refill join replaces this.)
- **`primer` input** (test hook): text injected as if generated; a primer
  ending in `</guideline>` triggers the fork path immediately, so fork/join is
  testable with a stock model that never emits the format.

### How to run the smoke tests

```bash
# build the inferlet
cd pie-npr/inferlets/npr && cargo build --target wasm32-wasip2 --release

# build the engine (macOS: link workaround, see issues below)
cd pie-npr && cargo rustc --release -p pie-bin --bin pie --features driver-dummy \
  -- -C link-arg=-framework -C link-arg=Accelerate

# combined standalone config: [gateway]/[worker] sections; model.hf_repo must
# be a RESOLVED local snapshot path; boot with PIE_CONFIG=<cfg> pie serve
# client: JSON-over-WS (see scratchpad run_npr.py of the session, or below)
```

Client protocol on this branch (the vendored `pie_client` does **not** work,
see issues): connect to `ws://host:port/v1/ws` with header
`x-pie-identity: tenant/user`; send `ClientMessage` JSON text frames
(`add_program` with the wasm as a **single** chunk, then `launch_process`);
read `ServerMessage` JSON frames (`response`, `process_event` with
`stdout`/`return`/`error`), ignore `{"type":"turn_done"}`.

### Results

- Dummy driver: 256 tokens through the full per-token guest↔host loop in
  99 ms; budget stop; clean JSON return.
- Portable CPU + stock Qwen3-0.6B, no primer: model reasons in its own
  `<think>` style (solves the sample problem), 0 parallel blocks — confirms
  graceful sequential behavior on a non-NPR model.
- Portable CPU + primer + `min_fork_budget=32`, budget 150: forks 2 branches,
  both decode concurrently to exactly the shared budget (152 charged ≥ 150 →
  both stop), join + takeaway cue run, returns
  `parallel_blocks=1, branches_total=2` with the merged trajectory.
- Throughput on CPU ggml is ~0.5–2 tok/s (fine for smoke tests; use Metal
  (`PIE_PORTABLE_METAL=1`) or CUDA for anything real).

### Upstream issues found on `fork/dev` (worth fixing/reporting)

1. **`Context::destroy` traps the instance** — the host `destroy` deletes the
   resource-table entry; the subsequent wit-bindgen handle drop calls the host
   `drop` handler, whose `table.get(&this)?` fails → wasm trap. Fixed on the
   `npr-inferlet` branch in `sdk/rust/inferlet/src/context.rs` (destroy then
   `mem::forget` the handle).
2. **macOS link regression for `pie-bin`** — `worker/build.rs` emits
   `cargo:rustc-link-arg=-framework Accelerate`, but link-args from a lib's
   build script don't propagate to the final binary (post "thin bin shells"
   refactor), so `pie-bin` fails to link ggml's vDSP symbols. Workaround:
   `cargo rustc -p pie-bin -- -C link-arg=-framework -C link-arg=Accelerate`;
   real fix: emit the framework links from `bin/pie`'s own build script.
3. **Client/gateway protocol drift** — the vendored Python `pie_client` sends
   msgpack frames; the new gateway ingress parses JSON only, and its error
   replies are Text frames the client ignores → silent hang. Also, the
   turn-based WS model breaks **chunked** `add_program` uploads: intermediate
   chunks produce no response, so a chunk-turn never terminates, and a second
   frame replaces the first turn's reply stream (single-chunk upload is the
   workaround).

### Still open before the real NPR run

- Pull the NPR-4B checkpoint (Qwen3-4B + 8 added tags, resized embeddings) and
  verify pie's weight loader + tokenizer accept it; run with Metal or CUDA.
- Multi-round behavior (`<takeaway>` → next `<guideline>`) and nesting are
  implemented but only exercised structurally, not against an NPR model.
- Per-`<step>` repetition penalty 1.02 (probe top-k distribution + guest-side
  sampling) — deferred, minor.

---

## 10. Phase 2 implementation notes (2026-08-11)

### What was built

The faithful refill join (§4) is implemented and **numerically validated**.
`join_mode` selects `"refill"` (default) or `"textual"` (phase-1 baseline,
kept for A/B). Core pieces:

- **`PCtx`** — a context plus the invariant `position(slot) = slot + delta`;
  `delta` moves from 0 at the first join and re-anchors at every join.
- **Refill join**: drain branch 1's pending tail at its own positions; refill
  each sibling's tokens at positions restarting at `p_fork` with 4-run BRLE
  hole rows `[0, fork_slots, hole, own+1]`; fill `<takeaway>\n` at
  `p_fork + max(branch extents)` with slot-causal rows; keep the last cue
  token buffered and re-anchor `delta`. Chunked at 512 tokens/pass.
- **SDK additions** (`sdk/rust/inferlet`): `Forward::positions()` (position
  override for page-owning auto-inputs), `Forward::pass_speculation()`,
  `Generator::position_offset()` (decoupled decode + auto slot-causal mask
  rows), `Generator::disable_pass_speculation()`, `Context::take_buffer()`.
- **Refill-mode restriction**: forks only at depth 1 (exact nested refill
  needs per-token position+visibility records; NPR trajectories are
  overwhelmingly flat sequences of depth-1 rounds). Textual mode keeps
  nesting.
- **`selftest=true`** — the numeric oracle: an isolation matrix probing the
  next-token distribution at a refilled sibling's last token. All six
  equivalences hold **exactly** (TV = 0.0000 on Qwen3-0.6B/portable-cpu):
  positions plumbing, mask plumbing, 4-run rows, hole isolation at natural
  positions, refill after a sub-page sibling, refill after a page-spanning
  sibling. The causal control (sibling visible) differs by TV ≈ 0.22 —
  proving the masks bite.

### Engine bugs found and fixed on the way (all on `npr-inferlet` branch)

4. **Portable driver used position ids as KV write indices**
   (`driver/portable/src/plan.cpp`): `kv_idxs = physical_idx(..., pos_i)` —
   overlapped-position refills silently **overwrote live prefix KV**; and a
   validation `pos >= seq_len` rejected forward-shifted fills. Fixed: write
   target is the token's *slot* (`kv_before + i`); positions are RoPE-only.
   The CUDA driver almost certainly shares this assumption — audit before
   GPU runs.
5. **Portable driver clamped custom mask rows at `position`**
   (`build_attn_mask_f16`): custom BRLE visibility was truncated to
   `[0, p_i]`, hiding all KV slots past a token's (compressed) position.
   Fixed: custom rows are slot-space; the runs alone define visibility.
6. **Committed-page position monotonicity** (`runtime/src/context.rs`): the
   commit check `pos > max_committed_position` rejected overlapped commits.
   Relaxed for tokens carrying an explicit mask (page hashes and restore
   replay already store `(token, position, mask)` verbatim, so non-monotonic
   positions round-trip); watermark update made monotone.
7. **Host forward failures return an empty *successful* output**
   (`runtime/src/api/inference.rs::FutureOutput::ready`): a driver-side
   error is logged at `warn` and surfaced to the guest as `Ok` with no
   tokens — lineage is silently skipped and later commits fail with
   confusing errors. NOT fixed (needs WIT surface for late errors);
   documented. Check `future output failed` in server logs when debugging.
8. **Stale run-ahead speculation vs destroyed branch contexts**: the chain
   extender's staged single-token passes for a destroyed sibling context can
   fire in the same batch as the join's refills; `speculator.rs` documents
   the race as "benign", but one crash in `build_qwen3_graph` (ggml reshape
   assert) was observed under exactly this timing. Mitigated guest-side:
   NPR disables pass-level speculation on every pass. The engine-side race
   deserves a real fix (drain staged entries before batch build).

### Validation runs (portable driver, stock Qwen3-0.6B, CPU)

- `selftest=true`: **pass** (all TVs 0.0000; control 0.2186).
- Primer fork, budget 500, `max_step_tokens=60`: 2 branches decoded
  concurrently, refill join (1 + 67 + 4 token passes at the logged
  slot/position splits), then ~230 post-join takeaway tokens decoded through
  `position_offset` — the full decoupled-position pipeline on a real model.
- Dummy driver: structural run green (fork → join → return, 60 ms).

### GPU bring-up (2026-08-12, RunPod H200, NPR-4B checkpoint)

The real NPR-4B model runs end-to-end on pie's CUDA driver. Setup via
`inferlets/npr/pod-setup.sh`; the fp32 FSDP-merged checkpoint must be cast
to bf16 first (`convert_bf16.py` pattern — pie's loader does not cast).

- **GPU selftest: pass** — positions/masks exact (TV 0.0000); refill
  equivalences within flashinfer-vs-causal kernel numeric noise
  (0.001–0.03); control 0.55.
- **True trajectories (no primer)**: the model emits its own
  `<guideline>/<plan>/<step>` structure. Paper case-study domain problem →
  `(2,12) ∪ (12,102)` (matches Table 6). AIME 2025 I/1 → `70` (correct):
  2 parallel blocks, 5 branches (3+2), all steps closing naturally on
  `</step>`, multi-chunk refill join of 1.5k-token siblings, 4,005 tokens
  in 17.4 s.
- **Two more engine bugs found on GPU**:
  9. `driver/cuda`: prefill-only batches (zero sampling rows) ran the
     lm_head over all N tokens into `ws.logits` sized `[max_logit_rows,V]`
     where `max_logit_rows = R0` (request cap) → OOB GEMM → CUDA 700 for
     prompts > R0 tokens. Fixed: skip the logits tail when
     `num_logit_rows == 0 && !is_pure_decode` (commit `8de3f5e`).
  10. Round-1 guidelines always fell back to sequential: the rendered
     segment sometimes lacks the literal `<guideline>` open (the model may
     open with a short multilingual preamble; tokenizer byte-table renders
     of added tokens are not fully trustworthy either), so `parse_plans`'s
     `rfind` anchor failed. Fixed guest-side: scan from segment start when
     the anchor is absent (commit `e7a45dc`).
- Temp-1.0 sampling gives high run-to-run variance (occasionally a short
  non-format answer); quality/speed A/B vs textual join and vs sequential
  needs an avg@8 eval — the natural next step.

### Phase-2 leftovers

- Nested (depth ≥ 2) refill joins — needs per-token position/visibility
  records through `run_branch`; do when an NPR trajectory actually nests.
- `max_step_tokens` and `primer` are test hooks; the real NPR checkpoint
  terminates steps itself.
- KV for joined content is *recomputed* (one chunked prefill per sibling);
  the O(1) page-graft engine op (§4) remains the phase-3 upgrade.
