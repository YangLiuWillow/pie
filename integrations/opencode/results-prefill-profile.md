# Where pie's time goes on Metal — prefill profile, 2026-08-12

*Machine: `Lius-MacBook-Pro`, 48 GB, Metal. Both stacks serve the same
`mlx-community` checkpoints, one server at a time. Tool:
`integrations/opencode/profile_prefill.py`.*

**All numbers below are at MATCHED config** (see the parity table). An earlier
revision of this document measured pie in a KV-starved configuration and reached
two conclusions that did not survive matching — both are corrected at the end
rather than deleted, because the way they failed is the transferable part.

## Config parity — do this before quoting any ratio

Read what each stack *actually booted with*, not what you asked for.

| knob | vLLM-metal | pie (now) | pie (before) |
|---|---|---|---|
| checkpoint | `mlx-community` 4-bit | same artifact | same |
| compute dtype | `torch.bfloat16` | `bfloat16` | ✓ |
| `max_model_len` | 16384 | 16384 | ✓ |
| graph capture | `enforce_eager=False` | n/a on Metal | ✓ not crippled |
| prefill chunk | `max_num_batched_tokens=2048` | `max_forward_tokens=2048` | ✗ 1024 |
| KV block | `block_size=16` | `kv_page_size=32` | (immaterial — tested) |
| **KV pool** | **193,536 tokens** | **65,536 tokens** | ✗ **16,384** |

`total_pages = 512` was this repo's inherited default and it is **KV-starved**:
raising it to 2048 is worth **1.73× on prefill for free**. It was never
memory-forced — pie's activation pool sat at 24 MB of a 1024 MB budget.
`run_pie_opencode.sh` now generates the matched values by default
(`PIE_TOTAL_PAGES`, `PIE_MAX_FORWARD_TOKENS`).

## Method: fit a line, don't divide an aggregate

"pie does 186 tok/s, vLLM does 1183" cannot localize a bottleneck: it folds
fixed per-call cost together with per-token compute, and those are fixed by
unrelated work. So the profiler sweeps prompt length and reads the **slope**:

```
ttfc(n) = fixed_cost + n / marginal_rate
```

Read the **per-segment marginals**, not the fitted intercept — on a convex curve
least-squares drives the intercept negative and it means nothing.

## Result

Marginal prefill rate per context segment, matched config:

| | **Coder-30B (MoE)** | | | **Qwen3-0.6B (dense)** | | |
|---|---:|---:|---:|---:|---:|---:|
| segment | pie | vLLM | gap | pie | vLLM | gap |
| ~470→1310 | 692 | 2491 | 3.6× | 3221 | 11552 | 3.6× |
| ~1310→2560 | 449 | 1662 | 3.7× | 1868 | 7974 | 4.3× |
| ~2560→3810 | 335 | 1511 | 4.5× | 1339 | 6145 | 4.6× |
| ~3810→5060 | 256 | 1135 | 4.4× | 983 | 4740 | 4.8× |
| **mean gap** | | | **4.1×** | | | **4.3×** |

Fixed per-call cost: **pie ~0 ms, vLLM ~117 ms.**

## The bottleneck

**A uniform ~4× per-token deficit in the Metal forward pass.** The striking
thing is how *flat* it is across everything that could have explained it:

- **Same gap dense and MoE** (4.3× vs 4.1×). A 0.6B dense model has no expert
  routing, no grouped GEMM, and a trivial working set — and the gap is
  identical. **The MoE path is not the problem.**
- **Same gap across 50× of model size** (0.6B vs 30B).
- **Mildly worse with context** on both stacks (3.6× → 4.4×/4.8×), so pie's
  attention scaling is slightly behind but is not the story.

Ruled out by measurement:

- **Not per-call overhead.** pie ~0 ms against vLLM's ~117 ms — **pie wins that
  axis outright.** Nothing about wasm instantiation, rendering, address hashing
  or dispatch is where the time goes; optimising there recovers nothing.
- **Not prefill chunk size.** 1024 → 2048 → 4096 moves mid-length prefill ~18%
  and converges. Worth taking; not the gap. The OpenHands CUDA lever (12.0k →
  24.1k tok/s via "chunk 2048 + 64-row tiles") **does not transfer at that
  magnitude to Metal.**
- **Not KV page size.** 16 vs 32 is immaterial.
- **Not MoE.** See above — this is the conclusion that reversed on matching.

So it is the general per-token GEMM/attention path in the Metal driver, and it
is a driver task rather than an integration one.

## End-to-end, matched (Coder-30B, 6-turn agentic replay)

| turn | pie A | pie B | vLLM |
|---:|---:|---:|---:|
| 1 (cold) | 37.13 s | 26.54 s | 6.76 s |
| 2–6 (steady) | 27–43 s | ~4.0 s | ~0.67 s |
| **total** | **203.44 s** | **46.41 s** | **10.12 s** |

Comparable metrics (the totals are *not* comparable — pie generated 96 tokens
per turn, vLLM 11):

| | pie B | vLLM | ratio |
|---|---:|---:|---|
| cold prefill | 296 tok/s | 1216 tok/s | 4.1× |
| steady-state ttfc | 1.90 s | 0.46 s | 4.1× |
| steady-state decode | 46.2 tok/s | 51.7 tok/s | **1.12× (near parity)** |
| prefix reuse | ~97.5% cached | 78.9% hit | both work |

Strategy B over Strategy A, matched: **4.4×**. Matching the config was itself
worth 1.50× on arm A and 1.31× on arm B.

**Decode is at near-parity; prefill is 4.1× behind.** That is the whole story in
one line.

## Runtime profile: is that 4× GPU kernels, or pie's orchestration?

The sweep above is black-box — it says *the per-token cost is 4× worse*, not
*where in the process that time is spent*. Sampling the live server settles it.

`sample <pid> 10` against `pie serve` during a ~9k-token prefill on the
Coder-30B:

- **95%+ of process samples are parked** — `psynch_cvwait`, idle rayon workers,
  parked tokio workers, `__workq_kernreturn`.
- The one thread doing anything is the Metal driver thread, and its deepest
  frames are:

```
pie::metal::RawMetalContext::Impl::await_event(unsigned long long)
  -[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:]   (in IOSurface)
    iokit_user_client_trap                                     (in IOKit)
```

**The host is blocked on a GPU event.** pie's prefill time is GPU execution
time, not orchestration, not wasm, not the engine's scheduling.

**And it is not a synchronisation-cadence problem either.** `mtl4_context.mm`
encodes *all* command buffers for a forward, then calls `commit_and_signal`
**once**, then `await_event` **once** (`mtl4_context.mm:2338-2352`). One sync
per forward, not per layer — so the gap is not death-by-round-trips.

Combined with the black-box result, the localization is:

| candidate | verdict | evidence |
|---|---|---|
| per-call host overhead | **ruled out** | pie ~0 ms vs vLLM ~117 ms |
| dispatch/sync cadence | **ruled out** | one commit + one await per forward |
| MoE expert path | **ruled out** | same gap on a 0.6B dense model |
| attention scaling | **secondary** | both stacks decay; pie slightly worse |
| **GPU kernel throughput** | **this is it** | host parked in `await_event`; uniform ~4× per token |

### An observability gap worth fixing upstream

The Metal driver already measures exactly the right decomposition —
`M0TimingCounters` carries `encode_ms`, `gpu_exec_ms`, `forward_wait_ns`, and
`bf16_conversion_ns` (`context.cpp:1875`) — but it prints them only under
`cfg_.runtime.verbose`, and **that flag is not reachable from an operator
config**: the driver's TOML blob is engine-generated, and the Rust schema
rejects `verbose` in `[runtime]` (`unknown field 'verbose'`). So the one
breakdown that would separate encode time from GPU time on a real workload
cannot be switched on without a rebuild. Worth a `[driver] verbose` passthrough.

Also unrun on this machine: `pie config tune`, which doctor warns about on every
boot ("planner profile none; the forward step has never been timed here"). It is
a provisioning sweep that holds the whole device, so it was not run mid-session,
but it is the obvious next action and may recover part of the gap.

## Cross-check against CUDA

`openhands-integration-updated:integrations/openhands/runpod`, same model, H100,
fairly configured vLLM:

| | pie | vLLM | gap |
|---|---:|---:|---|
| prefill marginal (tuned) | 24.1k tok/s | 44.9k tok/s | **1.86×** |
| c1 in-call tok/s | 147.4 | 160.4 | 1.09× |
| c8 s/iter, matched instances | 4.26–4.68 | 3.90–4.68 | 1.00–1.08× |

**The prefill gap is not a Metal artifact — it is 1.86× on tuned CUDA.** Metal
widens it to ~4×. Consistent direction across two backends and two harnesses.

## Two conclusions that reversed, and why

Both came from measuring pie starved while vLLM was not. Recorded because the
failure mode is more useful than the numbers.

1. **"The gap is a flat ~6× multiplier."** It was 6.4× starved and is 4.1×
   matched — and matched it *grows* with context rather than being flat. The
   flatness was an artifact of a bottleneck that dominated everything else.
2. **"MoE adds 1.44× on top of a 4.4× dense baseline."** Matched, MoE adds
   **0.94× — nothing.** The starved pool penalised the 30B's large KV footprint
   far more than the 0.6B's small one, manufacturing an architecture-shaped
   difference out of a memory-sizing one. This one had a plausible mechanism
   (OpenHands found MoE GEMM tactics were *their* main CUDA prefill lever), which
   is exactly why it was worth re-deriving rather than believing.

Also corrected: this repo cited OpenHands measuring pie **~26% faster** than
litellm+vLLM. That baseline was deliberately crippled (`--enforce-eager` + an
untuned MoE kernel); their fair rerun puts pie at **parity to ~9% behind**.

## Three cache traps, none of which announced itself

Every one was caught by a number being *implausible*, not by anything failing:

1. **Bench warm-up primed the prefix cache** by sending turn 1's own messages —
   converting turn 1 from a cold prefill into a hit, and only on arms that *have*
   a cache. Read as 0.338 s ttfc, ~22k tok/s.
2. **The profiler's own sweep was nested prefixes.** Building longer prompts by
   repeating one filler makes every prompt a literal prefix of the longer ones.
   Read as **80,543 tok/s marginal on a 30B laptop**. Fixed with a *leading*
   nonce — a trailing one leaves the shared part in front.
3. **`cached_tokens` is unpopulated by vllm-metal** — it reports 0 while its own
   logger reports 78.9% hits. Trusting the usage field would have concluded
   "vLLM has no prefix caching" and flattered pie by ~4×.

## What to do with this

1. **Take the config fix everywhere.** `total_pages = 2048` is free and worth
   1.5× end-to-end. Caveat from the CUDA work: the chunk lever *inverts* under
   concurrency (chunk 2048 at c8 shrank the pool and collapsed), so this is a c1
   recommendation.
2. **Stop optimising this integration for prefill wins.** Strategy B's 4.4×
   against pie's own baseline does not close a 4.1× kernel gap. The session
   work's remaining value is what APC structurally cannot express — B-2 in-place
   context editing, B-3 subagent forking.
3. **If the gap is worth closing it is a driver task**, and the dense/MoE
   equivalence says start at the general per-token path, not the expert GEMMs.

## Reproduce

```sh
python3 integrations/opencode/profile_prefill.py \
  --base-url http://127.0.0.1:8080 --label pie --model coder30b \
  --repeat 2 --words 400,1200,2400,3600,4800
```

---

# B-2 (in-place context editing): design settled, one driver gate open

*2026-08-12. Foundation built and tested; engine plumbing gated on a driver
capability that must be verified before it can be trusted.*

## The design, and why it is masking

**Mask, do not delete.** `WorkingSet::discard` removes whole pages, but RoPE is
baked into K *at write time*, so the surviving tail still encodes its original
absolute positions. Renumbering densely breaks every relative distance against
that tail. Leaving a positional gap instead is rejected by the driver, which
requires `position_id < seqlen` — "position is outside its request KV extent"
(`batch/forward.cpp`). Masking changes neither the KV extent nor any position,
so both constraints hold trivially.

**The cost is affordable only because of the session design.** A dense mask is
`[lanes, kv_len]` bools per fire. On a resumed turn the fire spans the *delta*
(~190 tokens on our bench), not the history, and lanes=1 — so the mask is
kilobytes. A cold turn has nothing stale to mask and never pays it. The slot
itself is pre-allocated at `max_forward_tokens × total_pages × kv_page_size`,
which is **128 MB** at our config.

**It is opt-in per request** (`pie_context_policy`). A server that silently
stops attending to tokens the client sent is serving a different context than it
was given — the same silent-divergence class as every other bug found this
session. The wire stays full-history; this is one additive field, not the delta
wire.

## What is built and tested

`pie-openai-serving::context_policy` — `SpanKind`/`Span`, the `ContextPolicy`
wire type, and `spans_to_mask`, with 6 tests covering: absent policy is a no-op,
newest-N retention, **never masking system/user/assistant turns**, a minimum
span size, spans past the retained length being unmaskable, and keep-more-than-
exist. 70 native tests green.

## The open gate

`runtime/engine/tests/inferlets/ptir-prefill-e2e` exists as a **reproducer for a
driver-level gap** in exactly this mechanism:

> The device-geometry AttnMask dense-pack computes `lanes = qo_indptr.size() - 1`
> = number of SEQUENCES and packs ONE mask row per lane. It is DECODE-SHAPED: it
> cannot express an `[N_query, KV]` prefill mask (a single sequence with N query
> rows collapses to lanes=1 + a garbled stride).
> ⇒ Variable-length prompt prefill on the ptir path is UNBUILT at the driver level.

Metal uses the same convention — "byte per lane, 0/1 — the dense `[lanes,
stride]` convention" (`pipeline/descriptor_resolve.hpp:115`).

For B-2 the lanes=1 collapse is **not itself a problem**: every query token
wants the *same* mask, so one row per lane is the shape we want. What is unknown
is whether the stride is handled correctly on Metal today — the note is dated
2026-07-09, describes the CUDA `executor.cpp`, and says the result was
incoherent output rather than an error.

**Do not build on this until the reproducer is run on Metal.** A mask that is
silently mis-strided produces fluent, wrong output — the failure mode this whole
integration is built to avoid, and one that no test here would catch.

Second constraint, from `pipeline/fire/geometry.rs:36-39`: a mask-carrying fire
is **kept solo by the scheduler**. Harmless for our single-row workload;
disqualifying for batching under multi-tenancy, which is worth knowing before
B-2 is proposed as a multi-tenant win.

---

# Where the remaining speed is — evidence, and one tested dead end

## The decomposition that matters

| | pie | vLLM-metal | gap |
|---|---:|---:|---|
| steady-state **decode** | 46.4 tok/s | 52.0 tok/s | **1.12× — parity** |
| **prefill** | 297 tok/s | 1214 tok/s | **4.1×** |

Same weights, same 4-bit checkpoint, same unified memory, same box. Decode at
batch-1 is bandwidth-bound — you stream the whole model per token — and pie is
at parity there, so weight loading, dequantization, KV reads and the general
plumbing are all fine. Prefill is compute-bound, and that is where 4× goes
missing. **The deficit is specifically in the large-M path.**

And it is **not** the mixture: a 0.6B **dense** model shows the same 4.3× gap.
No routing, no expert GEMM, no long context — and still 4× behind.

## The hypothesis this supports

`driver/metal/src/device_tuning.hpp` is a decode-tuned table. Its constants are
framed in decode units throughout — crossovers "in rows per request", tok/s "at
eight lanes", "keeps a fleet of decodes on the per-row kernel" — and nothing in
it cites a prefill sweep. `qmm_min_batch = 8`,
`sdpa_tile_min_rows_per_request = 32`, `moe_tile_wide_per = 1 << 24`.

OpenHands found exactly this shape on CUDA and got ~2× from prefill-specific
tile work (12.0k → 24.1k tok/s), where the default turned out to be a decode-
shaped 128×16 tile.

## Tested and dead: the 64-row MoE tile

`moe_tile_wide_per = 1 << 24` hard-disables the wide MoE row tile, and OpenHands'
CUDA lever was "chunk 2048 + **64-row tiles**", so this looked like the same bug.
Swept `PIE_METAL_MOE_TILE_WIDE_PER=64` on the Coder-30B:

| prompt | default | wide=64 |
|---:|---:|---:|
| 1309 | 1.824 s | 1.911 s |
| 2557 | 4.599 s | 4.739 s |
| 3805 | 8.327 s | 8.433 s |
| 5053 | 18.417 s | 13.371 s |

**Inconclusive, trending negative.** The first three points are 2–5% worse; the
apparent win at 5053 is the point that has been noisy in every run this session
(18.41 s and 15.81 s measured for *identical* configs). Not a win, and consistent
with the chunk-size lever also failing to transfer from CUDA.

This is also the second CUDA lever that did not carry over, which is itself
information: Metal's gap is not the same gap CUDA had.

## Where to look next, in priority order

1. **Microbenchmark pie's quantized GEMM against MLX's, at prefill shapes.**
   This is the decisive experiment and nobody has run it. vLLM-metal *is* MLX, so
   "why is pie 4× slower than MLX on the same weights and hardware" is directly
   answerable by timing `quantized_matmul` at M=2048 against pie's kernel at the
   same shape. If MLX wins by 4× there, the answer is the kernel and the fix is
   to match its tiling or adopt it. If they tie, the gap is dispatch/layout/
   staging and the microbenchmark says so. Either outcome ends the guessing.
2. **Use the 0.6B dense model as the optimization target**, not the 30B. It shows
   the same 4.3×, boots in seconds, and removes routing, expert GEMMs and long
   context from the picture entirely.
3. **Sweep the dense-GEMM and attention knobs in the prefill regime**, since the
   table was built for decode: `PIE_METAL_QMM_BN_CROSSOVER_TG` (the BN=16→32 tile
   crossover, default 160), `PIE_METAL_SDPA_TILE_MIN_ROWS` (32),
   `PIE_METAL_SDPA_MMA`, `PIE_METAL_FP16_QMM`. All are env vars — no rebuild.

## What is NOT worth more effort

- **Per-call overhead** — pie already wins it (~0 ms vs vLLM's ~117 ms).
- **Chunk size** — swept; ~18% at mid lengths, converges.
- **MoE tiles** — swept; see above.
- **Config sizing** — done, and it was worth 1.73×.
- **Anything in this integration.** Session KV reuse took pie 6.6× faster than
  itself and moved the vLLM gap from 6.4× to 4.1×. The rest is not here.

## Metal tuning knobs: three more tested, all neutral

Following the CUDA finding that a decode-shaped **128×16** default tile cost
~73% on prefill (a sweep found 128×128×128 cluster 1×2 at 23.6k vs the default's
13.6k), the Metal analogues were swept on the 0.6B dense model:

| | 1308 tok | 2556 tok | 5052 tok |
|---|---:|---:|---:|
| default | 0.358 s | 1.004 s | 3.229 s |
| `PIE_METAL_QMM_BN_CROSSOVER_TG=8` (BN=16→32 sooner) | 0.359 s | 1.006 s | 3.228 s |
| `PIE_METAL_FP16_QMM=1` | 0.359 s | 1.007 s | 3.233 s |

Identical to the millisecond. With the earlier `MOE_TILE_WIDE_PER` sweep
(neutral-to-negative) and the chunk-size sweep (~18%, converges), that is **four
tuning levers tested and none of them touches the gap**.

*Caveat:* results this identical are also consistent with the env vars not
reaching the driver at all. Nothing logs the resolved tuning table, so this
could not be confirmed — treat these as "no observed effect" rather than "the
knob does nothing".

**Read together with the CUDA record, the conclusion is that Metal's deficit is
not a tuning-constant problem.** On CUDA the constants *were* the problem and
tuning bought 2×. Here the exposed constants do nothing, which points at the
kernel or the dispatch/layout rather than at a selection heuristic — and makes
the MLX microbenchmark the decisive next step rather than one option among many.

---

# A1 result: prefill is **99.95% GPU execution**. The host is not involved.

The Metal driver was already computing an encode/wait split per forward and
discarding it. `PIE_METAL_TIMING=1` now surfaces it (additive, env-gated,
default off; `runtime.verbose` remains unreachable from an operator config
because the driver's TOML blob is engine-generated).

One 2,556-token prefill on Qwen3-0.6B, Apple M5 Pro (20-core GPU):

| | time | share |
|---|---:|---:|
| encode (host builds command buffers) | **0.33 ms** | **0.05%** |
| forward wait (host blocked on the GPU event) | **657.07 ms** | **99.95%** |
| cpu epilogue | 0.00 ms | 0% |
| bf16 conversion | 0.00 ms | 0% |

**This eliminates the entire host side in one measurement**: command-buffer
encoding, dispatch construction, the epilogue, and dtype conversion together
account for 0.05% of prefill. No amount of host-side work can recover anything.
It also confirms the sampling profile from the other direction — the process was
parked in `await_event` because there is genuinely nothing else to do.

## What that leaves: the kernels, and a roofline to aim at

Same work, same weights, same machine — 3.07 TFLOP of GEMM for 2,556 tokens:

| | time | achieved |
|---|---:|---:|
| pie | 1.024 s | **3.00 TFLOPS** |
| vLLM-metal (MLX) | 0.269 s | **11.40 TFLOPS** |

pie is doing the same arithmetic at **~3.8× lower FLOPS on the same GPU**. That
is not a scheduling, batching, or bookkeeping problem — it is the kernel.

(FLOPs estimated as `2 × params × tokens`, which ignores attention and is
therefore a slight undercount for both stacks equally; the ratio is what
matters.)

## Consequence for the plan

A1 was meant to either localize the gap or eliminate the host, and it did both.
**A2 (the MLX microbenchmark) is now the only open question on the driver
track**, and it is narrower than it was: not "where does the time go" — that is
answered — but "what does MLX's `quantized_matmul` do at M=2048 that pie's
kernel does not". Everything else on Track A can wait on that answer.

---

# A2 result: it is **attention**, not the GEMM

A1 proved the time is GPU-side. A2 asks which kernel. Answer: mostly the one I
was not looking at.

## Method

Prefill cost decomposes as `t(n) = a·n + b·n²` — the linear term is the dense
projections (GEMM), the quadratic term is attention. Least-squares fit over the
5-point sweep for both stacks (worst residual 1.7% for pie, 3.0% for vLLM),
evaluated at n = 2048, on Qwen3-0.6B / Apple M5 Pro:

| | pie | vLLM | ratio |
|---|---:|---:|---|
| **GEMM** (linear) | 310.4 ms | 131.2 ms | **2.4×** |
| **attention** (quadratic) | 403.1 ms | 67.2 ms | **6.0×** |
| total | 713.6 ms | 198.4 ms | 3.6× |

**The fit is validated independently.** vLLM's fitted GEMM term (131.2 ms) lands
within 5% of MLX's directly measured `quantized_matmul` sum for the same
projections (124.6 ms, at 14.48 TFLOPS). A curve fit agreeing with a direct
kernel measurement it never saw is about as good as this kind of decomposition
gets.

Direct MLX reference for attention: `mx.fast.scaled_dot_product_attention` at
the same shape (16q/8kv heads, d=128, causal) is **0.954 ms/layer = 26.7 ms**
for 28 layers. vLLM's 67.2 ms is ~2.5× that, which is the honest price of paged
attention with block tables over a dense fused kernel — so **67 ms, not 27 ms,
is pie's fair target.**

## What this changes

I had assumed the GEMM, and the CUDA record encouraged it — their prefill lever
was tile shapes, and their fused-path tactic sweep found a decode-shaped 128×16
default costing 73%. On Metal that reasoning was wrong in a way no amount of
GEMM tuning would have revealed:

- **Attention is 56% of pie's prefill forward at 2048 tokens** (and only 34% of
  vLLM's), so it is both the larger share *and* the larger gap.
- It also explains a symptom recorded much earlier and never accounted for:
  pie's prefill rate decays with context faster than vLLM's (326→179 vs
  2491→1135 tok/s). That is the quadratic term dominating.
- The two SDPA knobs (`sdpa_tile_min_rows_per_request = 32`,
  `sdpa_mma = true`) are already in their intended state for a 2048-row prefill,
  so this is **not** a selection heuristic — it is the paged prefill attention
  kernel itself.

## The target

Closing attention alone — 403 ms → ~67 ms — takes the forward from 713 ms to
~377 ms, i.e. **~1.9× on prefill**, and is the single largest available win.
Closing the GEMM as well (310 → 131) would reach ~198 ms, which is parity.

Order of work, revised:

1. **Paged prefill attention on Metal.** 6× behind, 56% of the forward. Start
   here.
2. **The quantized GEMM.** 2.4× behind. MLX reaches 14.48 TFLOPS on these
   shapes; pie's implied rate is ~6.
3. Nothing else — A1 closed the host side at 0.05%.

## Reproduce

```sh
~/.venv-vllm-metal/bin/python integrations/opencode/parity/mlx_gemm_roofline.py \
    --tokens 2048 --layers 28 --pie-forward-ms 657
# and the SDPA reference via bench_sdpa(2048, 16, 8, 128)
```

---

# B1: cross-conversation head sharing — done

opencode re-prefills a ~7.2k-token system+tools head for every new session. At
pie's measured 297 tok/s that is ~24 s of pure repetition per conversation, and
it is exactly the cold-start share OpenHands measured as **71% of concurrency-1
prefill** — the part no within-conversation cache can touch.

## Is opencode's head shareable? Partly, and position decides how much

OpenHands built this and it **never fired**: their head hash differed per
instance because `FileEditorTool` embedded the cwd in its description
(`d93f6cffe`). So that was the first thing checked here.

| | chars | shareable by a *prefix* cache |
|---|---:|---|
| system prose | 8,695 | ✅ invariant |
| environment block — `Working directory:`, `Is directory a git repo:`, `Platform:`, `Today's date:` | 953 | ✗ per session |
| tool schemas | 21,188 | ✗ **byte-identical, but downstream of the block** |

The largest invariant chunk — 69% of the head — is unreachable purely because
opencode emits the environment in the *middle* of its system prompt. Reordering
would fix it and is not ours to do: it changes what the model sees.

## What was actually wrong on our side: boundary granularity

Two bugs, both of which made sharing impossible regardless of the content:

1. **Boundaries were one per render op.** opencode's head is a *single*
   `EquipAfterSystem` op, so its only boundary was the whole ~7.2k-token head —
   which never matches between sessions, because the tail differs.
   Fixed with `BOUNDARY_STRIDE = 256`: interior candidates every 256 tokens, so
   wherever two token streams agree, a boundary lands inside the agreement. Same
   idea as vLLM's block-level APC, coarser because each boundary costs a digest
   snapshot rather than a page-table entry.
2. **Only the tip boundary was retained.** The lookup computed every boundary
   and then stored one. A branch's list therefore held nothing another
   conversation could match. Fixed by retaining the whole render's boundary list
   (every entry ≤ the retained length is a valid resume point).

The attempt cap also went: OpenHands caps at 8 because each try is a
`Context::open` engine round trip, but ours is a string compare against an
in-memory list — and a cross-conversation match lands ~2.1k tokens into a
~7.3k-token render, nowhere near the longest boundary, so a cap of 8 would never
reach it.

## Measured

| case | shared | |
|---|---:|---|
| two real sessions, same repo + same day | **7,250 / 7,268** | **99.8%** |
| different repo **and** different date | **1,792 / 7,223** | **24.8%** (the floor) |

The floor matches the 28.2% predicted from the char counts, short by one stride
(a boundary must land on a 256-token multiple).

**The earlier "~24 s per session" estimate was wrong**: it assumed the whole head
would share. The honest figures are ~24 s for a developer working in one repo on
one day, and ~6 s as the guaranteed floor across repos and dates.

## It broke two tests, and both were right to break

`test_divergent_history_misses_cleanly` asserted that an edited history resumes
*nothing*. That held when an edited turn invalidated the only boundary there
was. Now the **system turn ahead of the edit is a genuine shared prefix**, and
resuming it is correct — it is the same mechanism as head sharing. The test now
asserts the real safety property, which is stronger: the resume must stop
*before* the edit (measured: 23 tokens against 33 unedited).

The suite also had to be isolated. Every test shared one `SYSTEM` constant, so
after the first test the prefix was already resident and "first turn" assertions
saw a non-zero `cached`. Each case now gets its own tag. The feature worked well
enough to break the tests that were meant to police it.

Suites: 25/25 acceptance, 5/5 resume, 2/2 head sharing, 70 native.

---

# The attention fix: found, scoped, not yet applied

A2 said attention is 6× behind and 56% of pie's prefill forward. Tracing that
into the Metal driver found the cause, already diagnosed **in the driver's own
comments**, with the fix **already written** — and wired for one model family
that is not Qwen.

## The cause, in the driver's words

`device_tuning.hpp`, on `sdpa_mma`:

> `sdpa_paged_tiled` computes Q Kᵀ and P V as **hand-walked dot products** — it
> was measured at 35.8% of a 2048-token gpt-oss prefill, running near
> **0.5 TFLOP/s while the quantized GEMM one dispatch away reaches ~5.6** on the
> same silicon. The arithmetic is a matmul; issuing it as one is what
> `sdpa_paged_mma.metal` is.

An 11× kernel-level gap on the op that is 56% of our prefill. That is the 6×.

## Why Qwen does not get the fast path

Everything needed exists:

| piece | state |
|---|---|
| `sdpa_paged_mma.metal` | **written**, templated `<T, D, KT, WITH_SINK>` |
| `sdpa_paged_mma_dispatch` in `qwen3_5/decode_dispatch_mb.hpp` | **present** |
| `sdpa_mma()` tuning gate | **on by default** |
| instantiation | **only** `("_sink", bfloat16, bfloat, 64, 16, true)` — gpt-oss |
| PSO slot | **only** `gptoss/kernels.hpp: sdpa_sink_paged_mma` — not in the shared set |
| Qwen selection (`decode_step_mb.cpp:932`) | picks `sdpa_paged_tiled_strided`, never consults `sdpa_mma()` |

And the stated bound on adding a width:

```cpp
// The matrix path stages three tiles of KT*D halves in 32 KB of threadgroup
// memory, which is what bounds the list: adding a width means choosing its KT.
inline constexpr int kSdpaMmaHeadDim = 64;
```

**Qwen3 uses head_dim 128.** The bound is not binding for it:

| | threadgroup memory | |
|---|---:|---|
| D=64, KT=16 (today, gpt-oss) | 6,144 B | fits |
| **D=128, KT=16** | **12,288 B** | **fits in 32 KB** |
| D=128, KT=32 | 24,576 B | also fits |

## The change

Same shape as the CUDA decode fix, which its own porting doc describes as
"three files, ~25 lines. No new CUDA — the prefill kernel is already compiled
and already used for prefill." Here there is no new Metal kernel either:

1. `sdpa_paged_mma.metal` — add
   `instantiate_sdpa_paged_mma("", bfloat16, bfloat, 128, 16, false)`.
2. `kernels/decode_psos.{hpp,cpp}` — a PSO slot for it in the shared set.
3. `qwen3_5/decode_dispatch_mb.hpp` — `kSdpaMmaHeadDim` becomes a supported set
   {64, 128}.
4. `qwen3_5/decode_step_mb.cpp:932` — prefer the MMA PSO when `sdpa_mma()`, the
   head width is supported, and `sdpa_should_tile` already holds. Mirror
   `gptoss/encode.cpp:378`.

**Expected:** attention 403 ms → toward vLLM's 67 ms, prefill 713 ms → ~380 ms,
i.e. **~1.9× on prefill** and the pie-vs-vLLM gap from 4.1× to roughly 2.2×.

## Why it is not applied here

The driver's own warning: the matrix path "depends on the register layout of
`simdgroup_matrix<T,8,8>` … a machine whose layout differs would produce **wrong
numbers rather than slow ones**." A new head width is exactly where that bites,
and this repo has no numerical attention check — the suites here (25/25
acceptance, 5/5 resume) would all pass on subtly wrong attention output, because
they assert on wire shape and cache behaviour, not on logits.

So the honest prerequisite is a numerical guard: run the same prompt through the
tiled and MMA paths and compare logits, or extend `llama_bench`'s greedy gate,
which `device_tuning.hpp` names as the thing that would catch it. That is the
next piece of work, and it should come **before** the four-file change, not after.

## The greedy gate is in place

`integrations/opencode/test_attention_paths.py` compares generated tokens across
attention implementations at temperature 0. Greedy is the sensitive probe:
the engine takes `reduce_argmax` at temperature 0, so any numerical drift large
enough to flip one argmax becomes a visible, exactly-located divergence, and
drift too small to flip an argmax anywhere in a long generation is drift that
does not matter.

Validated in both directions on Qwen3-0.6B:

- **Positive** — the tiled kernel against the per-row kernel (forced with
  `PIE_METAL_SDPA_TILE_MIN_ROWS=1000000`, so `sdpa_should_tile` never fires).
  Two genuinely different attention implementations, **byte-identical output on
  all three prompts** (21 / 323 / 1523 prompt tokens).
- **Negative** — one character flipped inside the longest generation. The gate
  fails, locates it at char 68, prints both contexts, and exits 1.

So the harness distinguishes, and today's two paths agree. When the MMA path
lands, capture a third with it enabled and compare against the tiled capture —
same harness, no changes.

**What this does and does not establish.** It proves the comparison is sensitive
to a single token and that the two *existing* kernels agree. It does not prove
the gate would catch any wrong MMA kernel, because there is no wrong kernel here
to test against — that rests on the argmax-amplification argument above, which
is the same reasoning `device_tuning.hpp` invokes when it names the greedy gate
as the catcher. Three prompts is also a thin set: widen `PROMPTS` rather than
trusting a green run on three.
