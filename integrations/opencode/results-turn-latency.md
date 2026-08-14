# What makes pie slower: a prefill fire whose row count is wrong mod 8

**Date:** 2026-08-13. **Machine:** Apple M5 Pro, 48 GB.
**Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, strategy B
(`opencode-session`), 6-turn canned opencode replay at 7.2–8.2k prompts.

## Result

| | total | steady turn | cold prefill (7211 tok) | steady prefill (189 tok) |
|---|---:|---:|---:|---:|
| before | 30.6 s | 1.53 s | 22.3 s | 1159 ms |
| pool quantization only | 30.7 s | 1.44 s | 22.3 s | 1150 ms |
| **+ aligned prefill chunks** | **18.1 s** | **0.97 s** | **11.4 s** | **~690 ms** |

**1.69× on the whole replay, 1.96× on cold prefill.** All 25 acceptance tests
pass, generated token counts per turn are unchanged, and the gap to vLLM on this
workload narrows from 3.0× to 1.78×.

## How it was found

pie was reusing 7211 of 7403 prompt tokens and still taking 1.53 s per turn.
At 7.4k context a decode step costs ~21 ms, so the ten generated tokens are
~0.2 s; the ~190 fresh prompt tokens should be a fraction more. About a second
per turn was unaccounted for, and nothing outside the guest could attribute it —
the shim, the gateway and the engine each see one opaque call.

So the guest was made to time its own phases (`handler.rs` `phases_ms`,
`engine.rs` `gen_ms`). Turn 3 onward:

| phase | ms |
|---|---:|
| render full history (~7.4k tokens) | 2.6 |
| address every boundary | ~0.0 |
| resume scan | ~0.0 |
| **prefill the 189-token delta** | **1159** |
| cue + first token | 78 |
| decode 10 tokens | 288 |

Rendering, addressing and resuming — the parts a prefix-caching design might be
expected to pay for — cost 2.6 ms together. The second is one prefill fire of
189 rows.

**A wrong hypothesis first, kept because it is instructive.** `pool_pages` grows
~6 pages per turn and becomes the length of the `pages_p` channel; a channel's
shape is part of the program container (`ChannelDecl.shape`), so every turn
handed the driver an unseen container. `decode-rows-probe` measures ~600 ms to
compile one. That predicted the whole second. Quantizing the pool to 256-page
granularity (which strategy A already did, and which strategy B had never got)
changed the prefill by **9 ms**. It did help decode — 31 → 38 tok/s, because the
decode fires' shapes stabilised too — so the change is kept, but it was not the
answer. The measurement is what said so.

## The actual cause

`decode-rows-probe` was widened from decode widths to prefill widths and fired
the exact geometry at a fixed context. It reproduces the serving number exactly
— 189 rows at 7424 context, **1157 ms**, against serving's 1150–1160 ms. So the
whole second is inside one fire.

The cost curve is not monotonic in work. At ctx 2048, **189 rows cost more than
512 rows** (882 ms vs 725 ms). Sweeping the neighbourhood at ctx 7424:

| rows | 184 | 185 | 186 | 187 | 188 | 189 | 190 | 191 | 192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | **579** | 1141 | 1140 | 1144 | 1144 | 1147 | 1148 | **583** | **583** |

189 rows costs 1.97× what 192 rows costs, doing less work. Confirmed on nine
more widths — 193, 194, 196, 198 slow; 199, 200, 207, 208, 215 fast:

> **A fire of `r` rows pays a flat ~560 ms penalty when `r mod 8` is 1–6, and
> does not when it is 0 or 7.**

Three things that rule out the obvious explanations:

- **Not alignment.** 191 is prime and fast; 188 is a multiple of 4 and slow.
- **Not `kv_len`.** Both contexts above are multiples of 8, which confounds
  `rows` with `ctx + rows`. Re-run at ctx **7420** (≡ 4 mod 8): the fast widths
  are still 191 and 192, not 187 and 188. The rule is on the row count alone.
- **Not the schedule bucket.** `m3_schedule_bucket` is `ceil(log2(rows))`
  (`m1_runtime.cpp:598`), so 184 and 191 share a bucket and a stage-cache key.

The penalty is context-independent (+560 ms at both 2048 and 7424), so it is
neither the attention read nor anything that scales with the cache — consistent
with a fixed extra pass rather than more work per token. It also has a lower
threshold: at 12 rows there is no penalty (99 ms) and at 20 rows there is
(554 ms against ~150 ms interpolated).

**The root cause is in the Metal driver and is not fixed here.** What it is
exactly — which kernel or dispatch path a row count of `8k+1 … 8k+6` selects —
is still open, and is the one loose end in this note.

## The fix

`aligned_prefill_chunks` (in `inferlets/opencode-session/src/engine.rs`) splits
every prefill span so no fire lands in the slow class: a multiple-of-8 head plus
a remainder below the 16-row threshold, capped at `max_embed_length()` rounded
down to a multiple of 8. A 189-token delta fires as 184 + 5, both fast.

It is a pure function with nine unit tests, including the exact value
`aligned_prefill_chunks(189, 2048) == [(0,184), (184,189)]` and a sweep asserting
the no-slow-fire property for every length 1..600 — a regression here is silent
and costs half a second per turn.

Cold prefill gained the most (22.3 s → 11.4 s) because `prefill_chunks` spreads
the remainder across the *first* chunks, so a 7211-token prompt became four
fires of ~1803 rows each — all four in the slow class.

**Caveat, stated rather than buried:** re-chunking moves fire boundaries, so
token-level output can differ the way any re-chunking can. `apc-graft-probe`
measures that effect as benign argmax flips near ties (a cold prefill chunked at
2048 already disagrees with one chunked at 1024 on 1 of 6 prompts). Per-turn
generated token counts are unchanged and the acceptance suite passes; this is
not a claim of bit-identical output.

## What the example inferlets say about the rest

`tests/inferlets/` (34 examples) is the reference for how pie is meant to be
driven. Census:

| idiom | examples using it | `opencode-session` |
|---|---:|---|
| `WorkingSet::new` | 34 | yes |
| `intrinsics::logits` | 34 | yes |
| `run_ahead` decode loop | **25** | **no** |
| `.capacity(channel_capacity())` | 29 | partly |
| `prefill_chunks` | 6 | replaced (above) |
| `fork` | 3 | no — refused on Metal |
| `from_index` / `update_index` | **0** | yes (strategy B's whole point) |

Two gaps worth acting on, both visible by comparison:

**1. The decode loop is host-driven.** In `naive-baseline` and
`prefix-tree-kv-cache` the epilogue feeds its own inputs — `tok_in.put(&token)`,
`kv_len.put(&next_length)`, `positions.put(&length)`,
`page_indptr.put(indptr(1, &page_count))` — so one program runs the whole decode
and the host only drains `tok_out`. Growth is carried in channel *values*, never
in channel *shapes*, which is also why those examples never hit the program
churn described above. `opencode-session` builds a fresh `Pass` per token
(`engine.rs:605`). Its module docs give a real reason — `run_ahead` overshoots
the stop token, and on a hybrid model the overshoot folds into recurrent state
that cannot be rewound — but Qwen3-Coder is not hybrid, so on this model the
device-carried loop is available and is the idiom.

**2. First token is on the critical path.** `text-completion-bench` attaches a
device-only `tok_in` channel to *both* the prefill and decode passes, so decode
fires are submitted immediately after the prefill submit and the host round-trip
for the first token runs in parallel, off the critical path. Strategy B waits
for it on the host. Measured cost here: 38–78 ms per turn.

Also learned, and load-bearing for the speculation work: **the driver refuses a
fire that reads more than 8 logits rows** (`batch/forward.cpp:4144`). That caps a
speculative verify at k ≤ 7 no matter what `DRAFT_K` says.

### On `integrations/opencode/ttb`

Worth being precise, because the name suggests otherwise: this directory holds
no inferlets. It is test-time-bench's *reporting contract* adopted by hand —
a benchmark config, grader-image digest pins, a dataset snapshot pin, and
`ttb_summary.py`, which emits `null` for any budget that is declared but not
enforced rather than printing the declared value. Its most useful contribution
so far is `model_calls`: 8.2 per case for pie against 2.8 for vLLM-metal, which
turns "the vLLM arm gave up" from an inference off wall-clock into a
measurement. TTB's own decoder *is* an inferlet
(`examples/inferlets/decoder-*-rust`), but that lives in the upstream repo and
is not vendored here.

## Round 2: three leads, one small win and two dead ends

Measured after the fix above, all at rows-per-fire 4096 so the arms compare.

**Wider fires — worth ~6%, in cold prefill only.** `PIE_MAX_FORWARD_TOKENS`
2048 -> 4096 took the replay 18.1 s -> 17.0 s, entirely from turn 1
(13.07 -> 12.03 s). Steady turns did not move at all, which is the expected
result once stated: the 189-token delta was never near the 2048 cap, so only the
chunked 7.2k cold prompt could benefit. 8192 is clamped by the driver back to
4096 (`rows per fire: 4096`, activation pool 336 MB of 1024 MB), so 4096 is the
usable setting. Keep it; do not expect more.

**Per-device tuning constants — no win available. Negative result.**
`tuning_for()` (`device_tuning.cpp:58`) switches on `apple_family` with cases for
9 (M3/M4) and 8 (M2) and falls through to the M1 Max measurements otherwise.
This machine reports **`apple_family = 10`**, verified directly against Metal, so
it does take the fallthrough — and `device_tuning.hpp`'s own header warns that a
generation off "costs the most". So the setup looked exactly like a free win.

It is not. Our 189-row prefill puts **11 rows per expert** through the routed
GEMM (`189 * 8 / 128`), and `moe_tile_mid_per` ships at 32 — so the knob
provably straddles at our width, which is the condition `benches/tune_device.py`
insists on before a sweep means anything.

| arm | total | steady prefill |
|---|---:|---:|
| default (M1 constants) | 17.02 s | 690 ms |
| `moe_tile_mid_per=8` | 17.26 s | 739 ms (7% **worse**) |
| `qmm_bn_crossover_tg=96` (family 9's value) | 17.03 s | 693 ms (no change) |
| both | 17.28 s | 741 ms |

The default is better. `qmm_bn_crossover_tg` is reported as *inconclusive* rather
than confirmed: it moved nothing, and this workload may simply not straddle it.

**Two tools that do not work on this path.** Worth recording so nobody spends
the afternoon again:

- `pie config tune` is **broken against this build's config schema**. It sets
  `[driver] memory_profile`, which `pie config list` does not have; without
  `--for` it refuses because the profile is `auto`, and with it the set fails.
  It also would not have helped here — it sweeps the *frame* knobs
  (`frame_size`, `frame_submit_depth`, `frame_dispatch_depth`), not the GEMM
  crossovers.
- `PIE_METAL_ABLATE` does not reach this fire. It is read in
  `batch/decode_timing.cpp`; ablating any MoE kind against a 184-row PTIR
  prefill prints no `[ablate]` line and changes the time by 0.06%. The
  "attention is 16.5% of a 2048-token prefill" figure quoted elsewhere comes
  from the llama/gemma encode path and should not be carried over to this one.

## Where the remaining time is

A 184-row fire at 7424 context costs **579 ms**. Its memory roofline is roughly
**110 ms**: the routed GEMM touches essentially all 128 experts at this width, so
call it the full 17.18 GB of weights at the ~203 GB/s this machine achieves
(~85 ms), plus 6 query tiles over a 0.727 GB cache (~22-31 ms). That is **~5x off
roofline**, and it is now 71% of a steady turn.

vLLM does the same cold prefill at ~1209 tok/s against pie's ~633, so it is
about 2x closer to the same roofline on the same weights and hardware.

## CORRECTION (2026-08-14, later the same day): Round 3's attribution was WRONG

Everything in "Round 3" below rests on `PIE_METAL_ABLATE`, and the hook I added
to make it work for the llama family converts llama's DAG `Kind` to the
driver-wide `Kernel` with `pso_kind()`. **`pso_kind` is a lossy PSO-SELECTION
map, not an identity.** Every projection kind — `QmvQ`, `QmvK`, `QmvV`, `QmvO`,
`QmvGate`, `QmvUp`, `QmvDown`, `Router`, `LmHead` — collapses onto
`Kernel::QmvGate`; every norm collapses onto `Kernel::Rms`; and the attention
node maps to nothing named `sdpa_paged` at all.

So the sweep ablated PSO classes, not the kinds it named. Concretely:

- `PIE_METAL_ABLATE=sdpa_paged` **never skipped the attention dispatch**.
  Proved by A/B: with the matrix path on, ablating it gives 579.26 ms against a
  579.26 ms baseline; with the matrix path off, 1241.52 against 1239.86. It
  changes nothing in either mode, while switching the path itself changes
  579 -> 1240.
- Therefore "attention accounts for ~0 of prefill" was an artifact.
- Therefore **"all 36 kinds ablated leaves 373 ms of 562" was also an
  artifact** — the "everything" list never included the attention that was
  running, and attention is the single largest consumer.

**The claim "roughly two thirds of every forward pass is not compute" is
withdrawn.** It was the residue of un-ablated work, not overhead. Advice I gave
on the back of it — that ICB recording upstream would not help us, that the
driver path was mostly non-GPU — does not follow and should be disregarded.

### What replaces it

`PIE_METAL_DISPATCH_TRACE`, which reads per-dispatch GPU intervals rather than
inferring from removals, on 20 fires of 184 rows at 7424 context:

| share | kernel |
|---:|---|
| **42.70%** | `sdpa_paged_mma_bfloat16_d_128` (attention) |
| 23.33% | `affine_qmm_t_routed_..._bm_32_bn_64` (routed MoE GEMM) |
| 14.21% | `affine_qmm_t_routed_..._bm_16_bn_64` (routed MoE GEMM) |

Attention is ~43% of prefill and the routed mixture ~38%. Two independent
methods agree on this: the trace's own shares, and the matrix-path A/B
(579 ms vs 1240 ms), which is a price rather than a share. The GPU is saturated,
not idle.

**The lesson worth keeping.** An ablation harness that silently matches nothing
reports "this kernel is free" — the same failure shape as a dead server
reporting "0 patches" and a `grep -c` fallback reporting a healthy arm void.
Three times in one day, an instrument's silence was read as a measurement. The
driver's own ablation banner warns about precisely this ("'X' IS NOT A KERNEL
KIND — it ablates NOTHING and this run will report the baseline"), and it did
not fire here because `sdpa_paged` IS a valid kind name; it simply is not the
kind that dispatch resolves to. A name-validity check is not a
did-anything-actually-change check.

## Round 3: the instrumentation, and what it found

**The fix that made attribution possible.** `PIE_METAL_ABLATE` silently did
nothing for this checkpoint. The hook exists in the qwen3_5 walks
(`decode_step_mb.cpp:761`) but was **missing from the llama walk**, and the llama
family is the one that covers `qwen3_moe` (`model/llama/geometry.hpp`) — so every
ablation of a Qwen3-Coder fire returned the baseline while looking armed. One
line in `model/llama/encode.cpp` (`if (kernel_ablated(pso_kind(d.kind))) continue;`,
mapping llama's DAG `Kind` to the driver-wide `Kernel` the same way the dispatch
does) makes the whole tool work for this family.

Two harness notes that cost real time: consecutive `pie run` invocations must be
separated until `:18080` is free, or the next boots into a held port and reports
`FAILED` with no banner — which reads exactly like "this kernel is free". And
`grep -i refus` matches the routine boot banner, so it is a useless failure
classifier here.

**Per-kernel attribution of a 184-row fire at 7424 context** (baseline 562 ms,
36 kinds swept one at a time):

| kind | ms | vs baseline |
|---|---:|---:|
| `ll_expert_gate` | 439.0 | **−123.1** |
| every other kind — `sdpa_paged`, all `qmv_*`, norms, rope, `kv_append`, shared expert, dense FFN | ~565 | ~+3 (noise) |
| `ll_moe_gather` | 610.9 | +48.8 |
| `ll_moe_combine` | 632.7 | +70.6 |
| `ll_moe_sort` | 699.4 | +137.3 |

Removing the MoE bookkeeping makes the fire SLOWER, which is the healthy answer:
sort/gather/combine are earning their cost by handing the expert GEMM contiguous
rows. And exactly one kernel's removal saves anything.

Note the labels are coarser than they look: the hook keys on `pso_kind(d.kind)`,
which collapses several DAG kinds onto one pipeline, so `ll_expert_gate` names
the routed expert projections as a group rather than `gate_proj` alone.

**The decisive run: ablate everything at once.**

| rows | full | all kernels ablated | actual compute | overhead share |
|---:|---:|---:|---:|---:|
| 1 | 21.0 ms | 13.4 ms | 7.6 ms | **64%** |
| 8 | 72.9 | 49.9 | 23.0 | 68% |
| 32 | 222.1 | 172.0 | 50.1 | 77% |
| 184 | 563.6 | 373.6 | 190.1 | 66% |
| 512 | 1286.9 | 838.1 | 448.9 | 65% |

**Roughly two thirds of every forward pass is not compute, at every width, and it
scales with the row count.** Three clocks inside the probe put it precisely: at
184 rows, guest-side trace building is **0.03 ms**, the `submit` WIT call is
**2.95 ms**, and **560 ms is inside the await** — engine plan, driver encode, GPU,
and the result's trip back. Of that 560 ms, 373 ms survives with every compute
kernel skipped.

So it is not the guest, not the wire, and not the kernels. It is the engine and
driver path around them, and it is row-dependent.

**This corrects an earlier reading.** `results-speculation.md` decomposed a decode
step as `FIXED + SLOPE x context` and called the fixed term "the MoE weight read,
already running at ~203 GB/s and therefore near this machine's peak". At one row
the non-kernel share is 64%, so that fixed term is mostly overhead, not the
weight read, and the "near peak" conclusion drawn from it does not hold.

**Still open:** which stage inside the await. The candidates are the engine's
launch-plan construction, the driver's per-row descriptor and argument-table
setup, and the KV write-descriptor resolution — all row-dependent, none yet
timed separately. That needs driver-side timers, which is the next round and is
a much narrower search than it was this morning.

## Next

1. **Attribute the 5x.** Instrument first (extend the ablation to PTIR, or GPU
   capture). This is the whole remaining gap to vLLM.
2. **Find the driver-side cause of the mod-8 penalty.** The guest now steers
   around it; every other pie guest still pays it, and `prefill_chunks` — the
   SDK helper six examples use — actively produces slow widths.
2. **Device-carried decode** on non-hybrid models (gap 1 above).
3. **Overlap the first token** (gap 2), worth 38–78 ms per turn.
4. Then re-measure against vLLM. The remaining steady-state gap is 0.97 s
   against 0.65 s.

## Reproducing

```sh
cd runtime/engine/tests/inferlets && cargo build --target wasm32-wasip2 --release -p decode-rows-probe
./target/release/pie -c /tmp/rows-probe/config.toml run \
  --path runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm \
  --manifest runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml
# edit ROWS in src/lib.rs to sweep widths; LONG to move the context off a
# multiple of 8 and separate the rows rule from the kv_len rule.

bash integrations/opencode/tools/phase_probe.sh   # the 6-turn guest phase breakdown
```


## Round 4: is upstream's driver better than ours? (No — 2.1x worse)

`preview-0.5` is the upstream release-preview branch on OUR architecture (not the
`rewrite`, which is a separate Rust/crates line). Its `driver/metal` is 138 files
to our 139; 65 of 273 differ. `driver/common` — the runtime-facing ABI — is
byte-identical across all 11 files, so the swap compiles against our runtime.

So it was swapped in wholesale (`git checkout refs/upstream/preview-0.5 --
driver/metal`, full CMake rebuild, all 35 translation units recompiled) and
measured with the same probe build and config.

| rows @ ctx 7424 | ours | ours, `PIE_METAL_SDPA_MMA=0` | preview-0.5 |
|---:|---:|---:|---:|
| 1 | 21.2 ms | 21.3 | 21.4 |
| 184 | **564.8** | 1183.5 | 1206.9 |
| 189 | 1144.6 | 1773.3 | 1777.0 |
| 192 | **567.8** | 1211.1 | 1249.2 |

**Do not adopt preview-0.5's driver.** It is 2.1x slower on prefill-width fires.
Decode (1 row) is identical on all three.

**The whole difference is one file.** `sdpa_paged_mma.metal` — the simdgroup
matrix-unit attention kernel — exists on our branch and not on `preview-0.5`,
`dev-sslee` or `tart`. Turning it off with `PIE_METAL_SDPA_MMA=0` reproduces
preview-0.5's numbers to within 4%. It only engages when `sdpa_should_tile` is
true (>= 32 rows per request), which is exactly why decode is unaffected and
prefill is halved.

**And the mod-8 penalty is upstream's, not ours.** It is present in all three
arms with a near-identical ABSOLUTE cost:

| arm | 189 minus 184 |
|---|---:|
| ours, MMA on | +579.8 ms |
| ours, MMA off | +589.8 ms |
| preview-0.5 | +570.1 ms |

So it is not something our 65 differing files introduced, and it is not in the
attention kernel either — it survives switching the attention path entirely,
which independently corroborates the ablation result that `sdpa_paged` accounts
for none of it.
