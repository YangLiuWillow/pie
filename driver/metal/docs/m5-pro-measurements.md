# Measurements — Apple M5 Pro, 48 GB

Constants measured on this machine. **These supersede published figures.**
Public M5 bandwidth numbers conflict (153 GB/s vs ~120 GB/s) and neither is
credible for a Pro tier.

Every row must carry how it was taken. A number without a method is a guess with
a decimal point.

| Field | Value | Method | Date |
|---|---|---|---|
| macOS / Darwin | macOS 26.5.1 (25F80), Darwin 25.6.0 | `sw_vers`, `uname -r`. Note: [M5] §4.1 ran Darwin 25.4 | 2026-08-19 |
| Toolchain | Apple clang 21.0.0 (CLT; no offline `metal` compiler — kernels compile at run time via `newLibraryWithSource`), cmake 4.4.2, rustc 1.97.1 | `c++ --version`; `xcrun -sdk macosx metal --version` fails, CLT-only install | 2026-08-19 |
| `apple_family` | **9** as the driver sees it; the device itself answers **yes to Apple10** (and no to 11/12), which the driver never asks | `tools/rawmetal/device_identity_probe`, clean env (`env -i`) | 2026-08-19 |
| `gpu_core_count` | **20** | same probe — IOKit `gpu-core-count` | 2026-08-19 |
| `DeviceTuning` block selected | **case 9 (M3/M4)**: `qmm_bn_crossover_tg` = 96, everything else at M1 Max defaults. NOT the M1 Max default block the plan expected — the family probe stops at Apple9, so an M5 is indistinguishable from an M3/M4 until the probe list is extended to Apple10 (Phase 1) | same probe: selected vs default-constructed fields | 2026-08-19 |

## Suite state (T0.1)

Build and on-device suite on this machine, 2026-08-19, branch
`docs/m5-pro-bringup-plan` tree:

| Step | Result | Method |
|---|---|---|
| `cargo build --workspace --all-targets --exclude pie-server-py` | OK (2m42s cold) | exit 0, warnings only |
| `cargo rustc -p pie-loader-capi --lib --crate-type staticlib` | OK | produces `target/debug/libpie_loader_capi.a`, found by the CMake configure |
| `cmake -S driver/metal -B /tmp/metaltests -DCMAKE_BUILD_TYPE=Release -DPIE_METAL_BUILD_TOOLS=ON` + build | OK | 42 ctest targets + rawmetal tools |
| `cargo test --workspace --exclude pie-server-py` | 772 passed / 1 failed after two machine-independent fixes (below); the failure is pre-existing engine/DSL drift, analysed below | 50 suites. The plain `--workspace` form CANNOT link on macOS — `pie-server-py` hard-enables `driver-cuda`, and feature unification poisons `pie-engine`'s lib test with undefined CUDA symbols |
| `ctest --test-dir /tmp/metaltests` | **40/42**; both failures analysed below | 75s wall |

Fixes needed to get there (in the T0.1 PR, none touch the driver):

1. **`chat::cue()` → `chat::cue(true)`** in 16 guest inferlets under
   `tests/inferlets/` and `runtime/engine/tests/inferlets/`. Commit
   `1503c7c8b` changed the WIT to `cue(thinking: bool)` and migrated only the
   production `inferlets/`; the old `cue()` was the thinking-on variant
   (host mapped `thinking ? cue() : cue_no_think()`), so `cue(true)` is
   behavior-preserving. This unblocked `pie-bin`'s `boot_smoke` and the
   `text-completion-bench` inferlet T0.3 needs.
2. **Two `rng_contract` scan exemptions**: `llama_numerics_test.cpp`'s
   murmur3-style test-weight hash trips the `>>40` needle, and
   `openai-serving/src/session.rs`'s FNV fingerprint trips the golden-ratio
   word. Both are the scan's own "unrelated user" idiom, not contract owners.

ctest failures, analysed:

- **`llama_numerics_test`: 50 pass / 19 fail — exactly the recorded baseline**
  (`docs/HANDOVER.md` §5.4: 18 pre-existing MoE routing ties + 1 deliberate
  row-gate tie). Matching the recorded baseline is a pass for T0.1; no new
  failure appeared on M5.
- **`host_buffer_bind_test`: fails only at the 12 GB sparse-mapping step**,
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` at commit — **under
  contention**: a concurrent `pie serve` from another session (pie-opencode
  checkout) held 16 GB with a model GPU-resident, leaving ~10.6 GB free. The
  same binary **passes at 6 GB** (1024/1024 elements at +5.37 GB offset), so
  the wrap/bind/read mechanism is intact on M5. **Follow-up**: re-run at
  12 GB on a quiet host to confirm the pass; only if it still fails does the
  residency-accounting question (whole sparse mapping backed at commit vs
  demand-faulted) open up.

Pre-existing breakage, recorded not fixed (engine/DSL, not driver, not M5):
`pie-engine`'s `north_star_e2e::north_star_mtp_grammar_composition` fails
deterministically — "seed channel 9: 16 bytes, expected 128". The 16-byte
seed is `mtpverify`'s `[k=4]` i32 draft; a 128-byte channel at n=32 is
`positions` — the guest's seed order and the registered program's channel
numbering have drifted, most plausibly in the recent SDK channel refactors
(`ead310dd4`, `65cf1a848`). Machine-independent; needs its own task.

Known flake, recorded not fixed: `pie-engine`'s
`terminal_cells_recycle_only_after_native_attempt_retirement` failed once
under concurrent build load and passes 5/5 + 3/3 isolated. The terminal-cell
pool is a process-global `SegQueue` shared by parallel tests, and
`pool_contains` races with any concurrent `WorkItemCompletion::new` pop; a fix
wants a design decision (serialise pool-touching tests), not a Phase 0 patch.

## Roofline

| Field | Value | Method | Date |
|---|---|---|---|
| Peak sustained bandwidth (GB/s) | _TBD_ | `roofline_probe`, ≥8 GB stream, 5 runs, ±3% | |
| Peak achieved TFLOP/s (fp16, simdgroup_matrix) | _TBD_ | `roofline_probe` whole-step | |
| Ridge point (FLOP/byte) | _TBD_ | peak_FLOPS ÷ peak_bandwidth | |
| Sustained-clock behaviour | _TBD_ | `dvfs_probe` | |

## Tensor path (Phase 2)

| Field | Value | Method | Date |
|---|---|---|---|
| `matmul2d` ÷ `simdgroup_matrix`, fp16 | _TBD_ | T2.1, shapes 4096³/2048³/1024³/(128×5120²) | |
| Peak TFLOP/s via `matmul2d` | _TBD_ | T2.1 | |
| fp8 (E4M3) ÷ fp16 | _TBD_ | T2.2 | |
| int8 ÷ fp16 | _TBD_ | T2.2 | |
| Accumulator width | _TBD_ | T2.2, Rigel-style probe | |
| **Phase 3 go/no-go** | _TBD_ | ratio > 1.2× ⇒ go | |

Reference points for comparison — **[RIG]** on M4 Max: `matmul2d` 1.05–1.21×
over `simdgroup_matrix`, ceiling ~14.8 TFLOP/s inside the ALU limit; fp8 at
0.94× fp16 (emulated); accumulator ≥ fp32.

## Crossovers (Phase 1)

Fill from `benches/tune_device.py`. Each field needs its own arm-by-arm table in
the `device_tuning.hpp` house style before it goes into the source.

| Field | M1 Max default | M5 Pro measured | Control agrees? | Date |
|---|---|---|---|---|
| `qmm_min_batch` | 8 | _TBD_ | | |
| `qmm_min_batch_emulated` | _see [DT]_ | _TBD_ | | |
| routed / MoE crossovers | _see [DT]_ | _TBD_ | | |
| `sdpa_tile_min_rows_per_request` | _see [DT]_ | _TBD_ | | |

## Model throughput — three-way baseline (T0.3, 2026-08-19)

**Method.** `benches/three_way.py`, one invocation per (rep, prompt length,
model): every invocation runs one rep of each engine back to back (arms
alternated), 5 invocations per cell, reps outermost. Each invocation runs two
latency-mode shapes of 4 sequential requests (warmup 2): `max_tokens=1` and
`max_tokens=128`. Per rep, from mean request latencies:

    prefill tok/s = prompt_tokens / lat(mt=1)              # launch-inclusive
    decode  tok/s = Δoutput_tokens / (lat(mt=128) − lat(mt=1))

Cells are mean ± stddev over the 5 reps. Checkpoints: `mlx-community/…-4bit`
served by pie and mlx-lm 0.31.3 (MLX 0.32.0); `unsloth/…-Q4_0.gguf` by
llama.cpp b9960 (`Q4_0` is the closest bpw match — see `three_way.py`'s
header). Machine gate before the sweep: `roofline_probe` streaming roof
283 GB/s cold one-shot (recorded warm roof 294–298); GPU tenancy held
exclusively (peer sessions coordinated off the device). pie ring sized by
`--max-model-len 16384` — see the harness-fix note below. Raw JSONL + sweep
log: `docs/bench-archives/t03_*` (local, gitignored).

**Read the numbers with three caveats.**

1. Prefill is **TTFT-derived and launch-inclusive** — engine-server overhead
   is in the denominator. It is comparable across these engines at the same
   length; it is NOT comparable to [M5] Tables 2–3, whose pp is bench-style
   raw batch prefill (T0.4 reproduces BaseRT with its own instrument instead).
2. All runs are `--no-ignore-eos` (mlx-lm cannot pin output length). On
   Llama-3.2-1B the engines EOS at different points (actual-work table
   below), so its decode cells average over fewer tokens and carry more
   noise; Qwen ran the full 128 everywhere.
3. pie applies its own chat template (~2 extra prompt tokens here; per-engine
   prompt counts recorded below).

### Prefill tok/s (launch-inclusive)

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie | 5274 ± 32 | 6235 ± 47 | 6703 ± 34 | 7101 ± 48 | 7108 ± 11 |
| Qwen3-0.6B q4 | mlx-lm | 1794 ± 18 | 3337 ± 14 | 5314 ± 203 | 7964 ± 204 | 9467 ± 99 |
| Qwen3-0.6B q4 | llama.cpp | 8053 ± 533 | 10389 ± 489 | 10759 ± 433 | 11141 ± 226 | 9554 ± 112 |
| Llama-3.2-1B q4 | pie | 3108 ± 18 | 3278 ± 18 | 3351 ± 14 | 3419 ± 12 | 3347 ± 26 |
| Llama-3.2-1B q4 | mlx-lm | 2486 ± 4 | 3366 ± 732 | 5575 ± 31 | 7589 ± 45 | 8454 ± 291 |
| Llama-3.2-1B q4 | llama.cpp | 6574 ± 272 | 7459 ± 332 | 8054 ± 180 | 8435 ± 50 | 7776 ± 284 |

### Decode tok/s

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie | **379.5 ± 1.5** | **368.1 ± 3.4** | **351.9 ± 6.4** | **327.2 ± 2.6** | **285.6 ± 2.9** |
| Qwen3-0.6B q4 | mlx-lm | 345.6 ± 2.4 | 296.7 ± 88.9 | 325.1 ± 6.0 | 291.7 ± 2.6 | 252.5 ± 2.8 |
| Qwen3-0.6B q4 | llama.cpp | 349.9 ± 1.5 | 330.0 ± 26.3 | 330.6 ± 3.1 | 310.3 ± 1.6 | 283.2 ± 2.2 |
| Llama-3.2-1B q4 | pie | **291.6 ± 2.9** | **281.3 ± 6.2** | 269.3 ± 3.7 | 246.8 ± 0.8 | 196.7 ± 29.5 |
| Llama-3.2-1B q4 | mlx-lm | 281.6 ± 3.2 | 373.8 ± 214.1 | 269.5 ± 4.5 | 262.5 ± 1.9 | 238.1 ± 26.9 |
| Llama-3.2-1B q4 | llama.cpp | 267.1 ± 5.4 | 279.7 ± 14.9 | 267.3 ± 7.9 | 260.3 ± 2.8 | 235.4 ± 38.9 |

### Actual work per cell (mean prompt tokens / mean output tokens at mt=128)

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie / mlx / ll.cpp | 152/128 all | 278/128 all | 530/128 all | 1043/128 all | 2069/128 all |
| Llama-3.2-1B q4 | pie | 177/128 | 303/128 | 555/128 | 1068/128 | 2094/128 |
| Llama-3.2-1B q4 | mlx-lm | 175/84 | 301/22 | 553/128 | 1066/48 | 2092/99 |
| Llama-3.2-1B q4 | llama.cpp | 175/49 | 301/8 | 553/19 | 1066/46 | 2092/88 |

**What the table says.** pie holds the best decode at every Qwen cell
(+8–13% over mlx-lm, +1–9% over llama.cpp) and the short-context Llama cells;
its long-context Llama decode trails (196.7 at pp2048, noisy — the unequal
EOS behaviour makes that cell soft for every engine). pie's launch-inclusive
prefill is flat with length (~7.1k on 0.6B, ~3.4k on 1B): fastest startup at
short prompts (24 ms TTFT at pp128 vs mlx-lm's 71 ms) but a per-layer
dispatch-bound plateau that mlx-lm passes by pp1024. This is the small-dense
regime — the same tree measured the opposite prefill ordering on the 30B MoE
(`docs/HANDOVER.md`) — and it is the M1-constants baseline that Phase 1's
tuning entry and T5.1 re-measure.

**Harness fix this task required** (in `benches/three_way.py`, same PR):
pie_bench's own `--max-model-len` default (2048) sizes the Metal KV ring at
exactly 64 pages, so the 2048-token arm was refused ("allocation of 65 units
can never fit") while the other engines ran it. `three_way.py` now forwards
its `--max-model-len` to pie exactly as it already did to llama.cpp, and
grew a `--pie-extra` passthrough for pie-only knobs. Ring-size control: one
qwen/pp512 pie rep under the new ring — prefill 6716 vs recorded 6702 ± 34,
decode 354.4 vs 351.9 ± 6.4 — so pre-fix cells stand.

### Qwen3.6-27B (deferred to Phase 1/5)

| Model | Quant | Engine | Prefill tok/s | Decode tok/s | Date |
|---|---|---|---|---|---|
| Qwen3.6-27B | q4 | pie (case-9 constants) | _TBD_ | _TBD_ | |
| Qwen3.6-27B | q4 | pie (M5 entry) | _TBD_ | _TBD_ | |

BaseRT's own published figures are in **[M5] Tables 1–3**; ±10% reproduction
with BaseRT's own instrument is the T0.4 gate, measured below.

## BaseRT reproduction (T0.4, 2026-08-19)

**Method.** `basert bench <model> -p {128,256,512,1024,2048} -n 128`,
BaseRT v0.1.6 (`~/.basert`), 5 timed reps per point (bench default), adaptive
warmup (bench default). Quiet GPU (same coordinated window as T0.3).
Darwin **25.5.1** here vs the paper's **25.4** — the one uncontrolled
variable. Models: `Qwen/Qwen3-0.6B` convert-on-pull default-q4 (623.5 MB
.base, 28 layers, BaseQ4, tied embeddings) and
`unsloth/Llama-3.2-1B-Instruct` (1.0 GB, 16 layers — `meta-llama/…` answers
401 without an HF token; unsloth is the weight-identical mirror), each
cross-checked against the pre-converted catalog artifact
(`basecompute/Qwen3-0.6B` 410 MB, `basecompute/Llama-3.2-1B-Instruct`
1.0 GB). Raw log: `docs/bench-archives/t04.log` (local).

### Qwen3-0.6B q4 — reproduces: every point within ±3% ⇒ **gate PASSED**

| Metric | Measured | [M5] published | Δ |
|---|---|---|---|
| pp128 | 12207 ± 28 | 11873 | +2.8% |
| pp256 | 17602 ± 49 | 17575 | +0.2% |
| pp512 | 20353 ± 154 | 20706 | −1.7% |
| pp1024 | 20844 ± 28 | 21213 | −1.7% |
| pp2048 | 20111 ± 19 | 20105 | +0.0% |
| tg128 | 536.9 (526.5–545.9 across the five runs); catalog artifact 532.7 | 530.8 | +1.1% |

Observed oddity, recorded without interpretation: the 410 MB catalog artifact
and the 623 MB local conversion decode at the same rate (532.7 vs 536.9).

### Llama-3.2-1B q4 — reproduces **only at matched quantization profile**

**Resolved (same day).** The −30% decode shortfall below is a quantization
*profile* mismatch inside "4-bit", not an engine or machine problem. The
converter's `default-q4` profile (and the current catalog artifact) pins
`embed_tokens.weight` at **f16**; with tied embeddings that is a 525 MB bf16
lm_head read per decoded token. The weights-bytes arithmetic is exact:
1073 MB = 547 MB (973M non-embedding params at q4) + 525 MB (262.7M-param
embedding × 2 B). A bandwidth-bound decode of that artifact tops out at
240 tok/s (258 GB/s effective, 87% of the machine roof) — the published
342.1 would need 367 GB/s, **above the roof**, so the paper cannot have
measured this artifact. Rebuilding from the same bf16 source with the
repo's own `default-q4-embq6` profile (embedding at q6; Apache-2.0
converter, profile fetched from `basecompute/baseRT`) gives a 761 MB
artifact and:

| Metric | f16-embedding artifact (pull default) | embq6 artifact | [M5] published | Δ (embq6) |
|---|---|---|---|---|
| tg128 | 238–240.8 (every rep, both artifacts) | **325.1 ± 0.4 / 324.4 ± 0.9** | 342.1 | **−5.0% ⇒ gate PASSED** |
| pp128 | 8033 ± 67 | 8740 ± 14 | 8936 | −2.2% |
| pp2048 | 12313 ± 12 | 12399 ± 12 | 12451 | −0.4% |

Two consequences worth keeping: (1) anyone reproducing [M5] via
`basert pull` gets the f16-embedding artifact and lands 30% under Table 1 on
llama-arch decode — the paper's "matched quantisation" (their ~4.9 bpw
effective ≈ GGUF Q4_0's 730 MB) holds only with an embedding-quantized
profile; (2) Qwen decode is *insensitive* to this (646 MB f16-emb and 410 MB
catalog artifacts decode identically at ~533–546), i.e. the 0.6B decode is
not bandwidth-bound — it sits on a dispatch/latency floor, which is exactly
the regime pie's T0.3 decode numbers live in too.

The elimination record below is kept as the process that got there.

#### The original finding (pull-default artifact) — decode 30% under Table 1

| Metric | Measured | [M5] published | Δ |
|---|---|---|---|
| pp128 | 8033 ± 67 | 8936 | −10.1% (at the gate edge) |
| pp256 | 10758 ± 21 | 11032 | −2.5% |
| pp512 | 11743 ± 14 | 12134 | −3.2% |
| pp1024 | 11854 ± 11 | 12084 | −1.9% |
| pp2048 | 12313 ± 12 | 12451 | −1.1% |
| tg128 | 238.2 (236.4–239.5) | 342.1 | −30.4% (see resolution above) |

**The discrepancy, identified as far as a binary-only engine permits.**
Eliminated, each by measurement:

- *Thermal / machine state*: our independent llama.cpp b9960 tg128 on the
  same weights, same machine, same session (T0.3) is **267.1 vs the paper's
  267.1 — exact**; mlx-lm 281.6 vs 298.0 (−5.5%). A thermal or OS-load
  problem would move every engine, and it moved none of them 30%.
- *Conversion*: the catalog's own pre-converted artifact decodes at 240.1.
- *KV mode*: `--paged-kv` decodes at 240.2.
- *Prompt length*: tg128 is 236–240 for every `-p` from 0 to 2048.
- *Noise*: stddev ≤ 1.5% on every run.

Remaining candidates, not distinguishable from outside the engine: a Darwin
25.4 → 25.5 change in something BaseRT's llama-arch decode leans on, or an
unpublished engine configuration behind the paper's Llama rows. v0.1.6 ships
no `basert-profile`, so the per-op split is unreachable without vendor input.

*Re-measured on request, same day, two fresh alternated rounds with the
27B download suspended (`kill -STOP`) for clean arms:* BaseRT Llama tg128
**240.8 ± 0.2 / 239.9 ± 0.4** (catalog artifact; local conversion 240.8) —
stable at −29.8%. In the same rounds, `llama-bench` (llama.cpp's own
instrument, same GGUF) gave tg128 275.1/275.4 vs the paper's 267.1 (+3%),
and BaseRT Qwen gave 545.3/546.4 vs 530.8 (+2.8%). Both controls sit within
+3% of publication while the Llama decode sits 30% under it, in the same
minutes on the same silicon.

**Consequence (superseded by the resolution above):** the gate now passes
for both models; the standing rule it leaves behind is narrower — any
Pie-vs-BaseRT comparison must name the BaseRT **artifact** (profile + bytes),
not just "q4", because the pull default and the paper's artifact differ by
30% on llama-arch decode. The published cross-engine rows themselves
validated against our independent T0.3 numbers (llama.cpp exact,
mlx-lm −5.5%).

## Full [M5] Table 1–2 replication (T0.4 extension, 2026-08-19)

Requested extension: replicate every BaseRT pp/tg figure in [M5] Tables 1–3.
Same instrument and protocol as above (`basert bench -p {128…2048} -n 128`,
5 reps, adaptive warmup, quiet GPU — concurrent downloads suspended via
SIGSTOP around every bench). Raw log: `docs/bench-archives/t04_matrix_a.log`.

**The finding that decides which artifact to bench, confirmed on every dense
model tried:** the paper's "4-bit" is *all*-q4 — embedding included. A
`default-q4-embq4` profile (the repo's `default-q4-embq6` with the embedding
rules set to `base_q4`, archived in `docs/bench-archives/`) reproduces every
4-bit row; the pull-default / catalog q4 artifacts (f16 embedding) cannot
physically reach the published decode on tied-embedding models. The q8 rows
reproduce with pull defaults directly (the catalog's q8 artifacts quantize
the embedding; its q4 artifacts do not — an upstream inconsistency worth
reporting to basecompute).

### 4-bit rows (tg128, with pp128/pp2048 as prefill anchors)

| Model | Artifact | tg128 ours | published | Δ | pp128 Δ | pp2048 Δ |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | pull default (f16 emb; insensitive — see dispatch-floor note) | 536.9 | 530.8 | +1.1% | +2.8% | +0.0% |
| Llama-3.2-1B | **embq4 rebuild** | 355.1 | 342.1 | **+3.7%** | −0.6% | +0.1% |
| Llama-3.2-1B | pull default / catalog (f16 emb) | 240 | 342.1 | −30% | — | — |
| Qwen3.5-2B | **embq4 rebuild** | 217.1 | 219.6 | **−1.2%** | −0.6% | −0.2% |
| Qwen3.5-2B | pull default (f16 emb) | 137.3 | 219.6 | −37% | −8.5% | −8.0% |
| Qwen3.5-2B | embq6 rebuild | 197.1 | 219.6 | −10% | −1.8% | −0.3% |
| Llama-3.2-3B | catalog (f16 emb) | 109.1 | 137.0 | −20.6% | −5.5% | −4.1% |
| Llama-3.2-3B | **embq4 rebuild** | 139.2 | 137.0 | **+1.7%** | −0.3% | +6.4% |
| Gemma-4-E2B | — | _blocked: HF-gated, no catalog entry, no token_ | 149.8 | | | |
| Gemma-4-26B-A4B | — | _blocked: same_ | 85.0 | | | |
| Qwen3-30B-A3B | pull default (original HF repo, not the catalog's -Instruct-2507) | 105.8 | 105.1 | **+0.7%** | +0.2% | +1.3% ‡ |
| Qwen3.5-35B-A3B | pull default | 109.5 | 110.6 | **−1.0%** | −1.6% | −2.9% § (pp512 −0.6% after idle re-take) |
| Qwen3.6-35B-A3B | **pull default from bf16 source** (693 tensors, `HAS_MOE`) | 110.7 | 110.7 | **+0.0%** | −2.1% | +6.2% |
| Qwen3.6-35B-A3B | quant-from-quant from mlx-community 4-bit (733 tensors, no `HAS_MOE`) | 109.3 | 110.7 | −1.2% | −31% ¶ | +4.2% |
| Qwen3.6-27B | pull default (f16 emb; insensitive here too) | 18.19 | 18.1 | **+0.5%** | −0.3% | −9.0% ‡ (pp512 −1.8%, pp1024 −1.3% quiet) |

### 8-bit rows — all reproduce with pull/catalog defaults

| Model | tg128 ours | published | Δ | pp128 Δ | pp512 Δ | pp2048 Δ |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 370.7 | 365.6 | +1.4% | +1.5% | +0.8% | +0.6% |
| Llama-3.2-1B | 210.6 | 204.4 | +2.9% | −1.4% | +0.5% | +0.4% |
| Qwen3.5-2B | 132.2 | 132.4 | −0.1% | −0.5% | −0.1% | −0.2% |
| Llama-3.2-3B | 80.6 | 79.6 | +1.2% | −0.0% | −4.6% † | +0.2% † |

† Llama-3B q8's long prefill first measured −10%/−11.5% (pp1024/pp2048) at
the tail of a long hot bench sequence; re-run after a cool-down it lands
+0.4%/+0.2% of publication. Thermal sequence-position, not a finding — and a
live demonstration of the protocol's "alternate arms, don't batch" rule.

‡ 27B prefill drifts from −0.3% (pp128) to −9.0% (pp2048) across a
back-to-back sweep — inside the gate, same in-sequence thermal signature the
3B q8 row showed; a cold pp2048 re-run is the check if the margin ever
matters. 30B's pp1024/pp2048 were −5.1%/−12.6% in the hot sweep and
−0.2%/+1.3% after a cool-down (the values in the table).

§ Long-prefill points on the ≥17 GB models carry a third pollution mode
beyond thermal drift and GPU tenancy: **page-cache contention**. A 70 GB
source download streaming through the filesystem cache while a 21–27 GB
model is mapped evicts the model mid-bench — measured directly when a 27B
"cold re-run" during the download came back 370–390 pp with ±25 stddev
against 452–462 clean. Benches from that point on ran with the download
pipeline SIGSTOPped (`bench_clean.sh`, archived); 35B's pp512/pp2048
(−9.9%/−10.2%, elevated stddev) are still marked for an idle-machine
re-take once all downloads have settled.

And the damage does not undo by pausing the writer: a suspended-download
27B re-take still measured pp512/1024/2048 at 450/430/425 — *below* the
original download-free sweep (462/458/452), after ~130 GB of sources had
churned the cache. The original sweep stands as the 27B record. Standing
rule for big-model benching on a 48 GB machine: take the numbers BEFORE
queueing bulk downloads, or after a reboot — mid-campaign cache state is a
one-way ratchet that SIGSTOP does not release. (The affected long-prefill
cells here are all decode-irrelevant: tg128 reproduced within ±1% under
every cache condition tried.)

**Two 27B side-findings.** (1) BaseRT handles GatedDeltaNet: the artifact
identifies as arch `qwen35`, 64 layers, and decodes at the published rate —
the plan's "does BaseRT run GDN" question is answered empirically. (2) Its
decode implies **≥309 GB/s effective DRAM read** (16.99 GB weights ×
18.19 tok/s) — *above* the 294–298 GB/s `roofline_probe` streaming figure.
The probe's number is one kernel's achieved bandwidth, not the hardware
ceiling; T1.3 should treat 309 GB/s as a measured lower bound on the real
roof, and the earlier "impossible above the roof" arguments in this file
should be read against per-shape achieved bandwidth (~250–260 GB/s for the
1B-class decodes), which is what they actually used.

¶ **Resolved.** The quant-from-quant artifact is structurally different,
not just numerically: converted from the `mlx-community` 4-bit checkpoint it
has **733 tensors and no `HAS_MOE` flag**; converted from the bf16 source
(the paper's method) it has **693 tensors with `HAS_MOE`** — identical to
its Qwen3.5-35B sibling — and replicates every cell: tg128 110.7 (exact),
pp128/256/512/1024/2048 = 1333/1787/2345/2773/2962 vs 1361/1831/2342/2736/
2788 (−2.1%/−2.4%/+0.1%/+1.4%/+6.2%). Forty extra tensors = one per layer,
and a model the runtime does not know is MoE takes a different dispatch
path — the ~40 ms per-call penalty was ~1 ms × 40 layers of that. Kept
below as the record of how it was found. Original note:

The one row whose prefill did NOT replicate at short prompts, and the
one row converted from an already-quantized source. Under idle conditions
(stddev ±1): pp128 936 vs 1361 (−31%), pp256 1371 vs 1831 (−25%), pp512
1975 vs 2342 (−16%), pp1024 2525 vs 2736 (−7.7%), pp2048 2905 vs 2788
(+4.2%); tg128 109.3 vs 110.7. The deficit is a near-constant **~40 ms per
prefill call** that disappears into long prompts while the steady-state rate
matches or beats publication — a per-call fixed cost, not a GEMM-rate gap,
and it did not move between a loaded and an idle machine. The only variable
this row does not share with the other twelve is its source: the
`mlx-community` 4-bit checkpoint re-quantized via `--allow-quant-from-quant`
(taken to avoid a 45 GB bf16 download, since bytes and kernels are the same
either way). Bytes matched; something per-call did not — the converter's own
warning, observed. **Open follow-up:** convert from the bf16 source
(`Qwen/Qwen3.6-35B-A3B`, 10/26 shards cached) and re-bench pp128–512; until
then this row's decode is replicated and its short prefill is not claimed.

Post-reboot re-takes under a measured idle gate (5-min load < 2.5, no
scanner > 25% CPU — post-update Xprotect/Spotlight housekeeping is a FOURTH
pollution mode, and it adds a fixed CPU latency per call rather than a
rate change): 3.5-35B pp512/pp2048 → 2503/3012 (−0.6%/−2.9%), resolving
that row; 27B pp512/pp1024 → 485/492 (−1.8%/−1.3%); 27B pp2048 stayed
437–452 across every attempt (−9 to −12%) — the most memory-intensive cell
of the campaign (2048-token prefill over a 27 GB mapping in 48 GB) and the
single long-prefill point that never cleared comfortably. Recorded as is.

Full five-length pp values for every row are in the archived raw log; the
committed anchors are pp128/pp512/pp2048 to keep this table readable.
