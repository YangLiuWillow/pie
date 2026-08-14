# Speculative decoding: pie vs vLLM, with the feature enabled on both sides

**Date:** 2026-08-13. **Machine:** Apple M5 Pro, 48 GB.
**Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — the same weights on
both engines. **pie arm:** strategy B (`opencode-session`, retained working set).
**vLLM arm:** 0.27.0 with the `vllm_metal` platform plugin, `--enable-prefix-caching`.

Speculation parameters are matched one for one:

| `inferlets/opencode-session/src/draft.rs` | vLLM `--speculative-config` |
|---|---|
| `DRAFT_K = 4` | `num_speculative_tokens: 4` |
| `NGRAM_LONG = 3` | `prompt_lookup_max: 3` |
| `NGRAM_SHORT = 2` | `prompt_lookup_min: 2` |

Both are the same algorithm — prompt lookup: find where the recent output
occurred earlier in the context and copy what followed.

Each engine's control is the *same build and the same server* with speculation
off, run in the same session and interleaved with the other engine's cells so
thermal drift cannot line up with the engine axis. On pie the control is a
separate wasm (`SPEC_OFF=1`, because `option_env!` is read at compile time), and
each boot verifies by SHA-256 that the guest actually being served is the guest
just built.

---

## The headline

**pie's speculation works and is worth up to 2.17× on decode. vLLM's, on this
build, does nothing at all.** With the feature on for both, pie finishes a
highly-draftable turn in 10.18 s against vLLM's 13.63 s, decoding at 136.9 tok/s
against 60.5.

That is a narrow claim about a specific kind of output. On agent-shaped output
the picture reverses, and the reason is a Metal kernel dispatch threshold, not
the drafting. Both halves are below.

---

## 1. Why a k-row fire is the whole story

`runtime/engine/tests/inferlets/decode-rows-probe` times decode fires of 1, 2, 4,
5 and 8 rows at two contexts on the real driver, and solves the two-term cost
model per row count. Ten fires per configuration, median reported, first two
discarded (a new row count is a new program and the driver compiles it once).

| rows | fixed (ms) | slope (ms per 1k ctx) | slope vs 1 row | ms at 16k | vs 1 row |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.68 | 1.167 | 1.00× | 31.8 | 1.00× |
| 2 | 14.70 | 2.095 | 1.79× | 49.0 | 1.54× |
| 4 | 19.08 | 3.476 | 2.98× | 76.0 | 2.39× |
| 5 | 21.42 | 4.173 | 3.58× | 89.8 | 2.82× |
| 8 | 30.10 | 6.087 | 5.22× | 129.8 | 4.08× |

Both terms grow with the row count, and they grow for different reasons.

**The KV half.** The slope is `0.464 + 0.703·rows` ms per 1k of context. Only
40% of one row's KV cost is shared; each additional row adds a full re-read.
Qwen3-Coder-30B holds 96 KiB of KV per token (48 layers × 4 KV heads × 128 dims
× 2 for K and V × 2 bytes — read out of the artifact's own config, not assumed),
so a 16k context is 1.61 GB. Each extra row spends 11.5 ms on it: **140 GB/s**,
against the ~203 GB/s the weight read achieves on this machine. That is close
enough to say what is happening — every query row streams the entire cache for
itself, at roughly the speed the hardware can stream it.

**The weight half.** `10.19 + 2.49·rows` ms. This one is expected: in a
mixture-of-experts at batch sizes this small, distinct tokens route to distinct
experts, so more rows genuinely means more expert weights read. It is not a
defect and there is no obvious fix.

The KV half is the defect, and the driver already names it. `sdpa_paged.metal`
has two attention shapes:

- `sdpa_paged_decode` — one query row per threadgroup, its 32 simdgroups
  splitting that row's *keys*. Right for a decode, where there is only one row.
- `sdpa_paged_tiled` — 32 query rows per threadgroup, staging a block of keys
  into threadgroup memory that all 32 rows read. Its own comment: a fire of N
  rows otherwise "reads the prefix" once per row.

Which one runs is `sdpa_should_tile(rows, requests)`, and the crossover is
`sdpa_tile_min_rows_per_request = 32` (`driver/metal/src/device_tuning.hpp:263`).
A 5-row verify fire from one request scores 5. It takes the per-row kernel and
reads the cache five times.

The threshold is not wrong for the cases it was tuned on. A 32-request fleet of
one-row decodes measured 370 tok/s tiled against 728 per-row, so the tiled
kernel must be kept off it; a prefill is one request contributing thousands of
rows and wins outright. **A speculative verify fire is a third shape neither
kernel was written for** — few rows, one request, all sharing a key span — and
it falls to the worse of the two. What it wants is the per-row kernel's
key-parallel decomposition with each simdgroup computing all k rows against the
keys it has already loaded: one KV read, k dot products per key. Decode is
bandwidth-bound here, so the extra arithmetic is close to free.

**What that implies before any end-to-end run.** A `(1+k)`-row fire has to
produce this many tokens just to break even:

| rows | at 2k context | at 16k context | ceiling at 16k |
|---:|---:|---:|---:|
| 2 | 1.26 of 2 (26% acceptance) | 1.54 of 2 (54%) | 1.30× |
| 5 | 1.99 of 5 (25%) | 2.82 of 5 (46%) | 1.77× |
| 8 | 2.82 of 8 (26%) | 4.08 of 8 (44%) | 1.96× |

Speculation gets *harder* as the context grows, which is backwards from what an
agent turn needs.

---

## 2. Agent-shaped output: 4 turns, canned opencode transcript

`bench_ab.py --decode-probe`, which appends an identical request for a long
plan to every turn on every arm. ~2.1–2.7k-token prompts, 906 generated tokens
per pie arm and 772 per vLLM arm.

| | total | decode tok/s per turn | TTFC turn 1 |
|---|---:|---|---:|
| pie, spec on | 21.78 s | 48.5 / 61.1 / **80.3** / 58.3 | 6.47 s |
| pie, spec off | 22.43 s | 63.0 / 62.9 / 62.0 / 61.2 | 6.35 s |
| vLLM, spec on | 14.97 s | 60.7 / 60.6 / 59.3 / 58.6 | 1.27 s |
| vLLM, spec off | 15.02 s | 60.8 / 60.6 / 59.4 / 58.8 | 1.28 s |

pie's own accounting for those four turns: **7%, 28%, 58%, 26% accepted**
(6/92, 53/192, 194/336, 44/168 drafted). Set that beside the per-turn decode
rate and the break-even table above and every turn is explained:

- turn 3, 58% accepted → 80.3 tok/s, **1.30×** over its control;
- turn 1, 7% accepted → 48.5 against 63.0, **0.77×**. A pure loss, exactly as
  the 25%-at-2k break-even predicts;
- the wins and losses nearly cancel: **1.03×** overall.

Both pie arms emitted identical token counts on every turn (80/222/400/204),
which is the correctness check: verified speculation must not change the output,
and it did not.

vLLM's two arms are identical to within 1% on every turn.

## 3. Best-case output: what speculation is worth when drafting cannot fail

Same harness, output that is a verbatim twenty-times repetition — so nearly every
token is a copy of one a few positions back, and a 2/3-gram lookup cannot miss.

| | total | decode tok/s | acceptance |
|---|---:|---|---|
| pie, spec on | **10.18 s** | 88.7 / **136.9** | **98%, 97%** (316/321, 269/276) |
| pie, spec off | 14.73 s | 63.9 / 63.0 | — |
| vLLM, spec on | 13.63 s | 61.3 / 60.5 | not reported (see below) |
| vLLM, spec off | 13.78 s | 61.1 / 60.3 | — |

pie: **2.17× on decode**, 1.45× on wall clock. The probe predicted a 2.51×
ceiling at this context and 5 rows; 2.17× at 98% acceptance sits just under it,
which is the agreement worth having — the kernel measurement predicted the
end-to-end result rather than being fitted to it.

vLLM: **1.003×**. On output that is essentially free to draft.

### On vLLM's zero

Do not read that from the Prometheus counters. They report
`spec_decode_num_drafts_total 0` — but `SpecDecodingStats` appears nowhere in
the `vllm_metal` plugin, so those counters are never written on this backend and
the zero means "not reported", not "not drafted". That is why section 3 exists:
it asks the question functionally instead. The configuration is definitely live
— `ngram_proposer.py:116` logs `N-gram speculative decoding enabled
(prompt_lookup=[2, 3], num_speculative_tokens=4)` on the spec arm and not on the
control, and the engine disabled async scheduling because of it. It simply
produces no measurable speedup, at any draftability tested.

So the fair comparison the user asked for is fair in setup but one-sided in
outcome: **pie's speculation functions and vLLM's, in this build, does not.**
Enabling it on both did not change the ranking on agent-shaped work, and it
reversed the ranking on draftable work.

## 4. Where the remaining gap actually is

With the tools included, prompts at 7.2–8.2k, six turns, ~10 generated tokens
per turn — i.e. dominated by prefill:

| | total | per turn (2–6) | TTFC (2–6) | prompt reuse |
|---|---:|---:|---:|---|
| pie B | 30.6 s | ~1.50 s | ~1.24 s | 7211 of 7403 cached |
| vLLM | 10.2 s | ~0.65 s | ~0.45 s | 46.5% hit rate (its own log) |

pie is reusing essentially the whole prefix and is still 2.3× slower per turn.
At 7.4k context a decode step costs ~21 ms, so ten tokens is ~0.2 s of the
1.50 s; the ~190 fresh prompt tokens are a fraction of that again. **Roughly a
second per turn is neither prefill nor decode.**

**RESOLVED — see `results-turn-latency.md`.** It was prefill after all, but not
for any reason visible from outside the fire: the 189-row prefill fire lands in
a driver row-count class that costs a flat ~560 ms extra (`r mod 8` in 1..6),
and 189 rows measured 1147 ms against 583 ms for *192* rows. Chunking the span
so no fire lands in that class took the steady turn from 1.53 s to 0.97 s and
the whole 6-turn replay from 30.6 s to 18.1 s. The table above is therefore the
BEFORE state; pie's steady turn is now 0.97 s against vLLM's 0.65 s.

(vLLM does not populate `cached_tokens` in the OpenAI usage payload, which is
why its column reads 0 in the raw logs. Its own engine log reports the hit rate.)

## 5. What to do next, in order

1. **Find the ~1 s per-turn overhead in strategy B.** Biggest number, no kernel
   work required. Instrument the shim/gateway/guest boundary and attribute it.
2. **A small-k verify kernel.** Keep `sdpa_paged_decode`'s key-parallel shape,
   compute all k query rows per loaded key block. Turns the 5-row fire at 16k
   from 2.82× a 1-row fire into something near 1.2×, which takes speculation's
   ceiling there from 1.77× to ~4×, and makes it pay at ordinary acceptance
   rather than only at 98%.
3. **Then re-run this comparison.** Only after (2) does "pie vs vLLM with
   speculation" measure the drafting rather than the dispatch threshold.

## Reproducing

```sh
# the kernel measurement
cd runtime/engine/tests/inferlets && cargo build --target wasm32-wasip2 --release -p decode-rows-probe
./target/release/pie -c /tmp/rows-probe/config.toml run \
  --path runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm \
  --manifest runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml

# the four-cell comparisons (scratchpad drivers)
bash matched_spec.sh          # prefill-shaped, 6 turns
bash matched_spec_decode.sh   # agent-shaped decode, 4 turns
bash spec_ceiling.sh          # best-case draftability, 2 turns
```
