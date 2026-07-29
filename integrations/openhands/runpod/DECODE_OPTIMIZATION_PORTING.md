# Porting the decode optimization to `main`

Everything needed to reproduce the Qwen3.5/Qwen3.5-MoE decode speedup on a clean
branch, plus the measurement rig that proves it landed. Written to be followed
without the originating session's context.

Branch it was developed on: `openhands-integration-updated` (`YangLiuWillow/pie`).
Hardware: H200 (sm90), Qwen3-Coder-30B-A3B-Instruct, TP1, bf16 KV, page_size 32.

---

## 1. What the change is, in one paragraph

Pie's `qwen3_5` / `qwen3_5_moe` forward decodes through
`flashinfer::BatchDecodeWithPagedKVCacheDispatched` — FlashInfer's **CUDA-core**
decode kernel. FlashInfer also has a **tensor-core** path for decode, which it
implements by calling the *paged prefill* kernel at `qo_len=1`. Pie already had
the plumbing to route decode that way (`force_prefill_path`), but only ever
reached it as a **fallback** for GQA ratios outside FlashInfer's decode dispatch
set `{1,2,3,4,8}`. Qwen3-Coder-30B-A3B is ratio 8 — *in* the set — so it always
took the slow path. Being "supported" by the CUDA-core kernel is exactly what
kept it off the fast one. The fix flips that fallback into a deliberate default
for GQA group ≥ 4.

**Result: KV read 1,748 → 4,068 GB/s (36% → 85% of the H200's ~4.8 TB/s peak,
past vLLM's 3,708), decode +20.8% end-to-end, aggregate throughput 1.15×–2.34×
better at every concurrency from 1 to 96.**

---

## 2. The code change

Three files, ~25 lines. No new CUDA — the prefill kernel is already compiled and
already used for prefill.

### 2a. `driver/cuda/src/model/qwen3_5_config.hpp`

Declare the knob next to the other `qwen35_*` env helpers:

```cpp
bool qwen35_tensor_core_decode_enabled(int gqa_group);
```

### 2b. `driver/cuda/src/model/qwen3_5_config.cpp`

```cpp
bool qwen35_tensor_core_decode_enabled(int gqa_group) {
    // -1 = unset, decide from the ratio; 0/1 = explicit override.
    static const int forced = [] {
        const char* v = std::getenv("PIE_QWEN35_TENSOR_CORE_DECODE");
        if (v == nullptr || v[0] == '\0') return -1;
        return v[0] != '0' ? 1 : 0;
    }();
    if (forced >= 0) return forced == 1;
    return gqa_group >= 4;
}
```

`gqa_group >= 4` is FlashInfer's own threshold for the tensor-core path paying
off. Ratios below 4 keep the CUDA-core kernel — untested here, and not expected
to benefit.

### 2c. `driver/cuda/src/entry.cpp`

Two construction sites, `is_qwen3_5_arch` and `is_qwen3_5_moe_arch`. Each passes
`force_prefill_path`. Change:

```cpp
/*force_prefill_path=*/!pie_cuda_driver::flashinfer_decode_supports_gqa(gqa_q_moe),
```

to:

```cpp
/*force_prefill_path=*/
    pie_cuda_driver::model::qwen35_tensor_core_decode_enabled(gqa_q_moe) ||
    !pie_cuda_driver::flashinfer_decode_supports_gqa(gqa_q_moe),
```

(and the same with `gqa_q` at the non-MoE site).

### 2d. The banner — do not skip this

Print the resolved decision **at the qwen3_5_moe construction site**, not in the
`model_type=` banner:

```cpp
{
    const bool tc = pie_cuda_driver::model::qwen35_tensor_core_decode_enabled(gqa_q_moe);
    const bool in_set = pie_cuda_driver::flashinfer_decode_supports_gqa(gqa_q_moe);
    std::cerr << "[pie-driver-cuda] qwen3.5-moe attention: gqa=" << gqa_q_moe
              << " flashinfer_decode_supports_gqa=" << (in_set ? 1 : 0)
              << " tensor_core_decode=" << (tc ? "on" : "off") << "("
              << (std::getenv("PIE_QWEN35_TENSOR_CORE_DECODE")
                      ? "PIE_QWEN35_TENSOR_CORE_DECODE" : "default: gqa>=4")
              << ")"
              << " -> decode kernel="
              << ((tc || !in_set) ? "paged-prefill (tensor core)"
                                  : "BatchDecodeWithPagedKVCache (cuda core)")
              << "\n";
}
```

**Why the placement matters.** The `model_type=…` banner prints `llama_like`'s
`fwd_cfg`, which `qwen3_moe` never executes. Reading a feature off that line is
what made `xqa_decode=on` mislead the originating investigation for a day — the
model routes through `qwen3_5*`, and `grep -c xqa qwen3_5_moe_forward.cpp` is 0.
Verify the fix landed by reading *this* line, not that one.

---

## 3. Build

```bash
source /workspace/pie-bench-env.sh   # sets CMAKE_CUDA_ARCHITECTURES=90, CPM cache, HF_HOME
cargo build -p pie-server --release --features driver-portable,driver-cuda
```

~30 s incremental. Confirm the new code is actually in the binary before
measuring anything:

```bash
strings target/release/pie | grep -m2 'PIE_QWEN35_TENSOR_CORE_DECODE\|paged-prefill (tensor core)'
```

---

## 4. Verifying it worked

Two files carry the rig. Port both — the numbers below are meaningless without
them, and the method is the part that is easy to get subtly wrong.

- `integrations/openhands/runpod/context_sweep_client.py`
- `integrations/openhands/runpod/42_context_sweep.sh`

```bash
# decode ms/token vs context, batch 1, both arms
bash 42_context_sweep.sh
# aggregate throughput vs concurrency at fixed context (pie only)
SWEEP_MODE=batch SWEEP_ARMS=pie bash 42_context_sweep.sh
# A/B the change itself
PIE_QWEN35_TENSOR_CORE_DECODE=0 SWEEP_ARMS=pie bash 42_context_sweep.sh   # old path
SWEEP_ARMS=pie bash 42_context_sweep.sh                                   # new path
```

### How the measurement works, and why it is built this way

**Differencing.** Issue the same prompt twice, once generating 8 tokens and once
136, and subtract:

```
decode_ms_per_token = (lat_long - lat_short) / (tok_long - tok_short)
```

Prefill, chat rendering, transport and process launch are all
generation-independent, so they cancel. This is what makes Pie (websocket +
inferlet) and vLLM (HTTP) comparable at all — any constant per-call overhead,
however large, drops out. Token counts come from each engine's own report, so an
early EOS shortens the long run without biasing the ratio.

**Warm the prefix first.** Both engines cache prefixes. Without a warmup call the
short run pays full prefill and the long run pays none, and the subtraction
silently returns *decode minus prefill*. Every context gets a throwaway call
first.

**Slope and intercept are the answer, not the raw rate.** Fitting ms/token
against context splits decode into a context-independent part (intercept:
weights, MoE, router, launch) and a KV-proportional part (slope: attention). At
96 KiB of KV per token for this model — 48 layers × 4 KV heads × 128 dim × 2 (K
and V) × 2 bytes — the slope converts directly to effective KV read bandwidth.
Those two numbers say which half of decode you are actually looking at.

**Cross-check.** Pie's differenced number tracks its own
`timings.decode_ms` within 1.5% at all six contexts. If a port disagrees with
its own instrumentation by more than a few percent, the rig is wrong, not the
engine.

### Expected numbers (H200, Qwen3-Coder-30B-A3B, page_size 32)

| | off | on | vllm (FA3) |
|---|---|---|---|
| intercept (fixed per-token) | 5.496 ms | **5.030 ms** | 4.173 ms |
| slope per 1k KV tokens | 0.0562 ms | **0.0242 ms** | 0.0265 ms |
| implied KV read | 1,748 GB/s (36%) | **4,068 GB/s (85%)** | 3,708 GB/s (77%) |
| R² of the fit | 0.952 | — | 0.998 |

Decode ms/token by context:

| prompt tokens | off | on | speedup |
|---|---|---|---|
| 1,036 | 5.449 | 5.053 | 1.08× |
| 8,020 | 5.924 | 5.230 | 1.13× |
| 16,012 | 6.234 | 5.419 | 1.15× |
| 24,004 | 7.079 | 5.626 | 1.26× |
| 31,996 | 7.191 | 5.791 | 1.24× |

Aggregate decode tok/s vs concurrency:

| R | 16k off | 16k on | | R | 4k off | 4k on |
|---|---|---|---|---|---|---|
| 1 | 159.9 | 184.6 | | 16 | 945.9 | 1168.5 |
| 8 | 451.0 | 764.3 | | 32 | 1390.5 | 1998.6 |
| 16 | 538.6 | 1014.3 | | 48 | 1712.6 | 2818.0 |
| 24 | 657.6 | 1338.6 | | 64 | 2105.9 | 3946.4 |
| 32 | 679.9 | **1592.2** | | 96 | 1915.0 | **4356.3** |

End-to-end on `django__django-14373` (SWE-bench, temperature 0): decode
**147.5 → 178.2 tok/s (+20.8%)**, valid patch, no errors.

---

## 5. Traps that cost time — read before debugging a bad port

- **`full_attn_min_R=256` in the banner is inert for this model.** It looks like
  it routes decode to the full-attention kernel above a batch threshold. It is
  consumed only in `llama_like.cpp`, a path `qwen3_moe` never executes. Two
  separate agents have now misread this symbol.
- **Do not measure a rate under `PIE_QWEN35_MOE_PROFILE=1`.** It puts a
  `cudaEventSynchronize` around every stage, inflating leaf times ~30%,
  suppressing overlap, blocking graph capture, and depressing throughput enough
  to trip client websocket timeouts. It reported `full_attn` at 47% of decode
  where the profiler-free slope says 20%. Use it for *shares*, never rates, and
  re-measure rates unprofiled.
- **A `rc=0` run can still be garbage.** The profiled run above returned rc=0
  with `agent_iterations=0`, an empty patch, and a decode rate that looked
  entirely plausible. Always check `_metadata.error` and `agent_iterations`.
- **The R≈48-at-16k collapse is the KV pool, not the kernel.** 48 × 16,012 =
  768,576 tokens against `kv_tokens=632352` (printed in the memory-planner
  banner). Both arms thrash identically. Keep `R × ctx` under the pool or the
  result is meaningless.
- **`pkill -f 'pie serve'` matches your own shell.** Kill by PID; verify with
  `ss -ltnp` and `nvidia-smi`, not `pgrep`.
- **`30_ab_run.sh` does not source `pie-bench-env.sh`.** Forget it and `HF_HOME`
  falls back to `$HOME/.cache/huggingface`, and the engine starts a silent 57 GB
  re-download that fills the local disk. A guard now aborts instead; port it.

---

## 6. What this does NOT fix

The intercept — the context-independent 5.030 ms/token against vLLM's 4.173,
about **1.21×** — is untouched. That is the weights / MoE / router / launch path.
Kernel *selection* there has already been ruled out by measurement: of Pie's
three MoE decode paths, the `cublasGemmBatchedEx` M=1 default beat both
alternatives at batch 1 (WMMA −41%, aligned-gate-lowered −9.5%), and KV page size
16 vs 32 was a wash. Whatever closes the intercept is not a flag.
