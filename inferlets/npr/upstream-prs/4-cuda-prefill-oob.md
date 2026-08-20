## The bug

In `llama_like_forward_paged` (`driver/cuda/src/model/llama_like.cpp`), the logits tail runs unconditionally after the decoder layers. It decides how many rows to feed the lm_head like this:

```cpp
const bool compact_logits =
    logit_row_indices_d != nullptr && num_logit_rows > 0 && num_logit_rows < N;
...
int lm_head_rows = N;
if (compact_logits) { ...gather...; lm_head_rows = num_logit_rows; }
...
kernels::launch_gemm(..., ws.logits.data(), lm_head_rows, ...);
```

When `num_logit_rows == 0` — a batch with no sampling rows at all, e.g. a pure prefill / context flush that will never read logits — `compact_logits` is false, so `lm_head_rows` stays at `N`, the full token count of the batch.

But `ws.logits` is sized by the memory planner as `[max_logit_rows, V]`, and `max_logit_rows` is `output_rows` = the planner's **request cap** `R0` (typically 128–512), not `N`.

## The failure it causes

**Out-of-bounds GEMM write, and the process dies.** Any prefill longer than `R0` tokens makes the lm_head write `N * V` floats into a buffer sized for `R0 * V`. In practice this surfaces as a CUDA illegal memory access or `cuBLAS EXECUTION_FAILED`, and the CUDA context is poisoned — every subsequent launch on that context fails, so it is not a recoverable per-request error.

The trigger is just "a prompt longer than the request cap", which is an ordinary thing for a server to receive. It was hit immediately by an ~850-token prompt fill on a default configuration.

## The fix

Return before the logits tail when the batch produces no sampling rows and is not pure decode:

```cpp
if (num_logit_rows == 0 && !is_pure_decode) {
    return;
}
```

The three other meanings of `num_logit_rows` are preserved exactly:

- **pure decode** passes `num_logit_rows == 0` with `N == R` — it both needs its lm_head and fits the buffer (this is the shape graph capture records), so it is explicitly excluded from the early return;
- `num_logit_rows == -1` still means "full logits over every row" (MTP verify);
- `num_logit_rows > 0` still selects the compact gathered rows.

So the only behaviour that changes is the case that was previously an OOB write into a buffer nobody was going to read.

## Verification

- Read against current `main`: `ws.logits` is planned at `max_logit_rows = output_rows`; with `num_logit_rows == 0` and `N > R0` the GEMM writes past it. The guard is the first thing between the decoder loop and the lm_head, so no other path is affected.
- Behaviourally: reproduced on H200 as a hard CUDA failure on the first prefill over the request cap, and fixed by this change. The patched driver has since run long multi-hour workloads on H200 and L40S (including a 50-run evaluation sweep with 674k grafted tokens) with no logits-buffer faults.
- **Not compile-checked locally, and CI will not compile it either** — I have no CUDA toolchain on this host, and `ci.yml` (the workflow that runs on PRs into `main`) deliberately skips the cmake-driven C++ build; `build.yml`, which has the CUDA matrix, is `workflow_dispatch` only. So this change reaches you read-verified and GPU-runtime-verified but not compiler-verified in this repo's automation. It is a single early `return` in a `void` function, adding no symbols and no includes, so the compile risk is about as low as it gets — but if you would like a maintainer with a CUDA box to confirm before merging, that is a reasonable ask and I have no way to do it myself.

Related: the portable driver had the same "logit rows == request count" assumption in its sampling tail, fixed separately in #426.
