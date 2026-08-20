## The bug

Sampling slots in the portable driver come from `plan.sampling_pos_i32`, which is **flat across requests and independent of `n_request`**. `graph_common.cpp` already says so where it allocates the input tensor:

```cpp
// out_idx may exceed n_req when M8 spec decode adds per-draft slots.
const std::int32_t n_sample_slots =
    static_cast<std::int32_t>(plan.sampling_pos_i32.size());
in.out_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sample_slots);
```

`Executor::GraphCache::matches()` does **not** compare that count. Its signature is arch, `pure_decode`, `n_request`, `total_n_tokens`, `max_n_kv`, the sampler flags, the adapter, `total_pages_in_batch` (pure decode), the per-request `n_tokens_pad`/`n_kv` (slow path) and `state_slot_per_req`. Two batches can agree on every one of those and still carry different slot counts — a prefill-only request contributes none, a spec-decode verify pass contributes several.

So a cached graph gets reused for a batch whose slot count its `out_idx` was never built for.

## The failure it causes

`upload_graph_inputs` writes the new plan's slots into the **cached** tensor:

```cpp
ggml_backend_tensor_set(g.in.out_idx, plan.sampling_pos_i32.data(), 0,
                        plan.sampling_pos_i32.size() * sizeof(std::int32_t));
```

- **More slots than the cached graph was built for → buffer overrun.** `ggml_backend_tensor_set` writes past the end of `out_idx`, corrupting whatever the scheduler placed after it.
- **Fewer slots → stale trailing entries.** `out_idx` keeps values from a previous batch, so `ggml_get_rows` gathers the wrong hidden rows.

Either way the whole sampling tail is off, because it is shaped from the same count: the lm_head output is `[vocab, n_slots]`, the top-K gather reshapes on `n_slots = probs->ne[1]`, and `compute_()` reads back `n_slots` rows using the *new* plan's count. The result is wrong tokens rather than a clean abort, which makes it considerably nastier to notice than the reshape assert that #426 fixed.

## The fix

Add `n_sample_slots` to the cache signature — compared in `matches()`, recorded in `store_key()` — exactly alongside `total_pages_in_batch`, which exists in that signature for the same class of reason (a tensor sized from a plan array that can change while everything else in the signature stays put).

Three lines plus a comment; no behaviour change when the slot count is stable, which is the common case.

## Relationship to #426

#426 fixed the *other* half of this assumption: the uniform-top-K reshape keyed on `n_req` instead of the slot count, which tripped `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)`. That is merged; all five graph builders on `main` now derive `n_slots` from `probs->ne[1]`.

This PR is the cache-signature half of the same defect, which #426 did not touch. It is independently reachable, and it fails silently rather than loudly — after #426 the graph *shape* is right for the batch it was built from, which is precisely what makes reusing it for a different slot count dangerous.

## Verification

- Read against current `main`: `matches()`/`store_key()` do not reference `sampling_pos_i32`; `out_idx` is allocated to exactly its size at build time and written to its full size at upload time.
- `cmake -S driver/portable -B build -DGGML_METAL=ON && cmake --build build` on macOS/arm64: clean.
- Conservative by construction: adding a field to the cache signature can only cause *more* cache misses (a rebuild), never fewer, so the worst case of a wrong analysis here is a rebuilt graph, not a behaviour change.
