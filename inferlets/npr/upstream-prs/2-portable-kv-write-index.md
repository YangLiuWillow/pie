## The bug

`driver/portable/src/plan.cpp` contains two places that assume a token's **position id** equals its **KV slot index**. That holds for ordinary causal decoding and nowhere else.

**1. Position used as the KV write index** (`plan_single_request`):

```cpp
const std::int32_t pos_i = static_cast<std::int32_t>(a.position_ids[i]);
if (pos_i >= seq_len) { throw ... "exceeds seq_len" ... }
plan.kv_idxs_i64[i] = physical_idx(a.kv_page_indices.data(), pages_off, page_size, pos_i);
```

The write target for a new token must be its **slot** — append order after the pre-pass KV — not its position id. Position ids feed RoPE only.

**2. Custom attention-mask rows clamped at the token's position** (`build_attn_mask_f16`):

```cpp
const std::int32_t hi = std::min(n_kv - 1, p_i);
```

applied to *custom BRLE rows* as well as to the synthesized causal default. Custom rows are expressed in **slot** space, so clamping them at `p_i` hides every KV slot past the token's (possibly compressed) position.

## The failure it causes

**Silent KV corruption.** Any fill whose position ids are deliberately decoupled from slot order overwrites live prefix KV in place: `physical_idx(..., pos_i)` resolves to a slot that already holds a committed token, and the driver writes over it. The context is corrupted with no error — subsequent decodes from that context produce garbage, and because the damage is in the KV cache rather than in control flow, nothing upstream reports a fault.

The forward-shifted case fails the other way: `pos_i >= seq_len` throws outright, rejecting a legitimate fill.

For (2), a token whose custom mask row is clamped at a position below its own slot **cannot attend to itself** — every row past `p_i` is forced to `-INF`, including the token's own slot.

Both are reachable from ordinary guest code: any inferlet that writes tokens at positions it chooses rather than at `seq_len` (parallel branch refill at overlapping positions, position-compressed KV, prefix re-anchoring) hits them.

## The fix

1. Write index = slot: `kv_before_r + (i - qo_start)` where `kv_before_r = seq_len - n_tok` (the pre-pass KV length, i.e. the first new token's slot). Position ids stay in `plan.positions_i32` for RoPE. The old `pos_i >= seq_len` check is replaced by a check that the request actually has room for its tokens.
2. Custom BRLE rows are no longer clamped at `p_i` — the runs alone define visibility, since the runtime emits rows whose true-runs never extend past the slots a token may legitimately see. **The synthesized causal default keeps its clamp**, so default-mask behaviour is byte-identical.

## The other clamp in this file is correct — do not "fix" it too

`plan.cpp` contains a second `std::min(n_kv - 1, p_i)` at line 38, in
`build_phi3small_blocksparse_mask_f16`. This PR deliberately leaves it alone, and
a reviewer checking for a missed instance of the same pattern should know why.

That builder *synthesizes* a causal blocksparse mask. It takes no
`per_token_runs` parameter, so it has no custom-BRLE path at all — every row it
writes is causal by construction, and clamping at the token's own position is
exactly right there. It is the same reason this PR keeps the clamp on the
synthesized-causal branch of `build_attn_mask_f16` and removes it only from the
custom-rows branch.

The defect is not "the file clamps"; it is "the file clamps *guest-supplied
slot-space rows* as though they were positions". There is one such site, and this
PR changes that one.

## Verification

- Numeric oracle: an isolation-matrix selftest in which a sibling sequence is refilled at positions overlapping a shared prefix, behind explicit hole masks, and its next-token distribution compared against the same text decoded alone off that prefix. **TV = 0.0000** on Qwen3-0.6B with the fix; garbage output without it. A deliberately causal control in the same harness differs by TV 0.22–0.55, confirming the masks actually bite rather than the test being insensitive.
- `cmake -S driver/portable -B build -DGGML_METAL=ON && cmake --build build` on macOS/arm64: clean.
- Default-mask paths are unchanged by construction (the causal branch keeps the clamp; for an append-only fill `kv_before_r + i == pos_i`, so the new write index is the same value the old code computed).

Note for reviewers: the CUDA driver's `write_kv_kernel` was audited for assumption (1) and is already slot-correct, so this is portable-only.
