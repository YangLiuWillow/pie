## The bug

`ContextManager::commit_working_pages` (`runtime/src/context.rs`) requires every position in a committed batch to be strictly greater than the context's committed watermark:

```rust
if let Some(max_committed) = ctx.max_committed_position {
    for &pos in &positions {
        if pos <= max_committed {
            anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
        }
    }
}
```

That is the right check for a default-mask fill, where a position regression is always a bookkeeping bug. It is wrong for tokens that carry an **explicit per-token attention mask**, which is precisely the guest's way of saying "I have decoupled position ids from KV-slot order on purpose".

## The failure it causes

Any guest that fills a context at positions that overlap already-committed ones — with explicit masks describing exactly what each token may see — cannot commit. `commit` bails, and the tokens sit uncommitted in working pages. Concretely this blocks parallel-branch workloads: sibling branches that decode from a shared prefix restart at the same position and must be merged back into one context at their original positions. The engine refuses the merge, so the pattern is unavailable from guest code even though the KV layout supports it.

Second, smaller defect in the same function: the watermark update

```rust
ctx.max_committed_position = positions.iter().copied().max().or(ctx.max_committed_position);
```

takes the batch max unconditionally. Once lower-positioned batches are allowed at all, that lets the watermark **shrink**, which would then wrongly re-admit a genuinely regressive default-mask fill.

## The fix

- Exempt explicit-mask tokens from the strict check; default-mask fills keep it unchanged. The batch's `committed_token_infos` are already in hand, so the per-token mask is available at the check site — `explicit_mask` is simply "this token's lineage mask is non-empty".
- Make the watermark monotone: `.max(ctx.max_committed_position)`.

## Why non-monotonic positions round-trip correctly

Dedup and restore both carry the position and the mask, not just the token:

```rust
let hashes = pagestore::compute_page_hashes(
    page_size, &tokens, &positions, &masks, prev_hash, lineage_adapter_seed);
```

`masks` here is `materialize_lineage_mask(&info.mask, info.position)` per token. So two pages with the same tokens but different positions or different masks hash differently and are never conflated in the trie, and the committed-chain replay used on restore regenerates the same `(token, position, mask)` triples. Nothing downstream of the check depends on positions being monotone.

## Verification

- `cargo check --workspace --exclude pie-server-py --all-targets`: clean.
- `cargo test --workspace --exclude pie-server-py --no-fail-fast` on macOS/arm64: the whole `pie` runtime suite is green — 439 lib unit tests, plus `context` (6), `e2e` (7), `program` (6), `contention` (9), `structured` (292), `rs_cache`, `smoke`, `tokenizer`, `tokenizer_compat`. The only failures are 7 `pie-bridge` shmem tests, which fail **identically on unmodified `main`**: macOS caps POSIX shared-memory names at 31 characters and the test helper builds longer ones, so `shm_open` returns `ENAMETOOLONG` before any of this PR's code runs. (Worth fixing separately — it means `cargo test --workspace` is red out of the box on macOS — but it is unrelated to this change and I have left it alone.)
- Behaviourally: with this fix a fork/join inferlet commits sibling branches back into a shared context at overlapping positions and the merged context continues decoding correctly; the join is numerically validated separately against a straight-line decode (TV = 0.0000 on Qwen3-0.6B) on the portable driver, and has since run for hours on CUDA (L40S, 50-run sweep) with no commit rejections.
- Default-mask behaviour is unchanged: the strict check still runs for every token whose lineage mask is empty, which is every token produced by ordinary decoding.
