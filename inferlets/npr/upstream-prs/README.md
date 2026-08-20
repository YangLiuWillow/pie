# The five upstreamable pie fixes

HANDOVER.md §4 flags five commits on `npr-inferlet` as bug fixes independent of
NPR, worth separate PRs to `pie-project/pie`. This directory holds the prepared
PR bodies and the script that opens them, so the work survives a lost clone.

## Which upstream branch — verified 2026-08-20, do not re-derive

**PRs target `pie-project/pie` `main`.** Not `dev`.

- `94043eb12`, the base of `npr-inferlet`, lives on `fork/dev`
  (`YangLiuWillow/pie`) and on **no** `pie-project/pie` branch.
- Upstream `dev` is a different codebase generation — a rewrite carrying
  `runtime/engine/`, `compiler/`, `controller/`, `model/`, `worker/`, and
  **no `driver/portable`**, no `sdk/rust/inferlet/src/context.rs`.
- Every file these fixes touch exists on upstream **`main`**, which is also the
  repo's default branch. `.github/workflows/ci.yml` runs only on `push`/
  `pull_request` into `main` (made explicit upstream in #516).
- Merge-base of `fork/dev` with upstream is `8824d3e5b`, which is on both
  upstream `main` and `dev`.

## The branches

Cut from `upstream/main`, one focused commit each, pushed to `fork`:

| branch | from | body | applied |
|---|---|---|---|
| `fix/sdk-context-destroy-trap` | `c0218d2` | `1-sdk-destroy.md` | clean |
| `fix/portable-kv-write-index` | `e10b9ac` | `2-portable-kv-write-index.md` | clean |
| `fix/runtime-explicit-mask-commit` | `86ed682` | `3-runtime-commit-check.md` | clean |
| `fix/cuda-prefill-logits-oob` | `8de3f5e` | `4-cuda-prefill-oob.md` | clean |
| `fix/portable-graph-cache-slot-count` | `b5380d3` | `5-portable-graph-cache-slots.md` | **rebased** |

## Two things that had drifted

**`b5380d3` is half obsolete.** Upstream merged its `ggml_reshape_3d` half as
PR #426 ("key uniform-sample reshape on slot count, not request count"); all
five graph builders on `main` now derive `n_slots` from `probs->ne[1]`. That
half was dropped. What is still live is the *other* half: `GraphCache::matches()`
does not compare the sampling-slot count, so a cached graph can be reused at a
count its `out_idx` — sized to exactly `plan.sampling_pos_i32.size()` — was
never built for, and `upload_graph_inputs` then overruns or under-fills it. That
half was rebased on its own and the branch renamed to say what it actually does.
It fails *silently* (wrong tokens) where #426's half failed loudly (GGML_ASSERT).

**`c0218d2` overlaps an upstream PR that never landed on `main`.** PR #484
("Fix borrowed context destruction at the component boundary") fixes the same
defect from the host side — it drops `table.delete(this)?` from
`HostContext::destroy` and lets the ordinary resource drop own deletion. It
merged into `tts-arena/main`, a branch that no longer exists; `main`'s `destroy`
still calls `table.delete(this)?`. The two fixes are **alternatives**: applying
both leaves nothing to delete the table entry, leaking a slot per destroyed
context. The PR body says so and names #484.

## Verification status

| fix | compile | behaviour |
|---|---|---|
| sdk destroy | `cargo check --target wasm32-wasip2` clean | destroy-heavy fork/join runs stable where they used to trap |
| portable KV write index | `cmake -DGGML_METAL=ON` build clean | selftest oracle TV = 0.0000 on Qwen3-0.6B (causal control 0.22–0.55) |
| runtime commit check | `cargo check`/`cargo test --workspace --exclude pie-server-py` — `pie` suite fully green | sibling branches commit and the merged context decodes correctly |
| CUDA prefill OOB | **not compiled anywhere** — no local CUDA toolchain, and upstream CI does not build CUDA on PRs (`ci.yml` skips the C++ build; `build.yml` is `workflow_dispatch`) | reproduced as a hard CUDA failure on H200 and fixed; patched driver ran multi-hour H200/L40S workloads |
| portable graph cache | `cmake -DGGML_METAL=ON` build clean | reasoning-only; conservative (a signature field can only add cache misses) |

Unrelated but worth knowing: `cargo test --workspace` is **red out of the box on
macOS** on unmodified upstream `main` — 7 `pie-bridge` shmem tests fail because
macOS caps POSIX shm names at 31 characters and the test helper builds longer
ones (`shm_open` → `ENAMETOOLONG`). Confirmed identical on a clean
`upstream/main` checkout, so it is not fallout from any of these fixes.

## Opening them

`gh pr create` does not work on this host — the fine-grained PAT cannot run
GraphQL mutations ("Resource not accessible"). `open-prs.sh` uses
`gh api -X POST repos/pie-project/pie/pulls` instead.
