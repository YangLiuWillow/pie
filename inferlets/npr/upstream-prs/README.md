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

## Re-verified 2026-08-20 (post-preparation drift check)

`upstream/main` is unmoved at `202f5405b` — **0 commits** since the branches were
cut — and each branch is exactly one commit whose parent *is* that tip, with
`fork/<branch>` matching local. So mergeability is clean by construction, not by
inference.

More importantly, all five bugs were re-read on current `main` rather than assumed
from a clean cherry-pick:

| fix | still live on `main`? | evidence |
|---|---|---|
| SDK destroy | yes | `api/context.rs:172` still `table.delete(this)?`; SDK `destroy` still bare `self.inner.destroy()` |
| portable KV write index | yes | `plan.cpp:290` still `physical_idx(..., pos_i)`; custom-row clamp still at `:67` |
| runtime commit check | yes | `context.rs:1754` still `for &pos in &positions` with no explicit-mask exemption |
| CUDA prefill OOB | yes | no `num_logit_rows == 0` guard anywhere in `llama_like.cpp` |
| portable graph cache | yes | no `n_sample_slots` anywhere in `executor/executor.cpp` |

No `main`-targeted PR since the survey touches these areas (the recent ones are
all closed-unmerged dependabot chores), and `author:YangLiuWillow` still returns 0
PRs, so nothing of ours exists to duplicate.

Two things the re-check turned up that the first pass did not:

- **`plan.cpp` has a second `std::min(n_kv - 1, p_i)` clamp** at `:38`, in
  `build_phi3small_blocksparse_mask_f16`. It is **correct there** and must not be
  touched: that builder synthesizes a causal blocksparse mask and takes no
  `per_token_runs`, so it has no custom-BRLE path — the same reason our fix keeps
  the clamp on the synthesized causal branch. Checked because an incomplete fix
  would have been worse than none.
- **PR #467 addressed the CUDA prefill family and never reached `main`** (merged
  to `tts-arena/main`, now deleted), via a per-fire `emit_logits` flag plus a
  `test_executor_prefill_only.py`. Neither is on `main`. This is the **second**
  instance of the pattern — see also #484 for the SDK destroy trap. Both PR bodies
  now name their counterpart explicitly. Worth knowing before assuming a
  `main` gap means nobody has looked at it.

## Prior-art sweep — all five, all 428 upstream PR heads (2026-08-20)

Two of the five turned out to have prior art that never reached `main`, both
found before this sweep. That is a strong enough prior to check the other three
properly rather than by keyword search, because "we already fixed that in #NNN"
arriving as a maintainer comment on three PRs at once is the one outcome that
makes a batch look careless.

**Method.** `git fetch upstream '+refs/pull/*/head:refs/remotes/upstream-pr/*'`
pulls every PR's head commit — **including PRs merged into branches that were
later deleted**, which is exactly where the two known cases live and which no
GitHub search surfaces. Then, per target file, group all 428 heads by that file's
blob and inspect each distinct version that differs from `main`. `priorart.sh`
(next to this README) does the grouping.

**The first attempt was wrong, and the positive control is what caught it.**
`git grep -l <pattern> $REFS -- <path>` over ~428 revisions silently returns
nothing. It reported a clean null for all three files. Before believing it I ran
it against the two PRs I already knew contained the changes — **#467 and #484 —
and it found neither.** The null was an artifact of the instrument. The loop form
above passes both controls and was used for the real sweep.

That is the second false-negative search on this task (see BUG16-SCOPE.md R1, the
`fn bid` grep). The lesson generalises: **a search that returns nothing is a
measurement, and it needs a positive control before its null means anything.**

**Result.**

| PR | prior art | detail |
|---|---|---|
| 1 SDK destroy | **#484** | Same defect, fixed **host-side** (drop the `table.delete` from `HostContext::destroy`). Merged to `tts-arena/main`, since deleted; never reached `main`. No PR anywhere carries the SDK-side `mem::forget` fix. **Named in the body**; the two are alternatives and must not both land. |
| 2 portable KV write index | **none** | Four distinct non-`main` blobs of `plan.cpp`; every one still computes the write index as `physical_idx(..., pos_i)`, and none introduces a slot-order variant. (One blob matched a `slot_i` grep — false positive, it was `rs_slot_ids`, recurrent-state slots, unrelated.) |
| 3 runtime commit check | **none** | Eleven distinct non-`main` blobs of `runtime/src/context.rs`; every one still has the unconditional `for &pos in &positions` strict check, none mentions an explicit-mask exemption. |
| 4 CUDA prefill OOB | **#467** | Same family, fixed by threading a **per-fire `emit_logits`** flag, plus a `test_executor_prefill_only.py`. Merged to `tts-arena/main`, since deleted; never reached `main`. No PR anywhere adds the `num_logit_rows == 0` guard. **Named in the body.** |
| 5 graph cache slot count | **none** | Three distinct non-`main` blobs of `executor/executor.cpp`; the identifier `n_sample_slots` appears in none of them. Its sibling half is #426, already named in the body. |

**The pattern worth carrying:** twice, a fix for one of these landed on a branch
that was later deleted and never reached `main`. **A gap on `main` in this repo
does not mean nobody has looked at the bug.** Both affected bodies name their
counterpart and state that cherry-picking it supersedes ours, taking no position
on which the maintainers prefer.

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

## Opening them — currently blocked on the credential

**`gh` on this host cannot open a PR against `pie-project/pie` by any API.**
The binding constraint is the token type's *repository reach*, not the endpoint
— an earlier note in this project claimed the problem was GraphQL mutations
specifically and that `gh api -X POST` was the workaround. That is wrong for
repos we do not own.

`gh` is authenticated with a **fine-grained** PAT, which reaches only repos
owned by the token owner or granted by an org. Against `pie-project/pie` it is
public-read:

```
$ gh api repos/pie-project/pie --jq .permissions
{"admin":false,"maintain":false,"pull":true,"push":false,"triage":false}
```

so both of these return `403 Resource not accessible by personal access token`:

```
gh pr create ...                                        # GraphQL
gh api -X POST repos/pie-project/pie/pulls ...          # REST
```

**One-step diagnostic:** `gh api -i user | grep -i x-oauth-scopes` — **no
`X-OAuth-Scopes` header means the token is fine-grained.** Classic tokens
always emit it.

To unblock, either:

1. **Re-auth** — `gh auth login` (web flow; the OAuth app token carries `repo`
   scope and works across public repos), or a **classic** PAT with the
   `public_repo` scope. Then run `open-prs.sh`, **one call at a time**, checking
   the returned PR number before firing the next. A retry after a partial
   success is how duplicate PRs get opened on someone else's repo.
2. **Open them from the GitHub UI** — the branches are live on the fork, so each
   is one click plus a paste of the matching body file. Base must be `main` on
   every one.

   - https://github.com/pie-project/pie/compare/main...YangLiuWillow:pie:fix/sdk-context-destroy-trap?expand=1
   - https://github.com/pie-project/pie/compare/main...YangLiuWillow:pie:fix/portable-kv-write-index?expand=1
   - https://github.com/pie-project/pie/compare/main...YangLiuWillow:pie:fix/runtime-explicit-mask-commit?expand=1
   - https://github.com/pie-project/pie/compare/main...YangLiuWillow:pie:fix/cuda-prefill-logits-oob?expand=1
   - https://github.com/pie-project/pie/compare/main...YangLiuWillow:pie:fix/portable-graph-cache-slot-count?expand=1

Titles are in `open-prs.sh`. Commits are authored as Liu with no AI co-author
trailers — verified on the pushed `fork` refs before the first attempt, since
that is the one thing that cannot be corrected afterwards without a force-push.
