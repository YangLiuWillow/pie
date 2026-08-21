# NPR-on-Pie — Session Handover

*Written 2026-08-12 for a fresh Claude Code session on a new device. Everything
needed to resume is in this repo; nothing depends on the old machine.*

---

## 1. What this project is

Reimplementing **Native Parallel Reasoner** (NPR, arXiv 2512.07461 — the user is
a co-author) as a **pie inferlet**, replacing NPR's patched-SGLang "NPR Engine"
with guest-side WASM code on pie's programmable inference runtime.

NPR makes an LLM reason in parallel natively: it emits `<guideline>` with N
`<plan>` entries, forks N `<step>` branches that decode independently from a
shared prefix, then joins them into a `<takeaway>` synthesis. In the original
engine, that fork/merge logic is hacked into SGLang's scheduler. In pie it
becomes ~700 lines of ordinary inferlet code, because `Context::fork()` is an
O(1) copy-on-write over a content-addressed KV page trie and the engine batches
concurrent branch decodes automatically.

**Read `DESIGN.md` next** (same directory) — it has the verified mechanics of the
original engine, the pie mapping, the refill-join design, and per-phase
implementation notes with every engine bug found along the way.

## 2. Where everything lives

| What | Where |
|---|---|
| **Code + all docs** | `YangLiuWillow/pie` (fork of `pie-project/pie`), branch **`npr-inferlet`** |
| Branch base | `fork/dev` @ `94043eb12` (the pie dev branch, not `main`) |
| The inferlet | `inferlets/npr/` — `src/lib.rs` is the whole implementation |
| Design study | `inferlets/npr/DESIGN.md` |
| GPU pod bootstrap | `inferlets/npr/pod-setup.sh` |
| Client (launcher) | `inferlets/npr/client.py` |
| fp32→bf16 converter | `inferlets/npr/convert_bf16.py` |
| Example configs | `inferlets/npr/configs/*.toml` |
| Real GPU run transcripts | `inferlets/npr/runs/*.txt` |
| NPR paper code (reference) | `github.com/bigai-nlco/Native-Parallel-Reasoner` — clone if you need to re-check the original engine; DESIGN.md §1 cites exact file:line |
| NPR-4B checkpoint | `huggingface.co/bigai-NPR/NPR-4B` (fp32 — must convert, see §5) |

**On the old machine** the working tree lived at `~/Documents/Liszt_ai/`
(previously `~/Desktop/Lin_startup/`): `pie/` (main clone, `liu/qwen-code-dev`
checked out), `pie-npr/` (a **git worktree** of `pie/` holding the
`npr-inferlet` branch), `pie-rl/` (whose `.env` carries `RUNPOD_API_KEY=...`),
and `Native-Parallel-Reasoner/` (the paper's code release). Everything the
project needs is now committed, so a fresh clone suffices — but if you move that
tree to the new device, note that `pie-npr` is a worktree: it only works
alongside its parent `pie/` clone, and `git worktree repair` fixes the recorded
paths after a move. The RunPod key must be re-supplied on the new device. The
design study used to sit in pie's *gitignored* `/docs/`, which is why it now
lives in `inferlets/npr/DESIGN.md` instead of only on one machine.

## 3. Status: what works

Phases 1 (control loop), 2 (faithful refill join), and 3 (`adopt_kv` — the
faithful join as a device-side KV row copy instead of recomputation) are done.
Phases 1–2 are validated on real GPU hardware; phase 3 is validated on Metal
(selftest TV 0.0008 + AIME 2025 I/1 → 70 correct with 3,790 tokens adopted,
0 fallbacks) and **compile-checked only on CUDA** — see DESIGN.md §13.
Important: §13 also records why the §4 `adopt_pages` refcount-graft proposal
is structurally infeasible; don't resurrect it without reading that analysis.

- **Numeric correctness**: `--input '{"selftest": true}'` runs an isolation-matrix
  oracle proving the refill join reproduces the exact next-token distribution of
  a straight-line decode. Six equivalences, all TV = 0.0000 on CPU; within kernel
  noise (0.001–0.03) on CUDA. The causal control differs by TV 0.22–0.55, proving
  the masks actually bite.
- **Real model, real problems** (NPR-4B on an H200): the model drives its own
  format with zero prompting hacks. The paper's case-study domain problem →
  `(2,12) ∪ (12,102)` (matches paper Table 6). AIME 2025 I/1 → **70** (correct),
  with 2 parallel blocks, 5 branches, multi-chunk refill joins of 1.5k-token
  siblings, 4,005 tokens in 17.4 s.

- **Third backend**: the ggml **Metal** path runs the whole pipeline on Apple
  silicon. Selftest passes on NPR-4B (TVs ≤ 0.0003, control 0.5571 vs the H200's
  0.55) and AIME 2025 I/1 solves correctly — but at 17.7 tok/s serial with poor
  batching, so it is for correctness work, not sweeps (DESIGN.md §12).

Eleven engine-level bugs were found on the way; **seven are fixed on this branch**
(the commits below), four are documented-but-unfixed (§7).

## 4. Commits on the branch (16)

```
6125f7a  fix(inferlets/npr): download the CMake tarball to a file, with retries
54425a3  fix(inferlets/npr): pod bootstrap installs CMake >= 3.23 and finds nvcc
f46f8f9  chore(inferlets/npr): pod bootstrap does the bf16 cast and installs eval deps
60c2a16  feat(inferlets/npr): avg@8 AIME25 eval harness + Metal bring-up
b5380d3  fix(driver/portable): size the sampling tail by slot count, not request count
124cb85  docs(inferlets/npr): restore CPU validation detail, correct migration note
d759a07  docs(inferlets/npr): handover — design study, pod artifacts, run transcripts
e7a45dc  fix(inferlets/npr): parse plans without relying on the rendered open tag
8de3f5e  fix(driver/cuda): skip logits tail for prefill-only batches
7b883f2  feat(inferlets/npr): RunPod bootstrap script + standalone JSON-WS client
2833245  feat(inferlets/npr): faithful refill join (phase 2) + numeric selftest
facebc9  feat(sdk): decoupled-position passes — Forward::positions, Generator::position_offset
86ed682  fix(runtime): allow non-monotonic committed positions for explicit-mask tokens
e10b9ac  fix(driver/portable): KV write index from slot order, not position id
6c1e1fe  feat(inferlets): npr — Native Parallel Reasoner control loop (phase 1)
c0218d2  fix(sdk): prevent trap in Context::destroy from double resource release
(+ this handover commit)
```

Five of these are **upstreamable pie bug fixes independent of NPR** — worth
separate PRs to `pie-project/pie`: `c0218d2` (SDK destroy trap), `e10b9ac`
(portable KV write index — a real data-corruption bug), `86ed682` (runtime commit
check), `8de3f5e` (CUDA prefill OOB — crashes any prompt longer than the request
cap, so it affects far more than NPR), and `b5380d3` (portable sampling reshape —
aborts the whole server process on any batch that mixes a prefill with decodes,
so it hits any concurrent workload, not just NPR).

### 4.1 Upstreaming those five (state as of 2026-08-20)

**Prepared, not yet opened.** Branches are cut, committed, pushed and verified;
opening the PRs is blocked on a credential, see "Blocker" below. The full PR
bodies, the exact titles, an `open-prs.sh`, and the analysis behind all of this
live in **`inferlets/npr/upstream-prs/`** — read that directory before redoing
any of this work.

**They target upstream `main`, NOT `dev`.** This is the finding that would
otherwise cost a round of confused review:

- `94043eb12`, the base of `npr-inferlet`, is on `fork/dev`
  (`YangLiuWillow/pie`) and on **no** `pie-project/pie` branch at all.
- Upstream `dev` is a **different codebase generation** — a rewrite carrying
  `runtime/engine/`, `compiler/`, `controller/`, `model/`, `worker/`, and
  **no `driver/portable`**, no `sdk/rust/inferlet/src/context.rs`. Four of the
  five fixes have no file to apply to there.
- Every file these fixes touch exists on upstream **`main`**, which is also the
  repo's default branch; `.github/workflows/ci.yml` runs only on PRs into
  `main` (made explicit upstream in #516).
- Merge-base of `fork/dev` with upstream is `8824d3e5b`, on both `main` and
  `dev`.

**`b5380d3` is half obsolete — do not cherry-pick it whole.** Upstream already
merged its `ggml_reshape_3d` half as **PR #426** ("key uniform-sample reshape on
slot count, not request count"); all five graph builders on `main` now derive
`n_slots` from `probs->ne[1]`. What is still live upstream is the *other* half
of the same defect: `GraphCache::matches()` does not compare the sampling-slot
count, so a cached graph can be reused at a count its `out_idx` — sized to
exactly `plan.sampling_pos_i32.size()` — was never built for, and
`upload_graph_inputs` then overruns it (more slots) or leaves stale entries
(fewer). That half was rebased on its own; the branch is named for what it
actually does. It fails **silently** (wrong tokens) where #426's half failed
loudly (`GGML_ASSERT`).

**`c0218d2` overlaps an upstream fix that never reached `main`.** PR **#484**
fixes the same defect host-side (drop `table.delete(this)?` from
`HostContext::destroy`, let the ordinary resource drop own deletion). It merged
into `tts-arena/main`, a branch that no longer exists; `main`'s `destroy` still
calls `table.delete(this)?`. The two are **alternatives** — applying both leaves
nothing to delete the table entry, leaking a slot per destroyed context. The PR
body says this and names #484.

| branch on `fork` | from | PR | applies to `upstream/main` |
|---|---|---|---|
| `fix/sdk-context-destroy-trap` | `c0218d2` | not yet opened | clean |
| `fix/portable-kv-write-index` | `e10b9ac` | not yet opened | clean |
| `fix/runtime-explicit-mask-commit` | `86ed682` | not yet opened | clean |
| `fix/cuda-prefill-logits-oob` | `8de3f5e` | not yet opened | clean |
| `fix/portable-graph-cache-slot-count` | `b5380d3` | not yet opened | **rebased, see above** |

All five: one focused commit, authored as Liu, no AI co-author trailers, base
`main`. Each was confirmed to fix code that is **still live on current `main`**
by reading `main`, not merely by getting a clean cherry-pick.

**Blocker.** `gh` on this host is authenticated with a **fine-grained** PAT
(`gh api -i user` returns no `X-OAuth-Scopes` header). Fine-grained PATs only
reach repos owned by the token owner or granted by an org, so on
`pie-project/pie` it has public-read only — `permissions: {pull: true, push:
false, ...}`. Creating a PR fails with `403 Resource not accessible by personal
access token` on **both** `gh pr create` (GraphQL) and
`gh api -X POST repos/pie-project/pie/pulls` (REST). The endpoint choice is not
the issue; the token type is. To unblock, either re-auth with `gh auth login`
(web flow) or use a **classic** PAT with `public_repo` scope — then run
`inferlets/npr/upstream-prs/open-prs.sh`, one call at a time, checking the
returned number before firing the next. Alternatively open them from the GitHub
UI; the compare URLs are in that directory's README.

**Unrelated, found while verifying:** `cargo test --workspace` is **red out of
the box on macOS** on unmodified upstream `main` — 7 `pie-bridge` shmem tests
fail because macOS caps POSIX shm names at 31 characters, so `shm_open` returns
`ENAMETOOLONG`. Reproduced on a clean detached `upstream/main` checkout, so it
is not fallout from any of these fixes. Noted in PR 3's verification section;
deliberately not opened as a sixth PR.

## 5. How to run it (from zero, on a GPU pod)

```bash
# 1. Provision a CUDA pod: Ampere+ (SM 8.0), >=24GB VRAM (H200/H100/A40/4090 all
#    fine), a runpod/pytorch *-devel image, >=60GB disk, PUBLIC_KEY env = your
#    ssh pubkey, port 22 exposed. RunPod REST API: https://rest.runpod.io/v1/pods
#    with `Authorization: Bearer $RUNPOD_API_KEY`; the *proxy* ssh user is
#    `<podId>-<hostSuffix>` from GraphQL `machine { podHostId }`, but plain
#    `ssh -p <port> root@<publicIp>` is simpler and worked.

# 2. Bootstrap (deps, rust+wasm target, clone, build engine+inferlet, venv,
#    download NPR-4B, write config). ~20-25 min, mostly the CUDA engine build.
curl -fsSL https://raw.githubusercontent.com/YangLiuWillow/pie/npr-inferlet/inferlets/npr/pod-setup.sh -o pod-setup.sh
bash pod-setup.sh      # run under nohup/setsid if over ssh; it survives disconnects

# 3. Serve + run. (pod-setup.sh now does the fp32->bf16 cast itself and points
#    the config at the result; that used to be a manual step.)
cd /workspace/pie
PIE_CONFIG=/workspace/npr-cuda.toml nohup ./target/release/pie serve > /workspace/serve.log 2>&1 &
/workspace/venv/bin/python inferlets/npr/client.py --input '{"selftest": true}'
/workspace/venv/bin/python inferlets/npr/client.py --input '{"max_new_tokens": 30000, "question": "..."}'

# 4. The sweep (see inferlets/npr/evals/README.md).
/workspace/venv/bin/python inferlets/npr/evals/run_eval.py --k 8 --concurrency 32
```

**Image note**: a plain `nvidia/cuda:12.8.1-devel-ubuntu22.04` works and is what
the eval pod used; `pod-setup.sh` installs a current CMake (Ubuntu 22.04's 3.22
is below driver/cuda's 3.23 minimum) and puts `/usr/local/cuda/bin` on PATH.
Install `curl` and `git` first — that image has neither.

Local CPU/Metal smoke tests need no GPU: use `configs/npr-portable.toml` (real
tokens, ~1 tok/s) or `configs/npr-dummy.toml` (random tokens, exercises control
flow only, ~100 ms). On macOS the binary needs an extra link flag — see §7.

**Inferlet inputs** (`Pie.toml` documents them): `question`, `max_new_tokens`
(default 30000), `join_mode` (`"refill"` default | `"textual"` A/B baseline),
`max_plans` (5), `max_depth` (5), `min_fork_budget` (1024), `temperature` (1.0),
`top_p` (0.7), plus test hooks `selftest`, `primer`, `max_step_tokens`.

## 6. The one idea you must understand before editing

**Position IDs are decoupled from KV slot order.** NPR's parallelism means
sibling branches occupy *different KV slots* but *overlapping RoPE positions*
(every sibling restarts at the position right after `</guideline>`; the takeaway
resumes at `p_fork + max(branch extents)`).

Almost every bug in this port was some layer assuming `position == slot`:

- the portable driver used position as the KV **write index** (corrupting live KV),
- it clamped custom attention-mask rows at `position` (hiding a token from itself),
- the runtime rejected commits whose positions weren't monotonically increasing,
- the runtime's *synthesized* causal mask is literally `all_true(position + 1)`.

That last one is not a bug but a live constraint: **after a join, every
multi-token prefill into that context must pass explicit BRLE mask rows**
(single-token decode is safe — it attends to all KV via page metadata). The
inferlet maintains this with `PCtx { ctx, delta }`, invariant
`position(slot) = slot + delta`, and `Generator::position_offset(delta)`.

If you touch the CUDA driver or add a new backend, audit it for the same
assumption. (`write_kv_kernel` in the CUDA driver was already slot-correct.)

## 7. Known gotchas / unfixed upstream issues

1. **Host forward failures surface to the guest as empty *success***
   (`runtime/src/api/inference.rs::FutureOutput::ready`). You get a confusing
   downstream error like `commit: need 848 tokens, have 0`. **Always
   `grep -a "future output failed\|pie-driver" serve.log`** — the real error is
   there. Unfixed (needs a WIT surface for late errors).
2. **macOS `pie-bin` link failure** — `worker/build.rs` link-args don't propagate
   to the binary. Build with:
   `cargo rustc --release -p pie-bin --bin pie --no-default-features --features driver-portable -- -C link-arg=-framework -C link-arg=Accelerate`
3. **Stale POSIX shared memory** after killing a server:
   `shm_open("/pie_shmem_g0"): Permission denied` on next boot. Unlink it (see
   `configs/npr-dummy.toml` header for the one-liner).
4. **Python `pie_client` is protocol-drifted** — it speaks msgpack, the dev
   gateway parses JSON only, and chunked uploads hang under the turn-based WS
   model. Use `client.py` (JSON frames, single-chunk upload,
   `x-pie-identity` trust-edge header, `/v1/ws` path). Don't "fix" it by going
   back to `pie_client`.
5. **Pass-level speculation is disabled** throughout the inferlet — stale staged
   run-ahead passes for destroyed branch contexts raced the join's refills and
   crashed the driver once. If you re-enable it, expect that race. *Caveat*: the
   `build_qwen3_graph` crash that motivated this turned out to have a second,
   fork-independent cause — see DESIGN.md §12, bug 11 — so the speculation race
   may never have been the culprit.
6. **`hf_repo` must be a resolved local snapshot path**, and the config uses the
   combined `[gateway]` / `[worker.*]` section layout (a flat `[[model]]` is
   silently not found).
7. **Temperature 1.0 → high variance.** Single runs prove plumbing, not quality.

## 8. Suggested next steps

1. **The eval**: `inferlets/npr/evals/` now holds the harness — AIME 2025 (30
   problems), a concurrent sweep driver, and a scorer for avg@k / pass@k /
   throughput, across `refill` vs `textual` vs a `sequential` baseline
   (`max_plans=0`). See `evals/README.md`; run it per §5 on a CUDA pod.
   Accuracy runs at high concurrency, the speed pass must run at concurrency 1.
   This is the headline result: *does the user-space inferlet reproduce NPR
   Engine quality and speed?*
2. **Verify phase 3 on CUDA** (done on Metal): the row-copy path
   (`SwapPool::copy_rows_d2d`) compiles but has never executed on a GPU;
   run the selftest (adopt + adopt_control arms) and an
   `--arms adopt,refill` A/B on the pod. The sweep now has an `adopt` arm;
   check `adopt_fallbacks == 0` in results.
3. **Upstream the four independent pie fixes** as PRs (§4).
4. **Nested (depth ≥ 2) refill joins** — currently refill mode forks at depth 1
   only; needs per-token position/visibility records through `run_branch`.
5. **Per-`<step>` repetition penalty 1.02** (NPR has it; pie has no such sampler
   — probe a top-k distribution and sample guest-side).

## 9. Working notes for the assistant

- The user is an LLM-infra/post-training expert and an NPR co-author — pitch at
  that level, skip basics, be precise about mechanism.
- Verify claims in code rather than trusting the paper's prose; several paper
  abstractions (e.g. "attention mask trick") turned out to be training-only,
  while the real inference mechanism was KV stitching + position alignment.
- When something fails, read the *server* log before theorizing (see §7.1).
- **Cost discipline**: GPU pods bill by the hour on a shared RunPod account that
  has run low before ($25 left at $11/hr across four pods). Terminate pods when
  experiments wrap; check for other sessions' idle pods and flag them rather than
  killing them.

## 10. Data-loss postmortem (2026-08-17)

The final A6000 sweep (4 arms × 25 problems × k=2, fixed per-request budget,
`aime25-a6000-final.jsonl`) reached 190/200 before the pod was stopped. Both
sweep pods had been deployed with `volumeInGb: 0`, so `/workspace` lived on the
ephemeral container disk, which RunPod wipes on stop — the results were
unrecoverable the moment the pod first stopped (confirmed by restarting it:
`/workspace` was empty). The only surviving numbers from that sweep are the
interim adopt-arm scoring taken on-pod at ~190/200: **avg 0.545, pass@2 0.591,
21% unanswered** (25 problems, k=2), vs the paper's 50.4 avg@8. The local
`results/aime25-a40.jsonl` predates the per-request ×degree ledger fix (adopt
avg 0.204, 78% unanswered — the token-starvation bug) and must not be quoted as
a post-fix result. Rule for any future sweep: deploy with a persistent volume
**and** pull results off-pod continuously (a local-side scp loop every few
minutes); treat pod disks as lossable at any moment. Both pods were terminated.

## 11. Final four-arm sweep (2026-08-17, L40S, `results/aime25-rerun.jsonl`)

AIME 2025, 25 problems × k=2 per arm, 30k per-request budget, concurrency 8,
prompt cache on, zero engine errors in 200 runs (one client-side timeout,
counted as unanswered). Paper reference: 50.4 avg@8.

| arm        | avg   | pass@2 | unanswered | mean gen | join mean |
|------------|-------|--------|------------|----------|-----------|
| adopt      | 0.460 | 0.520  | 20%        | 19.0k    | 1.17 s    |
| refill     | 0.460 | 0.520  | 26%        | 18.5k    | 2.52 s    |
| textual    | 0.360 | 0.440  | 36%        | 18.7k    | —         |
| sequential | 0.440 | 0.560  | 2%         | 8.4k     | —         |

Reads: (1) adopt ≡ refill exactly on avg and pass@2, with the graft join 2.2×
faster and 674k tokens grafted at 0 fallbacks — the KV-graft join is quality-
neutral and strictly faster. (2) Both are within noise of the paper (±0.14 at
n=50). (3) textual trails by 10 points — KV surgery matters. (4) At equal wall
clock (~580 s) the parallel arms generate 2.3× the tokens of sequential.
(5) The residual weakness is the 20–26% unanswered rate on parallel arms vs 2%
sequential — per-`<step>` repetition penalty 1.02 is the remaining
unimplemented NPR knob and the prime suspect. Pod hygiene for reruns: deploy
with a volume, gate hosts on `cuInit(0)==0` (3 of 4 community L40S hosts had
broken CUDA), stream results off-pod continuously.

### What these 50-run arms can and cannot resolve

The `adopt` arm's accuracy is a mixture, and the mixture is exact:
`share(b>=2) x acc|b>=2 + share(b<=1) x acc|b<=1 = 0.540 x 0.815 + 0.460 x
0.043 = 0.460`. Accuracy here is almost entirely *how often a run escapes its
first parallel block*, and barely at all how well it reasons once there. So
reaching the paper's 0.504 with both conditionals held needs `share(b>=2)` to go
0.540 -> 0.597 — **+2.2 correct runs out of 50**, against a Wilson half-width of
0.133. **No 25 x k=2 sweep can answer a 0.044 question through the mean.**

Pricing a sweep that could, because the obvious response is "run more":

| analysis | runs per arm for 80% power at 0.044 |
|---|---|
| unpaired, two marginal proportions | 2,021 |
| unpaired but clustered (ICC 0.767, k=2) | ~3,570 |
| **paired on per-problem differences** | **487** |
| paired, if the effect clips near p=1 | 838 |

The unpaired figure is the right arithmetic for the wrong design: both arms run
the same problems, and AIME difficulty is near-bimodal (between-problem variance
of latent `p_i` = 0.188, problem-level ICC = **0.767**), so a paired test fights
the within-problem variance `E[p(1-p)] = 0.060`, not the `p(1-p) = 0.248` the
unpaired formula charges. **Two conditions travel with that recommendation and
are part of what would be authorized, not downstream analysis choices:**

1. **It must be analyzed paired**, on per-problem differences. Buying 500
   runs/arm and then comparing two marginal Wilson intervals will produce
   overlapping intervals and a spurious null — paying for the data and
   discarding the design that made it affordable.
2. **Ignoring the pairing costs more than the naive figure suggests**, not
   less: analyzed as unpaired-but-clustered, ICC 0.767 inflates the k=2 design
   to ~3,570 runs/arm.

And it is an overnight run, not a piggyback: ~974 rows at this sweep's ~100
rows/90 min is **~15 hours** of pod time and 15 hours of exposure to pod death.
The resume story is real but manual — `run_eval.py --resume` (default) skips
keys already in the `--out` file, and the poller streams that file off-pod every
3 minutes, so a pod lost at hour 9 costs at most one poll interval *provided the
streamed copy is pushed back to the replacement pod's `$STATE/results/` before
relaunching*. Errored rows are deliberately not counted as seen, so they retry.

**The mean is purchasable for ~$15-25; it is not unreachable.** But it can only
say *whether* the penalty worked, never *which* of the two failure populations
moved — and that is what picks the next fix. The mechanism route is the better
buy on information per dollar, not the consolation prize for an underpowered
sweep.

## 11b. Repetition-penalty A/B (2026-08-21, L40S, `results/aime25-pen-ab.jsonl`)

AIME 2025, 25 problems x k=2 per arm, 30k budget, concurrency 8, prompt cache on,
100/100 rows, **zero engine errors**. Build `cf204689a` (`lib.rs` blob-identical
to `a35a68ac9`). Full writeup + method: `evals/results/aime25-pen-ab.md`.

| arm | avg@2 | pass@2 | unans | bud | share b>=2 | acc\|b<=1 | acc\|b>=2 | gen tok |
|---|---|---|---|---|---|---|---|---|
| `adopt` (penalty 1.02) | 0.520 | 0.560 | 0.220 | 0.420 | 0.540 | 0.217 | 0.778 | 18.5k |
| `adopt_nopen` (penalty 1.0) | 0.480 | 0.520 | 0.200 | 0.420 | 0.540 | 0.261 | 0.667 | 19.4k |

**The penalty is not the lever.** `adopt_nopen` replicates §11's 0.460 at 0.480,
so the control is sound. Against it the penalty leaves `share(b>=2)` at
**exactly 27/50 in both arms** and budget exhaustion at **exactly 21/50 in both
arms** — identical counts, not a noisy null, on precisely the metric DESIGN.md
§15 says the penalty exists to move. It also did not shorten stranded branches
(25,926 vs 25,378 mean generated tokens in the `blocks<=1` stratum — slightly
*more*) and did not cut the unanswered rate (0.220 vs 0.200).

The +4.0 pp on avg is **3 runs**, all inside the `b>=2` stratum (21/27 vs
18/27), with Wilson intervals overlapping across most of their range.
**Always write it as "0.520 against a same-sweep control of 0.480" — one
sentence, both halves.** A caveat on an adjacent line will eventually be dropped,
and 0.520 alone reads as the penalty clearing the paper's 0.504, the inverse of
the finding. Correct statement: quality-neutral within resolution,
mechanistically inert. Pre-registered before the rows landed (commit
`bd311c642`, at 11/100).

**This is answered, not underpowered.** An underpowered result is one more data
would resolve. Three independent channels came back flat — `share(b>=2)`
identical at 27/50, budget exhaustion identical at 21/50, and the penalty
failing its own stated mechanism. A powered mean sharpens an estimate; it cannot
make a flat channel non-flat.

**Correction to §11's reading, from `pr-split`'s `stop_reason` split.** I had
inferred from charge-to-budget ratios that the `blocks<=1` population was ~14
starved and ~9 killed by early branch EOS. The labels refute it: of 46
`blocks<=1` runs, **42 are `branch_budget`, 3 `eos`, 1 `branch_terminal`**. The
proxy failed because the ledger's primary check is **positional**
(`position_exhausted`), so a run exhausts budget while `tokens_charged` sits
well below it — **15 of the 42 had charge ratios below 0.90, as low as 0.517**.
The collapse is one phenomenon, budget exhaustion, not two.

**Next experiment — one pod, two questions (proposed 2026-08-21, not started).**
42 of 46 stranded runs died of budget/positional exhaustion and no sampler
touches that, so §14's second suspect is now the leading explanation for the
residual gap to 0.504.

*Primary — the budget question.* NPR charges each branch token x its parallel
degree against `max_new_tokens` (`schedule_batch.py:693-697`, replicated in the
inferlet's ledger). At the observed mean of ~4.3 branches, a 30,000 *charged*
budget leaves only ~7k *generated* tokens along the critical path, so if the
paper's 30,000 is effectively **per-sequence** in their eval path rather than
ledger-charged, their effective budget is **~3.5x ours** — which would explain
the whole residual gap without any quality difference. Test: re-run one
`adopt_nopen` arm at a per-sequence-equivalent budget and see whether the
`blocks<=1` population survives. **One arm, ~1/4 the cost of a powered mean, and
unlike the mean it can change the answer rather than measure it more precisely.**

*Rider — `pr-split`'s bug-17 §7 instrumentation.* One patched build, one run,
a pre-registered discriminating outcome either way; see `bug17/RUNBOOK.md`.

**Both need a CUDA pod, so they should be one pod, not two.** That is cheaper
than either alone and removes any reason to hold a pod open at the end of a
sweep. `pod/resume.sh` makes a mid-run pod death cost ~45 min rather than the
run, and `pod/gate.sh` plus the `runpod/pytorch` image default keep the hunt from
repeating the sshd/MooseFS triage. A powered paired mean (§11 above, ~24 h,
~$15-25) is **not** the recommended spend: the mechanism question is already
answered, and the budget question is both cheaper and live.

## 12. Frontier for a cold agent (2026-08-20): validate repetition penalty 1.02

Commit `e14082eb0` implemented NPR's last knob — per-`<step>` repetition
penalty 1.02, engine-faithful (DESIGN.md §15). **Both halves of this section are
now done — see §11b: CUDA-validated (selftest `ids_match=true max_dev=0.000000
max_logit=21.875`) and A/B'd (quality-neutral, mechanistically inert).** Original
text follows. Off until the first fork, then
on for every `<step>` child and the trunk after merge; HF semantics on **raw
logits** (`l<0 → l·p`, `l>0 → l/p`), output tokens only. Raw logits come from a
new `TopLogits { k }` probe (`sdk/rust/inferlet/src/sample.rs`) = the `dist`
slot with sentinel `temperature == 0`, implemented in both drivers
(`driver/portable/src/sampler.cpp`, `driver/cuda/src/response_subpass.cpp`).
It is validated on Metal (selftest `ids_match=true max_dev=0.000000
max_logit≈22`; AIME I/1 correct with 2 blocks). **Not yet done: CUDA
validation and the quality A/B.** Nothing is running and no pods exist.

### The run

Pipeline scripts are durable in `inferlets/npr/pod/` (README there). Exact
sequence:

```bash
cd inferlets/npr/pod
export NPR_STATE=$HOME/.npr-pod
bash hunt.sh                                   # cuInit + network gated, volume-backed pod
bash launch.sh adopt,adopt_nopen aime25-pen-ab.jsonl   # == run_eval.py --arms adopt,adopt_nopen --k 2 --limit 25 --concurrency 8 --prompt-cache --abort-after 10 --out results/aime25-pen-ab.jsonl
EXPECT_ROWS=100 python3 -c 'import subprocess,os; subprocess.Popen(["bash","poll.sh"],stdout=open(os.environ["NPR_STATE"]+"/poll.log","a"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)'
tail -F $NPR_STATE/poll.log                    # ends with "SWEEP COMPLETE ... pod terminated"
```

`adopt` = KV-graft join with penalty 1.02 (default); `adopt_nopen` = same with
`rep_penalty: 1.0` (the arm that produced §11's 0.460). 2 arms × 25 × k=2 =
100 rows, ~1.5 h, ~$1.5 on an L40S. If the pod must clone the private repo,
set `REPO=`/`REPO_RAW=` (chain.sh / pod-setup.sh honour them) — the defaults
point at the public fork `YangLiuWillow/pie`.

### Check 1 — CUDA sentinel (before trusting any number)

`$NPR_STATE/selftest.log` (pulled by the poller) must contain

```
[npr] selftest toplogits: ids_match=true max_dev=0.00xxxx max_logit=2x.x
```

`ids_match` compares the `TopLogits{k:10}` ids against a `Distribution`
probe on the same forward pass; `max_dev` is the max |softmax(logits) − prob|.
On Metal it was exactly 0.000000; on CUDA expect ≤1e-3 (bf16→f32 widening on
the host vs the softmax kernel). `ids_match=false` or `max_logit` ≈ 0–1 means
the CUDA sentinel path is returning probabilities, not logits — stop and fix
`compute_dist_slots` before running the sweep. The full selftest must still
end with its normal pass line.

### Check 2 — per-arm scoring

```bash
cd inferlets/npr/evals
/Users/liuyang/Documents/Liszt_ai/npr-eval-venv/bin/python score.py results/aime25-pen-ab.jsonl               # avg@2 / pass@2 / unanswered / tokens per arm
/Users/liuyang/Documents/Liszt_ai/npr-eval-venv/bin/python score.py results/aime25-pen-ab.jsonl --by-problem
```

and the collapse metric, which is the real question (§11 read 5; analysis in
the session: correct runs are short and have `parallel_blocks=2`, every
unanswered run had `parallel_blocks=1` and `stop_reason=branch_terminal`):

```bash
/Users/liuyang/Documents/Liszt_ai/npr-eval-venv/bin/python - <<'PY'
import json,collections,sys; sys.path.insert(0,"."); from score import equal
rows=[json.loads(l) for l in open("results/aime25-pen-ab.jsonl")]
by=collections.defaultdict(list)
for r in rows: by[r["arm"]].append(r)
ok=lambda r: equal(r.get("answer"), r["gold"])
un=lambda xs: sum(1 for r in xs if r.get("answer") in (None,""))/max(1,len(xs))
acc=lambda xs: sum(map(ok,xs))/max(1,len(xs))
for arm,rs in by.items():
    b1=[r for r in rs if (r.get("parallel_blocks") or 0)<=1]; b2=[r for r in rs if (r.get("parallel_blocks") or 0)>=2]
    print(f"{arm:12s} n={len(rs)} avg={acc(rs):.3f} unanswered={un(rs):.2f} | blocks<=1: n={len(b1)} acc={acc(b1):.2f} | blocks>=2: n={len(b2)} acc={acc(b2):.2f}")
PY
```

(Row fields: `arm`, `gold`, `answer`, `parallel_blocks`, `stop_reason`,
`tokens_generated`, …; `run_eval.py` is the source of truth. Use the eval venv —
system python3 is too old for `score.py`'s type hints.) Baseline to
beat (§11, no penalty): avg 0.460, unanswered 20%, blocks≤1 ≈5% correct vs
blocks≥2 ≈80%. Success = penalty raises avg toward the paper's 0.504 by
cutting the unanswered / blocks=1 share; neutral-or-worse = the penalty is not
the lever and the next suspect is the `branch_terminal` budget-exhaustion path
(§8). Either way: record the table in this file, update the artifact, and
commit; `results/` stays gitignored (results live only on the local worktree +
`$NPR_STATE`).

### After the run

- Pod terminated? `source pod/rp.sh; curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"` — only terminate pods *you* created.
- Open items unchanged: bug 16 deadlock (§7), split budget-exhaustion out of
  `branch_terminal`, concurrency-1 speed pass, upstream PRs (bug fixes,
  `adopt_kv`, `TopLogits`).
