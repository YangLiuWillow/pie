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
