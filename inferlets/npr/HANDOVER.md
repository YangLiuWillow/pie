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

**Lost in the device migration** (recreate if needed): the local
`~/Desktop/Lin_startup/{pie,pie-npr,pie-rl,Native-Parallel-Reasoner}` checkouts,
and the RunPod API key that lived in `pie-rl/.env` as `RUNPOD_API_KEY=...`.
Nothing else was device-local — the design doc used to be in pie's *gitignored*
`/docs/`, which is why it now lives in `inferlets/npr/DESIGN.md` instead.

## 3. Status: what works

Phases 1 (control loop) and 2 (faithful refill join) are **done and validated on
real GPU hardware**.

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

Ten engine-level bugs were found on the way; **six are fixed on this branch**
(the commits below), four are documented-but-unfixed (§7).

## 4. Commits on the branch (12)

```
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

Four of these are **upstreamable pie bug fixes independent of NPR** — worth
separate PRs to `pie-project/pie`: `c0218d2` (SDK destroy trap), `e10b9ac`
(portable KV write index — a real data-corruption bug), `86ed682` (runtime commit
check), `8de3f5e` (CUDA prefill OOB — crashes any prompt longer than the request
cap, so it affects far more than NPR).

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

# 3. Convert the checkpoint to bf16 (REQUIRED — the published NPR-4B is fp32 and
#    pie's loader does not cast; symptom otherwise:
#    "gemm_act_x_w: unsupported dtype combo (act=bf16, w=fp32, y=bf16)").
python3 /workspace/pie/inferlets/npr/convert_bf16.py    # edit paths at the top
sed -i 's|hf_repo = .*|hf_repo = "/workspace/NPR-4B-bf16"|' /workspace/npr-cuda.toml

# 4. Serve + run.
cd /workspace/pie
PIE_CONFIG=/workspace/npr-cuda.toml nohup ./target/release/pie serve > /workspace/serve.log 2>&1 &
/workspace/venv/bin/python inferlets/npr/client.py --input '{"selftest": true}'
/workspace/venv/bin/python inferlets/npr/client.py --input '{"max_new_tokens": 30000, "question": "..."}'
```

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
   crashed the driver once. If you re-enable it, expect that race.
6. **`hf_repo` must be a resolved local snapshot path**, and the config uses the
   combined `[gateway]` / `[worker.*]` section layout (a flat `[[model]]` is
   silently not found).
7. **Temperature 1.0 → high variance.** Single runs prove plumbing, not quality.

## 8. Suggested next steps

1. **The eval that's actually missing**: avg@8 on AIME25 (and HMMT25/AMC23)
   comparing `join_mode=refill` vs `join_mode=textual` vs a sequential baseline,
   plus tokens/sec, against the paper's Table 2/3 numbers. The NPR repo's
   `evals/evaluate.py` has the scoring harness (`math_equal`, `extract_answer`,
   pass@k) — driving it against pie means replacing the SGLang engine calls with
   `client.py` launches. This is the headline result: *does the user-space
   inferlet reproduce NPR Engine quality and speed?*
2. **Phase 3 — `context.adopt_pages`**: the O(1) KV page-graft op (DESIGN.md §4)
   that replaces refill recomputation and closes the last efficiency gap.
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
