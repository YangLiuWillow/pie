# Overcommit fix — live plan and state (2026-07-30, ~22:00 UTC)

Written mid-investigation so an SSH drop or context loss costs nothing.
Companion to `DEFECTS_OVERCOMMIT.md` (the defect record) and
`AGENT_HANDOVER_20260729.md` (the prior session). Repo state: everything
through `4fb3d367` is pushed to `origin/openhands-integration-updated`;
the park-after-generation inferlet fix (described below) is built and
installed on the pod but NOT yet committed — commit it once the repro
validates it.

## What the repro is and what it must show

`50_run_repro.sh` / `50_overcommit_repro.py`: 8 concurrent live-context
daemon sessions, UNIQUE ~28k-token histories (unique because the KV trie
is content-addressed — identical fillers dedupe and never overcommit),
124k-token pool + swap_pool=4096, 12 rounds of concurrent turns,
120 s/call wedge threshold. Exit 0 = clean, 2 = wedge (livelock), 1 =
inferlet error (so far: KV_INVARIANT_VIOLATION).

Run instrumented: `PIE_SCHED_DEBUG=1 PIE_SCHED_DEBUG_SECS=5 bash
50_run_repro.sh` — dumps scheduler state (active/pinned/stashed/suspended,
alloc-queue head need-vs-free) every 5 s to the server log
(`logs/repro_pie_serve_<TS>.log`).

**A run is IN FLIGHT right now** (server log timestamp 220004), the first
with the park-after-generation fix. Interpret its outcome:

- **Completes all rounds** → the parked-window was feeding both defects;
  livelock may need the harsher trigger (idle-suspend: add
  `--idle-suspend`) or higher pressure (`--sessions 10`). Try those before
  declaring victory.
- **Exit 2 (wedge)** → defect 1 reproduced CLEANLY. Read the last DumpSched
  lines in the server log: head `need > free` persistently = eviction loop
  never victimizes idle min-bid contexts for the starved requester;
  `need <= free` = stale queue / missing DrainKick. Fix accordingly (below).
- **Exit 1 (KV_INVARIANT again)** → the theft window is wider than
  fork-time; re-read the theory in §"Defect 2 diagnosis" — check whether
  `flush()` of the NEXT request's append can also share working pages, or
  whether the parent's tail page is referenced by the child even after the
  parent was re-parked by a LATER round while a prior child still lives.

## The diagnosis chain (how we got here)

1. c8 benchmark arms all failed 8/8 with 900 s timeouts regardless of
   inferlet policy: A (snapshots+swap), B (live + wake-bid 1.0),
   C (live + idle-suspend). Policy is not the problem.
2. Repro run 1 (exit 2): session 8's FRESH context starved while 7 idle
   parents sat resident at bid 1e-12 — **defect 1: alloc-path
   livelock** (eviction loop never evicts them, or queue never drains).
3. Repro runs 2–3 (exit 1): `KV_INVARIANT_VIOLATION total_kv =
   page_capacity + 1` — **defect 2**: `suspend()` FREES working pages
   without refcounting (`sched.rs` suspend Phase 1), and a fork's child
   references the parent's partial tail WORKING page (prompts aren't
   page-aligned; partial pages can't commit). Evicting a parked parent
   mid-generation steals the generating child's tail page.
4. Inferlet-side fix for defect 2's trigger (BUILT + INSTALLED, uncommitted):
   in `handle_request`, the parent is now held at active priority during
   generation and parked (idle bid / optional suspend) only AFTER the
   child's generation completes. See the two "CRITICAL ORDER" /
   "NOW it is safe" comments in `inferlets/openhands-coder-session/src/lib.rs`.
   The ENGINE bug remains: suspend must not free working pages a live fork
   references (refcount working pages, or copy-on-fork the tail page).

## The fix plan for defect 1 (task: alloc/eviction path)

Priority suspects in `runtime/src/context/sched.rs`:

- The alloc path branches on `requester_off_gpu` → `alloc_queue.push_back`
  ("wait for capacity **without restoring**") — a FRESH context may take a
  wait-only branch instead of the eviction loop (Step 4), so fresh fills
  can never evict anyone. Verify with the DumpSched head data.
- `DrainKick` exists as a message; check every page-free site actually
  sends it — a free with no kick strands alloc_queue/restore_queue waiters.
- If admission-starvation appears instead (need > free with churn):
  implement reserve-toward-head — freed pages accrue to the highest-bid
  waiter instead of returning to the general pool.

Validation ladder after any fix: repro (both policies, exit 0) → c1 smoke
(1 instance, live mode, `ab_h100_pie_overcommit_p32_c1` pattern; expect
93.8%-style reuse, 0 errors) → c8 arm (`armB2_driver.sh` pattern). Then
update fig4 + the writeup `[PENDING]` section, commit everything.

## Machine state

H100 pod, rebuilt 2026-07-30 (~18:40) by `00_setup_h200.sh` after /root
wipe — gate check PASS. Installed inferlet
`/root/.pie/programs/programs/openhands-coder-session/0.1.0.wasm` =
target build md5 `3829b3f50bf059efcc5a1272f3540394` (park-after-generation
fix). GPU must be free before any run (`nvidia-smi`; kill by PID, never
`pkill -f`). Chunk-sweep results and all benchmark rows are in
`predictions/` and summarized in the writeup draft
(`../docs/blog-pie-agent-serving.md`). GitHub token for pushes: ask the
human (earlier PAT is in this session's history only; a second token lives
in `/workspace/.env`).

## Open items beyond defect 1

- Engine fix for defect 2 (working-page refcount / copy-on-fork).
- Commit the park-after-generation inferlet change once repro-validated.
- Re-run c8 arms after fixes; fill writeup fig4 + `[PENDING]`.
- Decode-slope +10% at blk64 (borderline) wants one repeat before the
  writeup quotes 24,120 tok/s as the recommended config.
- Stretch: chat-apc HTTP surface (in `/workspace/pie-vllm-test`) as the
  transport for multi-harness (Codex) arms.

## RESOLVED (2026-07-30 ~23:55) — see DEFECTS_OVERCOMMIT.md §RESOLUTION

Repro 12/12 rounds; c1 smoke 1.28 s/iter (~6% over fair c1); c8 live arm
served every attempted instance (5 healthy + 1 empty-patch; stopped early
for the comparator); vLLM fair c8 on the same 8 instances: 8/8, and on
cleanly-matched instances Pie s/iter is 1.00-1.08x of vLLM — parity at
~2x overcommit with the STOCK 512 chunk. Next: the lever arm
(PIE_CUDA_PREFILL_TOKENS=2048 + blk64) and the CUTLASS grouped-GEMM probe.

## NEXT: CUTLASS grouped-GEMM dispatch probe (task #6, designed 2026-07-31 ~00:00)

Vendored tree: /root/.cpm-cache/flashinfer/b239/csrc/nv_internal/tensorrt_llm/
kernels/cutlass_kernels/ — MoeGemmRunner<bf16,bf16,bf16,bf16> explicitly
instantiated in moe_gemm/moe_gemm_kernels_bf16_bf16.cu.

API (include/moe_gemm_kernels.h): GroupedGemmInput{A, B, C,
total_tokens_including_expert (DEVICE int64 cumulative rows/expert ==
variable-M), n, k, num_experts, activation_type, gemm_config, stream};
runner.moeGemm(inputs, TmaWarpSpecializedGroupedGemmInput{});
runner.getConfigs(false) lists configs; runner.isTmaWarpSpecialized(cfg)
filters to the sm80-style configs that ARE compiled (the TMA-WS ones are
not — the known build gate).

Probe (extend pie_driver_cuda_moe_probe in driver/cuda/src/ops/
flashinfer_moe.cu): allocate A 512x2048 bf16, B 128x2048x1536 (up, as
NON-gated n=2I with elementwise SwiGLU after) and 128x768x2048 (down),
C, offsets = 512 rows spread over 128 experts; for each non-TMA config:
set gemm_config, moeGemm, cudaDeviceSynchronize, catch — report which
configs RUN (dispatch-level truth; getWorkspaceSize lies). Shapes to
probe: (n=1536,k=2048) up non-gated, (n=2048,k=768) down,
activation_type=Identity/InvalidType per non-gated convention (check
moeGemm's act handling — moeGemmBiasAct is the act-fused variant; plain
moeGemm should be act-free).

If configs run: integrate as the prefill MoE path (route/gather already
exist from the aligned path; replace the two cublasGemmBatchedEx with two
moeGemm calls + launch_chunked_swiglu between; remember add_to_residual
must ACCUMULATE and decode control must stay flat). Rebuild driver, run
42_context_sweep prefill mode vs the 24.1k baseline. Then the c8 lever arm.

## PROBE RESULT (2026-07-31 ~00:20): non-gated CUTLASS route CONFIRMED

moe-dispatch-probe: 9/9 non-TMA configs RAN (executed + synchronized) at
up n=1536 k=2048 E=128, down n=2048 k=768 E=128, and the control shape.
The variable-M grouped GEMM route needs NO TMA-WS build.

Integration design (moe_block prefill path, env-gated
PIE_QWEN35_MOE_CUTLASS_PREFILL=1 for A/B):
- Reuse the aligned gather (block=16, ~3% padding at N=2048): rows are
  already expert-sorted-contiguous in aligned_expert_in; padding rows waste
  3% FLOPs and are ignored by the existing reorder.
- New tiny kernel (or extend launch_moe_align_decode): int64 CUMULATIVE
  padded-rows-per-expert on device = total_tokens_including_expert.
- Replace the two cublasGemmBatchedEx with MoeGemmRunner<bf16>::moeGemm
  (up n=2*Im, then launch_chunked_swiglu, then down n=H). Config: first
  running non-TMA config (tile shape ID 4, stages 2), env override index;
  autotune later.
- Correctness gate: parity vs the cuBLAS path on one forward (compare
  outputs), then prefill sweep vs the 24.1k baseline, decode control flat.

## CUTLASS ROUTE MEASURED (2026-07-31 ~00:50): viable but NOT faster

Integration (PIE_QWEN35_MOE_CUTLASS_PREFILL=1, default OFF): correct
(clean generations, r2=0.991 fit), decode control flat (5.197 / 0.0333 —
also clears yesterday's borderline blk64 slope as noise). Performance:
22,614 tok/s at config 0; all 9 non-TMA configs within 608-618 ms on the
single-sample rank (1.6% band). Batched cuBLAS baseline: 24,120. The
Ampere-family kernels are config-insensitive here and lose to cuBLAS.
Default stays cuBLAS; flag kept as infrastructure. Remaining prefill path:
generate the Hopper TMA-WS TU instantiations (build project).
