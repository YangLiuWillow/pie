# Agent handover — 2026-07-31 (H100, the long night session)

> Continues AGENT_HANDOVER_20260729.md. Narrative + numbers for 2026-07-30/31.
> Companion docs: DEFECTS_OVERCOMMIT.md (defect chronicle + resolutions),
> PLAN_OVERCOMMIT_FIX.md (plan trail), blog-pie-agent-serving.md (the public
> writeup, now informal style, updated through tonight).

## Thirty-second version

1. **Overcommit fixed.** c8 went 0/8 -> parity with vLLM (1.00-1.08x on
   matched instances) via five measured fixes (engine strict-bid eviction,
   SDK counter resync, wake-bid-everywhere, child destroy, destroy-no-trap).
   Repro: 50_run_repro.sh (~4 min).
2. **Prefill mapped end to end.** 12.0k -> 24.1k tok/s via chunk 2048 +
   64-row tiles. Ampere CUTLASS grouped: 22.6k (loses). Fused Hopper TMA-WS
   SwiGLU: BUILT, CORRECT (canonical patch), but 14.5k at default tactics —
   tactic sweep pending. Full-agent c1 with levers: in-call 147.4 tok/s
   (gap 1.09x); s/iter noise-dominated by trajectories.
3. **APC dissected.** 71% of c1 prefill = unavoidable cold starts. Shared
   namespace alone cannot cross-share (saves are full renders only — also
   true of chat-apc); head-snapshot fix implemented + self-reporting
   ("shared-hit" mode) but not yet observed — suspect per-instance bytes in
   the first render unit. chat-apc's retention pass ported
   (PIE_SNAPSHOT_RETENTION=N; validated zero-cost at c1).
4. **Lever inversion recorded**: chunk 2048 at c8 shrinks the pool to ~2.5
   seats and collapses; lever at c1, stock under overcommit.

Everything through `14b74063` is pushed to origin (fork only). GPU shared
with a codex agent — /workspace/GPU_LOCK_PROTOCOL.md, honor it.

## Key numbers (H100 80GB, Qwen3-Coder-30B-A3B, vLLM 0.25.1)

| | Pie | vLLM |
|---|---|---|
| c1 in-call tok/s (levers) | 147.4 | 160.4 (1.09x) |
| c8 s/iter matched instances | 4.26-4.68 | 3.90-4.68 (1.00-1.08x) |
| prefill marginal | 24.1k (aligned+levers) | 44.9k |
| decode intercept / KV-read | 5.17-5.22 ms / 2,930-3,015 GB/s | 4.69 / 2,829 |

## What to do next, in order

1. **Fused-path tactic sweep** (PIE_NEMOTRON_FLASHINFER_MOE_SELECT=raw +
   GEMM{1,2}_INDEX; lite pattern like the config sweep). Decides whether
   the TMA build closes the 1.86x or the aligned path stays champion.
2. **System-prompt diff** (two minutes, no GPU): dump messages[0] for two
   instances; settles why shared-hit never fired.
3. **c8 snapshot-mode arm with PIE_SNAPSHOT_RETENTION=10**: retention's
   real prize — bounded snapshot pinning was the original starvation
   ingredient.
4. **Tell shsym**: their 0/84 long-context crash (pie-vllm-test
   results/FINDINGS.md; illegal access at executor.cpp:2751) matches our
   fixed defect 2 class (stale-counter under-reservation at page
   boundaries, b7b872a4). Retest on this branch with a CLEAN CUDA rebuild
   (their revert test was invalidated by a stale build; attn_ws must read
   160 MiB).
5. Engine debts (DEFECTS_OVERCOMMIT.md): suspend() frees working pages
   without refcounting (masked by park-after-generation, not fixed);
   restore reject-loop spins hot; zero-bid legacy flows cannot evict.
6. Writeup: voice-check by the human; add fused-path paragraph after (1).

## Cautions from this session (new ones only)

- **kill by PID, never pkill -f**: self-matched AGAIN (killed own queued
  driver). The handover rule exists because it keeps happening.
- **Two cargo builds concurrently = "build script failed"** with no error;
  serialize builds.
- **grep-filtered task output loses error bodies**: capture full output to
  a file, grep afterward.
- **A stale ctxsweep dir will happily answer your ls -t**: verify the
  timestamp matches the run you launched.
- **str.replace on shared code patterns lands in the wrong kernel**:
  anchor edits uniquely; three incidents tonight.
- TRT-LLM gated MoE reads [linear; gate] — the MIRROR of Pie's [gate; up].
  The swap-at-load + gate_second machinery handles it; the strided host
  path is incompatible with the swapped layout (debug-only).
- The vendored-CPM .inl patch is pod-local: patches/ has the idempotent
  script; reapply after any pod rebuild BEFORE building the driver.

## Machine state

/root rebuilt 2026-07-30 ~18:40 (00_setup_h200.sh). Installed inferlet =
target build (retention + head-snapshot + shared-hit marker + live-context
+ destroy fix). pie binary = TMA-WS build + fused wrapper + gate-swap
(02:31). GPU lock protocol in force. Tokens: /workspace/.env has GH_TOKEN
(shsym repos readable); pushes to YangLiuWillow/pie use the PAT the human
provides per session.
