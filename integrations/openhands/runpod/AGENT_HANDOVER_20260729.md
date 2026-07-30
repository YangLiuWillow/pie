# Agent handover — 2026-07-29 (H100)

> **Numbers live in [`results_20260729_h100.md`](results_20260729_h100.md).** This
> file is the narrative: what changed, what is closed, what to do next.
>
> **Supersedes for FINDINGS:** `AGENT_HANDOVER_20260728.md` §10 (H200 concurrency)
> is not wrong, but it was measured on different hardware — see §2.
> **Machine state and storage rules:** `AGENT_HANDOVER_H200.md` §3 (the MooseFS
> rule) and §6 (version pins) still apply verbatim.
> **`START_HERE_H200.md` §"30-second version" is partly wrong on H100** — it says
> don't set `VLLM_TUNED_CONFIG_FOLDER`. On H100 you must. `00_setup_h200.sh` now
> resolves this automatically; do not follow that line by hand.

---

## 1. The thirty-second version

The H100 re-baseline found that **prefill, not decode, was the entire remaining
gap to vLLM** — and fixed most of it with a one-line threshold change.

| | before | after |
|---|---|---|
| Pie prefill (agent workload) | 2,890 tok/s | **8,610 tok/s** |
| Pie s/iter (13 instances, c1) | 1.329 | **1.204** |
| Pie in-call throughput | 125.2 tok/s | **144.4 tok/s** |
| gap to vLLM `fair` (in-call) | 1.29× | **1.11×** |

Decode was already at parity and **Pie's decode attention now beats
FlashAttention-3 on this GPU** (KV read 2,988 vs 2,829 GB/s; 89% vs 84% of peak).

Everything is committed and pushed to `origin/openhands-integration-updated`
(`YangLiuWillow/pie`). Nothing went near upstream.

## 2. Read this before comparing to any H200 number

This pod is an **H100 80GB**, not the H200 the 2026-07-28/29 series ran on.
Consequences, all measured:

- **Absolute latencies are ~1.4× higher** and that is physics, not regression:
  H200 pie c1 p50 was 0.92 s against 1.277 s here = 1.39×, versus an
  HBM3e/HBM3 bandwidth ratio of 1.43×. Per-call time on this workload is
  bandwidth-dominated.
- **KV is ~5× smaller**: `kv_tokens=124032` here against the H200 sweep arms'
  622,112. Both engines came in under a naive estimate; see the results file for
  the measured per-engine pair (Pie 124,032 / vLLM 117,599 tokens).
- **`MAX_MODEL_LEN` is 113664, not 131072.** vLLM physically cannot serve 131072
  on an 80 GB board. Not binding (largest observed prompt 46,870) but a protocol
  deviation that must be stated.
- **The c8/c16 axis was NOT re-run here** and should not be inferred from H200's
  1.39× at c8. At 1.03× concurrency headroom a concurrent arm on this board is
  measuring KV pressure the H200 arms never saw. See §5.

## 3. What changed in the tree

| commit | what |
|---|---|
| `a03a6e30` | **the fix** — route MoE prefill through the on-device path |
| `49842b11` | CUTLASS MoE shape probe in `pie driver cuda-native doctor` |
| `a8fd0100` | harness made board-aware (was H200-only in several silent ways) |
| `ef1f150f` | `prefill` mode for the context sweep; fixed its `--mode` plumbing |
| `0a739ff3` | the measurements |
| `8c79a77c`, `7d28dc43`, `c1516036` | CUTLASS SwiGLU attempt, corrections, and revert |
| `358cf540` | aligned-decode negative result + paired c1 numbers |

**`7d28dc43`'s title is WRONG.** It says "SwiGLU at Qwen3-MoE's shape IS
supported". It is not — see §4. `c1516036` supersedes it. The commit is kept for
the API findings in its body, which are correct.

### The fix, and how to check it is live

`qwen3_5_moe_forward.cpp`: `PIE_QWEN35_MOE_DECODE_FAST_N` ceiling 128 → 8192,
default flipped to always-on-device. The prefill branch of `moe_block()` was
host-orchestrated: per LAYER a D2H copy of the routing table plus a **full
`cudaStreamSynchronize`**, then a 128-iteration expert loop. ~43,000 GPU ops and
48 pipeline stalls per forward, independent of token count.

To confirm it is active, read `pie_timings.phases` from any predictions row:
prefill should be **~8% of per-call time, not ~20%**, and
`prompt_tokens_prefilled / prefill_total_s` should be **~8,500-9,500 tok/s**, not
~2,900. `PIE_QWEN35_MOE_DECODE_FAST_N=64` restores the old path for A/B.

## 4. CLOSED LEVERS — do not retry these

Four separate attempts at the residual gap were measured and closed. Each looked
compelling on paper.

1. **Aligned block size** (`PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK` 16/32/64).
   +3.7% at 32, −1.9% at 64. `aligned_rows` uses a worst-case bound AT RUNTIME,
   so 32 and 64 cost 1.34× and 2.02× more issued FLOPs. A ~4% effect.
2. **Aligned path for single-stream decode** (`..._MIN_ROUTES=8`). **13.4% WORSE**
   (intercept 5.214 → 5.914 ms/tok). Rules out the GEMV shape as the cause of the
   weight-read inefficiency — see item 4.
3. **MoE kernel selection** — closed previously, `AGENT_HANDOVER_20260728.md`
   §6a-bis.
4. **The "29% of peak" weight read is NOT a Pie deficiency.** This was my error
   and it cost an experiment: I read Pie's decode intercept (5.06 GB/token at
   0.97 TB/s = 29% of peak) against its own attention (89%) and called the 3.1×
   gap Pie headroom. **vLLM reads the same weights at 1.08 TB/s = 32%.** Both
   engines sit at ~30% on batch-1 MoE weight reads; Pie is 11% behind on that
   term, not 3×. There is no large recoverable decode headroom, which is exactly
   why item 2 could not win.

**CUTLASS fused MoE with SwiGLU — blocked on a BUILD, not a capability.** The
Hopper TMA warp-specialized MoE GEMM launchers are not compiled: the gate is
`#ifndef COMPILE_HOPPER_TMA_GROUPED_GEMMS`
(`moe_gemm_template_dispatch_tma_ws.h:136`). Defining it is not a one-line change
— the vendored tree ships only `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`
and no `.cu` instantiating it, so enabling it fails at link with hundreds of
undefined `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm90, ...>`
symbols. Upstream GENERATES those TUs (`build_wheel.py --arch 90-real`).

**That failure mode is dangerous:** it is a `TLLM_THROW` at DISPATCH time, after
the model loads. Probing the workspace first does NOT protect you —
`getWorkspaceSize` returns a confident 40 MiB for a kernel that does not exist,
and the dispatch then kills the driver ("driver did not emit capabilities within
600.0s", zero rows). The `doctor` probe now prints a banner saying its `OK` rows
mean "shape is describable", not "kernel exists". **Do not delete that banner.**

The lower-level `MoeGemmRunner` split route is **not** a way around this — same
uncompiled launchers.

## 5. What to do next, in order

**1. Get a board with KV headroom and measure the concurrency axis.**
Highest value, and it is a MEASUREMENT not an optimisation. c1 is Pie's *worst*
case: on H200 Pie gained on vLLM as load rose (1.39× at c8 → 1.21× at c16, vLLM
saturating while Pie did not). This board cannot test that honestly. An H200 also
restores comparability with the whole §10 series.

**2. Prefill, if you want the last few points at c1.** ~6.5 of the ~15% residual
excess. Requires generating the Hopper TU instantiations (§4), a large compile,
and accepting that it widens the tactic set Nemotron-H selects from. Bounded and
now well understood — but do the `doctor` probe first, every time.

**3. `Context::suspend()` — NOT testable as the tree stands.** Investigated
2026-07-29; three preconditions are all unmet:
   - The inferlet holds **no live `Context` across requests**. Each request does
     `Context::new` → `open` → work → `save` → drop. KV is pinned by the saved
     *snapshot*, not by the object `suspend()` acts on. Holding a Context live is
     the prerequisite, and `lib.rs:650` calls it out as a deliberately deferred
     separate experiment — it moves the KV path off content-addressed snapshots,
     which is what keeps `--kv-verify` and reuse % comparable to prior arms.
   - **The prize is zero at c1.** Pie uses 38% of its KV pool and vLLM reported
     0 preemptions. Contention only appears at c4+ (4 × 46,870 = 1.51× the pool).
   - `swap_pool_size` is pinned at 0 — it is the one Pie-only non-default and
     vLLM has no equivalent, so enabling it makes the arm a different experiment.

   Correct order: board with headroom → measure c8/c16 → live-`Context` as its own
   experiment → `suspend()` on top. Two changes, two measurements, or the result
   cannot be attributed. Note §6b frames this as a **capability** claim, not a
   speed one; demonstrate it where it can pay.

**Do not bother with:** the fixed per-call term (1.14×, 147.6 vs 129.6 ms, ≈1% of
the arm), or the decode intercept (closed twice, and §4 item 4 explains why).

## 6. Cautions from this session

- **My prediction record was 0 for 4** on which lever would move: block=32 "free"
  (it costs 34% more FLOPs), CUTLASS "close" (twice), aligned-decode "clear
  mechanism, large ceiling" (13% regression). What was reliable was
  *decomposition* — the prefill diagnosis held up exactly. **Prefer a probe over a
  prediction for anything above ~30 minutes of work.** Three of those four cost
  under 30 minutes each precisely because they were measured, not built; the
  CUTLASS route cost hours because it was reasoned about first.
- **`getWorkspaceSize`'s third parameter is `fc1_output_size`, not `inter_size`**
  — they differ for gated activations (`2*inter`), while `runMoe` takes
  `inter_size` in the corresponding slot. The two calls want different values
  from one shape. Passing `inter_size` to both is right for Relu2 and silently
  wrong for SwiGLU. Do not "make them consistent".
- **That CUTLASS runner is order-dependent.** `getMaxWorkspaceSize` caches on
  `num_experts_`; a cold first call can report "Could not find valid config" for a
  shape it serves on the next call. The `doctor` probe repeats one row to make
  this visible.
- **`add_to_residual` is a trap for any new MoE path.** It points `moe_out` at
  `ws.y` (the live residual) and relies on the tail kernel ACCUMULATING. A kernel
  that OVERWRITES its output silently erases the residual stream — every layer's
  attention output lost, presenting as a model-quality problem, not a plumbing
  one. Decide your path's availability BEFORE `add_to_residual` is computed.
- **A "complete-looking" arm can still be empty.** Check `agent_iterations > 0`,
  non-empty `model_patch`, non-empty `response_latencies`, and `error == ""` on
  every row. `summarize_ab.py` will happily average zeros.
- **`pgrep -f` / `pkill -f` match your own shell.** Kill by PID; verify with
  `ps -o pid,ppid,etime,cmd -p <pid>` and `nvidia-smi`.

## 7. Escalate, do not improvise

Unchanged from `AGENT_HANDOVER_H200.md` §8. Ask before: changing a version pin,
changing the instance set, switching vLLM versions, or anything that invalidates a
collected arm. Note that **changing `MAX_MODEL_LEN` is now a hard failure rather
than a silent default** — `run_litellm_baseline_fair.sh` and
`10_vllm_serve_fair.sh` refuse to guess, because the old 32768 fallback gave vLLM
a 4× smaller context than Pie and read as an accuracy difference.
