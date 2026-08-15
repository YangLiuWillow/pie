// A decode attention that reads each KV head ONCE for all the query heads that
// share it.
//
// ## The defect
//
// `sdpa_paged_decode` launches one threadgroup per (query head, row) and each
// one walks the whole cache for its own kv head. Qwen3-Coder-30B has 32 query
// heads over 4 kv heads, so **eight threadgroups read the same K and V** and the
// hardware sees eight times the traffic on the one thing a decode step is
// bandwidth-bound on.
//
// Measured, on this workload rather than a synthetic one: at 23k context the
// dispatch trace puts `sdpa_paged_decode_bfloat16_d_128_p32` at **75-77% of a
// decode fire**. The unique KV at 12k context is 1.21 GB, which is 4.1 ms at
// this machine's 298 GB/s roof; the fire spends about 20 ms there.
//
// ## Why the head axis and not the row axis
//
// `sdpa_krow_decode.metal` shares a KV read across k QUERY ROWS, which is the
// right fix for a speculative verify and for co-batched concurrent decode. It
// does nothing for the case an agent actually runs: **one stream, one token, so
// k = 1**. The head axis is 8 wide at k = 1 -- it is the only sharing available
// to a single-stream decode, and single-stream decode is what an agentic turn
// is made of.
//
// The two compose (heads x rows) and this file deliberately keeps the k-row
// kernel's shape so they can later be one kernel.
//
// ## The register wall this is written against
//
// Per lane: HEADS sets of q registers and HEADS accumulators, `8*HEADS` floats
// at D=128, BD=32. The k-row experiment measured where that wall is on this
// device and it is not far away: KROWS=5 (40 floats) is linear, KROWS=6 (48)
// costs a 2.4x step on ONE extra row -- register spill. So HEADS is a template
// parameter and the probe SWEEPS it rather than assuming 8 fits. HEADS=1 is the
// same kernel doing exactly what `sdpa_paged_decode` does, so the HEADS/1 ratio
// isolates head sharing and nothing else.
//
// HEADS must divide `gqa_factor`, otherwise a threadgroup would span two kv
// heads and the single `kv_head` below would be wrong for half its work. The
// host is responsible for that; at gqa 8 the legal values are 1, 2, 4, 8.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef HEADS
#define HEADS 4
#endif

kernel void sdpa_hshare_decode(
    const device bfloat* queries      [[buffer(0)]],   // [1][n_q_heads][D]
    const device bfloat* k_pages      [[buffer(1)]],
    const device bfloat* v_pages      [[buffer(2)]],
    device bfloat* out                [[buffer(3)]],   // [1][n_q_heads][D]
    const constant int& gqa_factor    [[buffer(4)]],
    const device int* position_ids    [[buffer(5)]],   // [1]
    const device uint* kv_page_indices[[buffer(6)]],
    const constant int& page_size     [[buffer(7)]],
    const constant int& n_kv_heads    [[buffer(8)]],
    const constant float& scale       [[buffer(9)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int D = 128, V = 128, BN = 32, BD = 32;
  constexpr int qk_per_thread = D / BD;   // 4
  constexpr int v_per_thread  = V / BD;   // 4
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  // tid.x indexes a GROUP of HEADS query heads, not a single head.
  const int group     = int(tid.x);
  const int n_q_heads = int(tpg.x) * HEADS;
  const int head_base = group * HEADS;
  const int kv_head   = head_base / gqa_factor;

  threadgroup U red[BN * BD];
  threadgroup U tg_max[BN];
  threadgroup U tg_sum[BN];

  U q[HEADS][qk_per_thread];
  U o[HEADS][v_per_thread];
  U row_max[HEADS], row_sum[HEADS];

  for (int h = 0; h < HEADS; ++h) {
    const device bfloat* qp =
        queries + size_t(head_base + h) * D + simd_lid * qk_per_thread;
    for (int j = 0; j < qk_per_thread; ++j) q[h][j] = U(scale) * U(qp[j]);
    for (int j = 0; j < v_per_thread; ++j) o[h][j] = 0;
    row_max[h] = NEG_INF;
    row_sum[h] = 0;
  }

  const int q_pos = position_ids[0];

  for (int kp = int(simd_gid); kp <= q_pos; kp += BN) {
    const int page = int(kv_page_indices[kp / page_size]);
    const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
    const device bfloat* kptr =
        k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
    const device bfloat* vptr =
        v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;

    // ONE read of this key and value, reused by every query head in the group.
    // This is the whole point of the file; everything else is bookkeeping.
    U kv[qk_per_thread], vv[v_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) kv[j] = U(kptr[j]);
    for (int j = 0; j < v_per_thread; ++j) vv[j] = U(vptr[j]);

    for (int h = 0; h < HEADS; ++h) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; ++j) score += q[h][j] * kv[j];
      score = simd_sum(score);
      U factor, exp_score;
      sdpa_online_update(score, row_max[h], row_sum[h], factor, exp_score);
      for (int j = 0; j < v_per_thread; ++j) o[h][j] = o[h][j] * factor + exp_score * vv[j];
    }
  }

  // Cross-simdgroup reduction, once per head. `red` is reused between heads, so
  // threadgroup memory does not grow with HEADS -- which matters, because 32 KB
  // is the whole budget on this device (queried, not assumed).
  for (int h = 0; h < HEADS; ++h) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) { tg_max[simd_gid] = row_max[h]; tg_sum[simd_gid] = row_sum[h]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U m = tg_max[simd_lid];
    U new_max = simd_max(m);
    U fac = fast::exp(m - new_max);
    U tot = simd_sum(tg_sum[simd_lid] * fac);
    for (int i = 0; i < v_per_thread; ++i) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      red[simd_lid * BD + simd_gid] = o[h][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      U acc = simd_sum(red[simd_gid * BD + simd_lid] * fac);
      if (simd_lid == 0) {
        device bfloat* op =
            out + size_t(head_base + h) * D + simd_gid * v_per_thread;
        op[i] = bfloat(tot == 0 ? acc : acc / tot);
      }
    }
  }
}
