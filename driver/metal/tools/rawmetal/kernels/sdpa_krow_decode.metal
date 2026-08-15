// A k-row decode that reads the KV cache ONCE.
//
// `results-speculation.md` measured the defect and named the fix. Today the
// driver has two attention shapes and a k-row decode fits neither:
//
//   sdpa_paged_decode  one query row per threadgroup, 32 simdgroups splitting
//                      that row's KEYS. Right when there is one row.
//   sdpa_paged_tiled   32 query rows per threadgroup sharing a staged key
//                      block. Right for a prefill, and measured 370 tok/s
//                      against 728 on a fleet of one-row decodes.
//
// A speculative verify fire -- and a co-batched concurrent decode -- is a third
// shape: FEW rows, ONE request, ALL sharing a key span. It falls to the per-row
// kernel and reads the whole cache k times. Measured slope
// `0.464 + 0.703*rows` ms per 1k of context: only 40% shared, so an 8-row fire
// costs 4.08x a 1-row fire at 16k.
//
// This keeps the decode kernel's key-parallel decomposition and gives each
// simdgroup all k query rows, so a key loaded once is used k times. Decode is
// bandwidth-bound here, so the extra arithmetic is close to free -- that is the
// whole bet, and the probe measures it rather than assuming it.
//
// Per lane: KROWS sets of q registers and KROWS accumulators. At KROWS=8,
// D=128, BD=32 that is 8*4 + 8*4 = 64 floats -- the same register pressure that
// spilled in the NAX experiment, so it is measured, not assumed.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef KROWS
#define KROWS 4
#endif

kernel void sdpa_krow_decode(
    const device bfloat* queries      [[buffer(0)]],   // [KROWS][n_q_heads][D]
    const device bfloat* k_pages      [[buffer(1)]],
    const device bfloat* v_pages      [[buffer(2)]],
    device bfloat* out                [[buffer(3)]],   // [KROWS][n_q_heads][D]
    const constant int& gqa_factor    [[buffer(4)]],
    const device int* position_ids    [[buffer(5)]],   // [KROWS]
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
  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;

  threadgroup U red[BN * BD];
  threadgroup U tg_max[BN];
  threadgroup U tg_sum[BN];

  // Every row's queries, held per lane. This is the change: k rows resident so
  // one loaded key serves all of them.
  U q[KROWS][qk_per_thread];
  U o[KROWS][v_per_thread];
  U row_max[KROWS], row_sum[KROWS];
  int q_pos[KROWS];

  for (int r = 0; r < KROWS; ++r) {
    const device bfloat* qp =
        queries + (size_t(r) * n_q_heads + q_head) * D + simd_lid * qk_per_thread;
    for (int j = 0; j < qk_per_thread; ++j) q[r][j] = U(scale) * U(qp[j]);
    for (int j = 0; j < v_per_thread; ++j) o[r][j] = 0;
    row_max[r] = NEG_INF;
    row_sum[r] = 0;
    q_pos[r] = position_ids[r];
  }

  // The last row sees the most keys; earlier rows mask the tail off. One walk
  // of the cache for all of them -- the entire point.
  int kp_hi = q_pos[0];
  for (int r = 1; r < KROWS; ++r) kp_hi = max(kp_hi, q_pos[r]);

  for (int kp = int(simd_gid); kp <= kp_hi; kp += BN) {
    const int page = int(kv_page_indices[kp / page_size]);
    const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
    const device bfloat* kptr =
        k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
    const device bfloat* vptr =
        v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;

    // ONE read of this key, reused by every row.
    U kv[qk_per_thread], vv[v_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) kv[j] = U(kptr[j]);
    for (int j = 0; j < v_per_thread; ++j) vv[j] = U(vptr[j]);

    for (int r = 0; r < KROWS; ++r) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; ++j) score += q[r][j] * kv[j];
      score = simd_sum(score);
      if (kp > q_pos[r]) continue;            // causal, per row
      U factor, exp_score;
      sdpa_online_update(score, row_max[r], row_sum[r], factor, exp_score);
      for (int j = 0; j < v_per_thread; ++j) o[r][j] = o[r][j] * factor + exp_score * vv[j];
    }
  }

  // Cross-simdgroup reduction, once per row. `red` is reused, so threadgroup
  // memory does not grow with KROWS.
  for (int r = 0; r < KROWS; ++r) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) { tg_max[simd_gid] = row_max[r]; tg_sum[simd_gid] = row_sum[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U m = tg_max[simd_lid];
    U new_max = simd_max(m);
    U fac = fast::exp(m - new_max);
    U tot = simd_sum(tg_sum[simd_lid] * fac);
    for (int i = 0; i < v_per_thread; ++i) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      red[simd_lid * BD + simd_gid] = o[r][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      U acc = simd_sum(red[simd_gid * BD + simd_lid] * fac);
      if (simd_lid == 0 && simd_gid < uint(v_per_thread)) {
        // simd_gid indexes the v element after the transpose above.
      }
      if (simd_lid == 0) {
        device bfloat* op = out + (size_t(r) * n_q_heads + q_head) * D + simd_gid * v_per_thread;
        op[i] = bfloat(tot == 0 ? acc : acc / tot);
      }
    }
  }
}
