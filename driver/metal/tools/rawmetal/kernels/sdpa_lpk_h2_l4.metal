// Decode attention with LANES_PER_KEY as a dial, not a choice between two ends.
//
// ## Why this exists: both ends were measured and both are wrong
//
// The shipped kernel puts 32 lanes on ONE key: each lane multiplies D/32 = 4
// dimensions and a five-step `simd_sum` folds them, feeding a dependent
// online-softmax update. 232 of those chains run end to end at 7424 context.
//
// `sdpa_kpl_decode.metal` put ONE lane on each key, removing the reduction
// entirely. It is correct at every context and **0.53x the speed at QH=2, 0.31x
// at QH=4** -- measured on warmed clocks against the shipped kernel interleaved.
// Two things it did not pay for:
//
//   * **q left the registers.** A lane owning a whole key needs the whole 128-
//     long q row, which does not fit, so it goes to threadgroup memory and is
//     read back 128 times per head per key. The shipped shape keeps 4 floats in
//     registers.
//   * **K became a 32-way scatter.** Each lane pulls its own key's 256 B, so one
//     load instruction touches 32 separate regions where the shipped shape
//     touches one.
//
// The reductions really are 32x fewer. They just cost less than what removing
// them buys back.
//
// ## The dial
//
// LPK lanes cooperate on one key, so 32/LPK keys are in flight per simdgroup:
//
//     LPK   q regs/lane/head   keys in flight   shuffles/key   K regions/instr
//      32          4                 1              5.0             1     shipped
//      16          8                 2              4.0             2
//       8         16                 4              3.0             4
//       4         32                 8              2.0             8
//       1        128                32              0.0            32     rejected
//
// Every row reads each key's 256 B contiguously; only the NUMBER of separate
// regions per instruction changes. The bet is that the middle of this table
// beats both ends -- fewer dependent softmax updates and a shorter shuffle
// chain, while q is still in registers and K is still nearly coalesced.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef HEADS
#define HEADS 2
#endif
#ifndef LPK
#define LPK 4
#endif

kernel void sdpa_lpk_decode(
    const device bfloat* queries      [[buffer(0)]],
    const device bfloat* k_pages      [[buffer(1)]],
    const device bfloat* v_pages      [[buffer(2)]],
    device bfloat* out                [[buffer(3)]],
    const constant int& gqa_factor    [[buffer(4)]],
    const device int* position_ids    [[buffer(5)]],
    const device int* req_of_token    [[buffer(6)]],
    const device uint* kv_page_indices[[buffer(7)]],
    const device uint* kv_page_indptr [[buffer(8)]],
    const constant int& page_size     [[buffer(9)]],
    const constant int& n_kv_heads    [[buffer(10)]],
    const constant float& scale       [[buffer(11)]],
    const device uchar* attention_mask         [[buffer(12)]],
    const device uint& attention_mask_stride   [[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device bfloat* sinks                 [[buffer(16)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  (void)page_size; (void)attention_mask; (void)attention_mask_stride;
  (void)attention_mask_enabled; (void)window; (void)sinks;
  constexpr int D = 128, V = 128, BN = 32, BD = 32;
  constexpr int QH  = HEADS;
  constexpr int per = D / LPK;      // q slice a lane owns, still in registers
  constexpr int KPI = 32 / LPK;     // keys a simdgroup has in flight
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  const int group     = int(tid.x);
  const int head_base = group * QH;
  const int kv_head   = head_base / gqa_factor;

  // Lanes are grouped so the LPK lanes of one key hold CONTIGUOUS slices of D.
  // That is what keeps a key's 256 B one region: `pos` is the slice, `sub` is
  // which of the keys in flight this lane serves.
  const int sub = int(simd_lid) / LPK;
  const int pos = int(simd_lid) % LPK;

  threadgroup U red[BN * BD];
  threadgroup U tg_max[BN];
  threadgroup U tg_sum[BN];

  U q[QH][per], o[QH][v_per_thread];
  U row_max[QH], row_sum[QH];
  for (int h = 0; h < QH; ++h) {
    const device bfloat* qp = queries + size_t(head_base + h) * D + pos * per;
    for (int j = 0; j < per; ++j) q[h][j] = U(scale) * U(qp[j]);
    for (int j = 0; j < v_per_thread; ++j) o[h][j] = 0;
    row_max[h] = NEG_INF;
    row_sum[h] = 0;
  }

  const int r         = req_of_token[0];
  const int q_pos     = position_ids[0];
  const int page_base = int(kv_page_indptr[r]);
  const int total     = q_pos + 1;

  for (int b = int(simd_gid) * KPI; b < total; b += BN * KPI) {
    const int key  = b + sub;
    const bool live = key < total;
    const int kr   = live ? key : 0;
    const int page = int(kv_page_indices[page_base + (kr >> 5)]);
    const size_t slot = size_t(page) * 32 + size_t(kr & 31);
    const device bfloat* kp =
        k_pages + (slot * n_kv_heads + kv_head) * D + pos * per;

    U kv[per];
    for (int j = 0; j < per; ++j) kv[j] = U(kp[j]);

    for (int h = 0; h < QH; ++h) {
      U score = 0;
      for (int j = 0; j < per; ++j) score += q[h][j] * kv[j];
      // Reduce over the LPK lanes that share this key, and ONLY those: lanes of
      // one key differ solely in the low log2(LPK) bits, so an xor by anything
      // below LPK stays inside the group.
#pragma clang loop unroll(full)
      for (int off = LPK / 2; off >= 1; off >>= 1) score += simd_shuffle_xor(score, ushort(off));
      if (!live) score = NEG_INF;

      // ONE max and ONE sum for the KPI keys in flight. Every key's score is
      // replicated across its LPK lanes, so a full-simdgroup max is the max over
      // the distinct keys, and a full-simdgroup sum counts each key LPK times.
      const U bmax    = simd_max(score);
      const U new_max = max(row_max[h], bmax);
      const U factor  = fast::exp(row_max[h] - new_max);
      const U p       = fast::exp(score - new_max);
      row_sum[h] = row_sum[h] * factor + simd_sum(p) / U(LPK);
      row_max[h] = new_max;

      // V goes back to the original layout, where all 32 lanes own a slice and
      // the accumulation needs no reduction at all.
      for (int j = 0; j < v_per_thread; ++j) o[h][j] *= factor;
      for (int u = 0; u < KPI; ++u) {
        const int ku = b + u;
        if (ku >= total) break;              // uniform across the simdgroup
        const U pu = simd_shuffle(p, ushort(u * LPK));
        const int pg = int(kv_page_indices[page_base + (ku >> 5)]);
        const size_t sl = size_t(pg) * 32 + size_t(ku & 31);
        const device bfloat* vp =
            v_pages + (sl * n_kv_heads + kv_head) * V + simd_lid * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) o[h][j] += pu * U(vp[j]);
      }
    }
  }

  for (int h = 0; h < QH; ++h) {
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
        device bfloat* op = out + size_t(head_base + h) * V + simd_gid * v_per_thread;
        op[i] = bfloat(tot == 0 ? acc : acc / tot);
      }
    }
  }
}
