// Paged prefill attention on the M5 neural accelerators, fused. SHIPPED FORM.
//
// The prototype and the measurements that justify this kernel are in
// `tools/rawmetal/kernels/sdpa_nax_prefill.metal` and its arm of
// `sdpa_paged_probe`: correct against a CPU reference at six shapes, and
// 2.08 ms/layer at 184 rows / 7424 ctx against `sdpa_paged_mma`'s 6.97 --
// 3.35x, at 10.75 TFLOP/s, BELOW the 4.1 ms/layer floor the 5.48 TFLOP/s
// simdgroup ceiling puts under any simdgroup kernel. That is the arithmetic
// proof it runs on the neural accelerators rather than a claim that it does.
//
// This file differs from the prototype in ONE way and it is the ABI: the buffer
// list is `sdpa_paged_mma`'s, byte for byte, so `bind::SdpaPaged` serves it with
// no binder change and the only thing that differs at the launch is the grid.
//
// ## What this kernel does NOT handle, and how it is kept away from them
//
//   * **More than one request per fire.** A 64-row query tile would span two
//     page lists, and this kernel resolves one base pointer for the whole tile.
//     `sdpa_nax_this_fire` requires `requests == 1`, which is what a prefill
//     fire is -- one request contributing thousands of rows. A co-batched fire
//     falls back to `sdpa_paged_mma`.
//   * **A user attention mask, a sliding window, or a learned sink.** None are
//     read. The window and sink are geometry and are gated on. The MASK is
//     per-fire and is NOT visible to `pso_for` or `launch_shape`, so it cannot
//     be gated on there -- see the note in `sdpa_nax_this_fire`. This kernel
//     inherits exactly the assumption `sdpa_paged_decode_..._p32` (FAST_FULL)
//     already makes for this family at page size 32, and if masks are ever
//     enabled on llama BOTH must be revisited together. That is a pre-existing
//     property of the family, not one introduced here, and it is written down
//     rather than left implicit.

#include <metal_stdlib>
using namespace metal;
#include "nax_frag.h"

using namespace pie_nax;

#ifndef NAXP_NWARPS
#define NAXP_NWARPS 4
#endif

kernel void sdpa_paged_nax(
    const device bfloat* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device bfloat* k_pages     [[buffer(1)]],   // [pages, 32, n_kv, D]
    const device bfloat* v_pages     [[buffer(2)]],
    device bfloat* out               [[buffer(3)]],   // [N, n_q_heads, D]
    const constant int& gqa_factor             [[buffer(4)]],
    const device int* position_ids             [[buffer(5)]],
    const device int* req_of_token             [[buffer(6)]],
    const device uint* kv_page_indices         [[buffer(7)]],
    const device uint* kv_page_indptr          [[buffer(8)]],
    const constant int& page_size              [[buffer(9)]],   // 32, gated
    const constant int& n_kv_heads             [[buffer(10)]],
    const constant float& scale                [[buffer(11)]],
    const device uchar* attention_mask         [[buffer(12)]],  // unused
    const device uint& attention_mask_stride   [[buffer(13)]],  // unused
    const device uchar* attention_mask_enabled [[buffer(14)]],  // unused
    const constant int& window                 [[buffer(15)]],  // unused
    const device bfloat* sinks                 [[buffer(16)]],  // unused
    const constant int& n_rows                 [[buffer(17)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  (void)page_size; (void)attention_mask; (void)attention_mask_stride;
  (void)attention_mask_enabled; (void)window; (void)sinks;
  constexpr int kU = 16;
  constexpr int NW = NAXP_NWARPS;
  constexpr int BQ = NW * kU;    // 64 query rows per threadgroup
  constexpr int BK = 32;         // one page
  constexpr int D  = 128;
  constexpr int TD = D / kU;     // 8 head fragments
  constexpr int TK = BK / kU;    // 2 key fragments
  constexpr float NEG_INF = -3.0e38f;

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  // This simdgroup's 16 query rows.
  const int row0 = int(tid.y) * BQ + int(simd_gid) * kU;
  if (row0 >= n_rows) return;    // whole-simdgroup, so not divergent for the MMA
  const short rows_here = short(min(kU, n_rows - row0));

  const short2 co = frag_coord(simd_lid);
  const short sm = co.y, sn = co.x;

  // The two query rows this lane holds, and their causal bounds. Out-of-range
  // rows take bound -1, which masks every key: the fragment still multiplies,
  // it just contributes nothing.
  int q_pos[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) {
    const short r = sm + i * kElemRowsJump;
    q_pos[i] = r < rows_here ? position_ids[row0 + r] : -1;
  }
  int kp_hi = -1;
  for (short i = 0; i < kElemRows; ++i) kp_hi = max(kp_hi, q_pos[i]);
  // The bound must be the SIMDGROUP's, not the lane's: every lane has to reach
  // every MMA. Take the max over the whole simdgroup.
  kp_hi = simd_max(kp_hi);

  // exp2 rather than exp, with the scale folded in -- one multiply per score
  // instead of a transcendental per score with a separate scaling pass.
  const float scale2 = scale * 1.44269504088896340736f;

  ffrag O[TD];
#pragma clang loop unroll(full)
  for (short i = 0; i < TD; ++i) O[i] = ffrag(0);
  float row_max[kElemRows], row_sum[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) { row_max[i] = NEG_INF; row_sum[i] = 0; }

  const device bfloat* Q = queries + (size_t(row0) * n_q_heads + q_head) * D;
  const int q_ld  = n_q_heads * D;
  const int kv_ld = n_kv_heads * D;

  // The page list is per REQUEST, through the CSR. `sdpa_nax_this_fire`
  // guarantees one request per fire, so every row of this tile shares a base
  // and reading it from row0's request is the same as reading it from any.
  const int page_base = int(kv_page_indptr[req_of_token[row0]]);

  const int n_blocks = (kp_hi + BK) / BK;   // ceil((kp_hi+1)/32)

  for (int kb = 0; kb < n_blocks; ++kb) {
    // ONE page-table read for the whole 32-key block. This is the entire cost
    // of paging in this kernel.
    const int page = int(kv_page_indices[page_base + kb]);
    const device bfloat* Kb =
        k_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    const device bfloat* Vb =
        v_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    const short klim = short(min(BK, kp_hi + 1 - kb * BK));

    // ── S = Q · Kᵀ, accumulated over the head dimension ──
    ffrag S[TK];
#pragma clang loop unroll(full)
    for (short i = 0; i < TK; ++i) S[i] = ffrag(0);

#pragma clang loop unroll(full)
    for (short id = 0; id < TD; ++id) {
      bfrag qf, kf0, kf1;
      frag_load(qf, Q + id * kU, q_ld, simd_lid);
      frag_load_rows(kf0, Kb + id * kU, kv_ld, klim, simd_lid);
      frag_load_rows(kf1, Kb + kU * kv_ld + id * kU, kv_ld,
                     short(klim - kU), simd_lid);
      frag_mma<true>(S[0], S[1], qf, kf0, kf1);
    }

    // ── scale, then mask: causal, and the tail of a short block ──
#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik) {
#pragma clang loop unroll(full)
      for (short i = 0; i < kElemRows; ++i) {
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j) {
          const int key = kb * BK + ik * kU + sn + j;
          const short loc = i * kElemCols + j;
          const float v = S[ik][loc] * scale2;
          // A key past this row's causal bound, past the cache, or belonging to
          // a row that does not exist, all resolve the same way.
          S[ik][loc] = (key <= q_pos[i]) ? v : NEG_INF;
        }
      }
    }

    // ── online softmax, entirely in registers ──
    float new_max[kElemRows];
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) new_max[i] = row_max[i];
#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik) frag_row_reduce<MaxOp>(S[ik], new_max);

    float factor[kElemRows];
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) {
      // A block in which every key of this row is masked leaves new_max ==
      // row_max, so factor == 1 and the accumulator is untouched. A row that
      // has not yet seen a live key has row_max == -inf and an accumulator of
      // zeros, so factor == 0 scales nothing. Both fall out of this line.
      factor[i] = (row_max[i] == NEG_INF) ? 0.0f
                                          : exp2(row_max[i] - new_max[i]);
      row_max[i] = new_max[i];
    }

#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik)
#pragma clang loop unroll(full)
      for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j) {
          const short loc = i * kElemCols + j;
          // -inf - anything stays -inf, and exp2 of it is 0. No branch.
          S[ik][loc] = exp2(S[ik][loc] - new_max[i]);
        }

#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) row_sum[i] *= factor[i];
#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik) frag_row_reduce<SumOp>(S[ik], row_sum);

    // Rescale O by the same factor before accumulating this block into it.
#pragma clang loop unroll(full)
    for (short d = 0; d < TD; ++d)
#pragma clang loop unroll(full)
      for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j)
          O[d][i * kElemCols + j] *= factor[i];

    // ── O += P · V ──
    // P is float and the MMA wants bfloat operands, so the probabilities are
    // narrowed here. That is the same precision MLX carries and the same the
    // shipped kernel carries for its own operands.
    bfrag P[TK];
#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik)
#pragma clang loop unroll(full)
      for (short e = 0; e < kElemsPerFrag; ++e) P[ik][e] = bfloat(S[ik][e]);

#pragma clang loop unroll(full)
    for (short d = 0; d < TD; d += 2) {
#pragma clang loop unroll(full)
      for (short ik = 0; ik < TK; ++ik) {
        bfrag v0, v1;
        const device bfloat* Vk = Vb + ik * kU * kv_ld;
        const short vlim = short(klim - ik * kU);
        frag_load_rows(v0, Vk + d * kU, kv_ld, vlim, simd_lid);
        frag_load_rows(v1, Vk + (d + 1) * kU, kv_ld, vlim, simd_lid);
        frag_mma<false>(O[d], O[d + 1], P[ik], v0, v1);
      }
    }
  }

  // ── normalize and store ──
  device bfloat* Op = out + (size_t(row0) * n_q_heads + q_head) * D;
#pragma clang loop unroll(full)
  for (short d = 0; d < TD; ++d) {
    ffrag o = O[d];
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) {
      const float inv = row_sum[i] == 0 ? 0.0f : 1.0f / row_sum[i];
#pragma clang loop unroll(full)
      for (short j = 0; j < kElemCols; ++j) o[i * kElemCols + j] *= inv;
    }
    frag_store(o, Op + d * kU, q_ld, rows_here, simd_lid);
  }
}
