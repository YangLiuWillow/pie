// EXPERIMENT 3: pay for the mask and the tail only where they are owed.
//
// Experiments 1 and 2 agree on what the kernel is bound by, by elimination:
// not memory (staging the block to remove a nominal 32x read amplification made
// it 2.4x slower) and not the softmax epilogue (making it run half as often
// made it 1.4x slower, and a quarter as often 4x slower, tracking live
// registers exactly). What is left is register pressure and the inner loop's
// own instruction count.
//
// So this removes work rather than rearranging it. Two things ran on every key
// block that are owed by almost none of them:
//
//   * **the tail check.** `frag_load_rows` tests `r < lim` per element -- eight
//     conditionals per fragment, on all ~232 blocks of a 7424-context fire,
//     when only the final block is ragged.
//   * **the causal mask.** 16 comparisons per lane per block, when a prefill of
//     184 rows at 7424 context has ~226 blocks lying ENTIRELY below the
//     diagonal and ~6 straddling it.
//
// MLX makes both conditional (`align_K && is_last_k`, and `kb >= kb_min_causal`)
// and that is the difference being tested here. The key loop splits in two:
//
//     kb <  kb_safe   every key of every row is live and unmasked -- branchless
//                     loads, no comparisons, no per-element predicates
//     kb >= kb_safe   the straddling and ragged tail, exactly as before
//
// `kb_safe` is derived from the SMALLEST live position in the simdgroup, so a
// block is only in the fast region when it is below the diagonal for every row
// the simdgroup owns. Dead rows (a partial final tile) take INT_MAX in that
// minimum rather than -1, or one absent row would drag every block into the
// slow region and the experiment would measure nothing.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/nax_frag.h"

using namespace pie_nax;

#ifndef NAXF_NWARPS
#define NAXF_NWARPS 4
#endif

// One key block, with the masking and the tail check compiled in or out.
// A template rather than a runtime flag: the whole point is that the fast
// region contains no predicates at all, and a branch the compiler keeps is a
// branch the experiment did not remove.
template <bool MASKED, int TD, int TK>
inline void nax_block(
    thread ffrag (&O)[TD], thread float (&row_max)[kElemRows],
    thread float (&row_sum)[kElemRows], const thread int (&q_pos)[kElemRows],
    const device bfloat* Q, const device bfloat* Kb, const device bfloat* Vb,
    int q_ld, int kv_ld, int kb, int kp_hi, float scale2, short sn, uint simd_lid) {
  constexpr int kU = 16, BK = 32;
  constexpr float NEG_INF = -3.0e38f;

  ffrag S[TK];
#pragma clang loop unroll(full)
  for (short i = 0; i < TK; ++i) S[i] = ffrag(0);

  const short klim = MASKED ? short(min(BK, kp_hi + 1 - kb * BK)) : short(BK);

#pragma clang loop unroll(full)
  for (short id = 0; id < TD; ++id) {
    bfrag qf, kf0, kf1;
    frag_load(qf, Q + id * kU, q_ld, simd_lid);
    if (MASKED) {
      frag_load_rows(kf0, Kb + id * kU, kv_ld, klim, simd_lid);
      frag_load_rows(kf1, Kb + kU * kv_ld + id * kU, kv_ld, short(klim - kU), simd_lid);
    } else {
      frag_load(kf0, Kb + id * kU, kv_ld, simd_lid);
      frag_load(kf1, Kb + kU * kv_ld + id * kU, kv_ld, simd_lid);
    }
    frag_mma<true>(S[0], S[1], qf, kf0, kf1);
  }

#pragma clang loop unroll(full)
  for (short ik = 0; ik < TK; ++ik)
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
      for (short j = 0; j < kElemCols; ++j) {
        const short loc = i * kElemCols + j;
        const float v = S[ik][loc] * scale2;
        if (MASKED) {
          const int key = kb * BK + ik * kU + sn + j;
          S[ik][loc] = (key <= q_pos[i]) ? v : NEG_INF;
        } else {
          S[ik][loc] = v;   // below the diagonal: every key is this row's
        }
      }

  float new_max[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) new_max[i] = row_max[i];
#pragma clang loop unroll(full)
  for (short ik = 0; ik < TK; ++ik) frag_row_reduce<MaxOp>(S[ik], new_max);

  float factor[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) {
    factor[i] = (row_max[i] == NEG_INF) ? 0.0f : exp2(row_max[i] - new_max[i]);
    row_max[i] = new_max[i];
  }
#pragma clang loop unroll(full)
  for (short ik = 0; ik < TK; ++ik)
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
      for (short j = 0; j < kElemCols; ++j) {
        const short loc = i * kElemCols + j;
        S[ik][loc] = exp2(S[ik][loc] - new_max[i]);
      }
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) row_sum[i] *= factor[i];
#pragma clang loop unroll(full)
  for (short ik = 0; ik < TK; ++ik) frag_row_reduce<SumOp>(S[ik], row_sum);

#pragma clang loop unroll(full)
  for (short d = 0; d < TD; ++d)
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
      for (short j = 0; j < kElemCols; ++j)
        O[d][i * kElemCols + j] *= factor[i];

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
      if (MASKED) {
        const short vlim = short(klim - ik * kU);
        frag_load_rows(v0, Vk + d * kU, kv_ld, vlim, simd_lid);
        frag_load_rows(v1, Vk + (d + 1) * kU, kv_ld, vlim, simd_lid);
      } else {
        frag_load(v0, Vk + d * kU, kv_ld, simd_lid);
        frag_load(v1, Vk + (d + 1) * kU, kv_ld, simd_lid);
      }
      frag_mma<false>(O[d], O[d + 1], P[ik], v0, v1);
    }
  }
}

kernel void sdpa_nax_fast(
    const device bfloat* queries      [[buffer(0)]],
    const device bfloat* k_pages      [[buffer(1)]],
    const device bfloat* v_pages      [[buffer(2)]],
    device bfloat* out                [[buffer(3)]],
    const constant int& gqa_factor    [[buffer(4)]],
    const device int* position_ids    [[buffer(5)]],
    const device uint* kv_page_indices[[buffer(6)]],
    const constant int& n_kv_heads    [[buffer(7)]],
    const constant float& scale       [[buffer(8)]],
    const constant int& n_rows        [[buffer(9)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int kU = 16;
  constexpr int BQ = NAXF_NWARPS * kU;
  constexpr int BK = 32, D = 128;
  constexpr int TD = D / kU, TK = BK / kU;
  constexpr float NEG_INF = -3.0e38f;

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  const int row0 = int(tid.y) * BQ + int(simd_gid) * kU;
  if (row0 >= n_rows) return;
  const short rows_here = short(min(kU, n_rows - row0));

  const short2 co = frag_coord(simd_lid);
  const short sm = co.y, sn = co.x;

  int q_pos[kElemRows];
  int lo = 0x7fffffff;
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) {
    const short r = sm + i * kElemRowsJump;
    const bool live = r < rows_here;
    q_pos[i] = live ? position_ids[row0 + r] : -1;
    // INT_MAX, not -1: a dead row must not drag the whole simdgroup into the
    // masked region, which is exactly what taking its -1 as a minimum would do.
    lo = live ? min(lo, q_pos[i]) : lo;
  }
  int kp_hi = -1;
  for (short i = 0; i < kElemRows; ++i) kp_hi = max(kp_hi, q_pos[i]);
  kp_hi = simd_max(kp_hi);
  const int kp_lo = simd_min(lo);

  const float scale2 = scale * 1.44269504088896340736f;

  ffrag O[TD];
#pragma clang loop unroll(full)
  for (short i = 0; i < TD; ++i) O[i] = ffrag(0);
  float row_max[kElemRows], row_sum[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) { row_max[i] = NEG_INF; row_sum[i] = 0; }

  const device bfloat* Q = queries + (size_t(row0) * n_q_heads + q_head) * D;
  const int q_ld = n_q_heads * D;
  const int kv_ld = n_kv_heads * D;

  const int n_blocks = (kp_hi + BK) / BK;
  // Blocks whose last key is at or below the SMALLEST live position: every row
  // of this simdgroup attends all 32 of them, so no mask and no tail check.
  const int kb_safe = min(n_blocks, (kp_lo + 1) / BK);

  for (int kb = 0; kb < kb_safe; ++kb) {
    const int page = int(kv_page_indices[kb]);
    const device bfloat* Kb = k_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    const device bfloat* Vb = v_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    nax_block<false, TD, TK>(O, row_max, row_sum, q_pos, Q, Kb, Vb, q_ld, kv_ld,
                             kb, kp_hi, scale2, sn, simd_lid);
  }
  for (int kb = kb_safe; kb < n_blocks; ++kb) {
    const int page = int(kv_page_indices[kb]);
    const device bfloat* Kb = k_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    const device bfloat* Vb = v_pages + (size_t(page) * BK * n_kv_heads + kv_head) * D;
    nax_block<true, TD, TK>(O, row_max, row_sum, q_pos, Q, Kb, Vb, q_ld, kv_ld,
                            kb, kp_hi, scale2, sn, simd_lid);
  }

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
