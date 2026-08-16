// EXPERIMENT 2: a key block spanning several pages, to amortize the softmax.
//
// Experiment 1 established what this is NOT: prefill attention is not
// memory-bound, because staging the K/V block to remove a nominal 32x read
// amplification made it 2.4x slower. So the 3x between 10.75 TFLOP/s and NAX's
// 32.5 is somewhere else, and the per-block scalar epilogue is the candidate.
//
// Per 32-key block a lane runs 16 MMAs and then, in scalar ALU:
//
//     16 masked comparisons, 16 exp2, two row reductions of four
//     simd_shuffle_xor each, and 64 accumulator rescales
//
// On a unit that retires the MMAs in a handful of cycles that is plausibly the
// whole cost, and it is paid once per 32 keys ONLY because `BK` was pinned to
// the page size. Nothing requires that. A 16-key fragment lies inside one page
// whatever the block is; the block just needs one page pointer per 32 keys
// instead of one per block.
//
// `NAXB_NPG` pages per block: 1 reproduces `sdpa_nax_prefill` exactly and is the
// control, so every ratio below is the epilogue and nothing else.
//
// THE GUARD THAT MATTERS. `n_blocks` rounds up, so the last block can reach for
// pages past the end of the cache -- at kp_hi+1 == 32 with NPG=2 it would read
// page 1 of a one-page list. Those slots take page 0 with a zero live count, so
// the read is in bounds and every score from it is masked. Reading the wrong
// page and masking it is safe; reading off the end of the array is not.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/nax_frag.h"

using namespace pie_nax;

#ifndef NAXB_NPG
#define NAXB_NPG 2
#endif
#ifndef NAXB_NWARPS
#define NAXB_NWARPS 4
#endif

kernel void sdpa_nax_bk(
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
  constexpr int NW = NAXB_NWARPS;
  constexpr int NPG = NAXB_NPG;
  constexpr int BQ = NW * kU;
  constexpr int PAGE = 32;
  constexpr int BK = PAGE * NPG;
  constexpr int D = 128;
  constexpr int TD = D / kU;      // 8
  constexpr int TK = BK / kU;     // 2 per page
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
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) {
    const short r = sm + i * kElemRowsJump;
    q_pos[i] = r < rows_here ? position_ids[row0 + r] : -1;
  }
  int kp_hi = -1;
  for (short i = 0; i < kElemRows; ++i) kp_hi = max(kp_hi, q_pos[i]);
  kp_hi = simd_max(kp_hi);

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

  const int live_pages = (kp_hi + PAGE) / PAGE;          // ceil((kp_hi+1)/32)
  const int n_blocks   = (live_pages + NPG - 1) / NPG;

  for (int kb = 0; kb < n_blocks; ++kb) {
    const device bfloat* Kp[NPG];
    const device bfloat* Vp[NPG];
#pragma clang loop unroll(full)
    for (short p = 0; p < NPG; ++p) {
      const int pi = kb * NPG + p;
      // Past the end of the list: point at page 0 and let the mask do the rest.
      const int page = pi < live_pages ? int(kv_page_indices[pi]) : 0;
      Kp[p] = k_pages + (size_t(page) * PAGE * n_kv_heads + kv_head) * D;
      Vp[p] = v_pages + (size_t(page) * PAGE * n_kv_heads + kv_head) * D;
    }

    // ── S = Q · Kᵀ over the whole block ──
    ffrag S[TK];
#pragma clang loop unroll(full)
    for (short i = 0; i < TK; ++i) S[i] = ffrag(0);

#pragma clang loop unroll(full)
    for (short id = 0; id < TD; ++id) {
      bfrag qf;
      frag_load(qf, Q + id * kU, q_ld, simd_lid);
      // One `frag_mma` covers N=32, which is exactly one page. So the fragment
      // pair and the page boundary coincide and no fragment ever straddles.
#pragma clang loop unroll(full)
      for (short p = 0; p < NPG; ++p) {
        const int base_key = kb * BK + p * PAGE;
        const short lim0 = short(clamp(kp_hi + 1 - base_key, 0, kU));
        const short lim1 = short(clamp(kp_hi + 1 - base_key - kU, 0, kU));
        bfrag kf0, kf1;
        frag_load_rows(kf0, Kp[p] + id * kU, kv_ld, lim0, simd_lid);
        frag_load_rows(kf1, Kp[p] + kU * kv_ld + id * kU, kv_ld, lim1, simd_lid);
        frag_mma<true>(S[2 * p], S[2 * p + 1], qf, kf0, kf1);
      }
    }

    // ── one epilogue for the WHOLE block: this is the experiment ──
#pragma clang loop unroll(full)
    for (short ik = 0; ik < TK; ++ik)
#pragma clang loop unroll(full)
      for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j) {
          const int key = kb * BK + ik * kU + sn + j;
          const short loc = i * kElemCols + j;
          const float v = S[ik][loc] * scale2;
          S[ik][loc] = (key <= q_pos[i]) ? v : NEG_INF;
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

    // The rescale is per BLOCK, so a wider block pays it less often too --
    // 64 multiplies per block whatever BK is.
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

    // ── O += P · V ──
#pragma clang loop unroll(full)
    for (short d = 0; d < TD; d += 2) {
#pragma clang loop unroll(full)
      for (short ik = 0; ik < TK; ++ik) {
        // NOT `half`: that is a Metal scalar type and shadowing it is a
        // parse error, not a warning.
        const short p = ik >> 1;
        const short hf = ik & 1;
        const int base_key = kb * BK + p * PAGE + hf * kU;
        const short vlim = short(clamp(kp_hi + 1 - base_key, 0, kU));
        const device bfloat* Vk = Vp[p] + hf * kU * kv_ld;
        bfrag v0, v1;
        frag_load_rows(v0, Vk + d * kU, kv_ld, vlim, simd_lid);
        frag_load_rows(v1, Vk + (d + 1) * kU, kv_ld, vlim, simd_lid);
        frag_mma<false>(O[d], O[d + 1], P[ik], v0, v1);
      }
    }
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
