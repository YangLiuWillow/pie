// EXPERIMENT 1: stage the K/V block once, and share it across simdgroups AND
// across the query heads of a GQA group.
//
// ## The defect being chased
//
// `sdpa_nax_prefill` loads every K/V fragment straight from device memory, per
// simdgroup. At the serving shape that block is read
//
//     4 times over  -- the four simdgroups of a 64-row query tile each load it
//     8 times again -- the eight query heads sharing one KV head each load it
//
// i.e. 32x the unique bytes. The roofline says that matters: one threadgroup
// reads its KV head's whole K and V (3.80 MB at 7424 ctx), 96 threadgroups run,
// so 365 MB per layer = 1.23 ms at this machine's 296 GB/s -- against 0.69 ms
// of NAX compute and 2.08 ms measured. **Prefill attention is memory-bound**,
// and the in-situ speedup decaying from 3.14x on fire 1 to 2.48x on fire 6 is
// what memory pressure growing with the cache looks like.
//
// ## Why this can go wider than the decode kernel did
//
// `sdpa_paged_decode_hshare` had to stop at 2 heads because the grid is
// `n_q_heads / QH` threadgroups and a decode has one row, so at QH=4 there were
// eight threadgroups and the device went idle. A prefill fire is 4096 rows:
// 32 heads x 64 tiles = 2048 threadgroups at QH=1. Dividing that by 4 still
// leaves 512. **Occupancy is not the binding constraint here**, so QH is swept
// rather than assumed to stop at 2.
//
// ## The shape
//
//   * A threadgroup is QH query heads x RT row-tiles of simdgroups. Every
//     simdgroup still owns 16 query rows of one head, so REGISTERS PER LANE ARE
//     UNCHANGED (O is 8 fragments either way) -- this trades threads and
//     threadgroup memory for bandwidth, not registers.
//   * QH must divide `gqa_factor`, or a threadgroup would span two KV heads and
//     the single staged block would be wrong for half of it.
//   * Staging is 32 keys x 128 dims x 2 B, for K and V: 16 KB of the 32 KB this
//     device reports. Independent of QH, which is what makes QH cheap.
//
// ## The thing that will bite if it is got wrong
//
// **No early return.** The device-memory kernel opens with
// `if (row0 >= n_rows) return;`, which is safe there because there are no
// barriers. Here a simdgroup that returns never reaches the staging barrier and
// the threadgroup hangs. Out-of-range simdgroups therefore participate in every
// barrier and simply store nothing.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/nax_frag.h"

using namespace pie_nax;

#ifndef NAXS_QH
#define NAXS_QH 2
#endif
#ifndef NAXS_RT
#define NAXS_RT 4
#endif

kernel void sdpa_nax_stg(
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
  constexpr int QH = NAXS_QH, RT = NAXS_RT;
  constexpr int BQ = RT * kU;
  constexpr int BK = 32, D = 128;
  constexpr int TD = D / kU, TK = BK / kU;
  constexpr int NTHREADS = QH * RT * 32;
  constexpr float NEG_INF = -3.0e38f;

  threadgroup bfloat sK[BK * D];
  threadgroup bfloat sV[BK * D];
  threadgroup int tg_hi[QH * RT];

  const int head_off  = int(simd_gid) / RT;
  const int row_tile  = int(simd_gid) % RT;
  const int head_base = int(tid.x) * QH;
  const int q_head    = head_base + head_off;
  const int n_q_heads = int(tpg.x) * QH;
  // Uniform across the threadgroup: QH divides gqa_factor and head_base is a
  // multiple of QH, so every head here shares one KV head. The staged block
  // would otherwise be a different head's for part of the threadgroup.
  const int kv_head   = head_base / gqa_factor;
  const int tidx      = int(simd_gid) * 32 + int(simd_lid);

  const int row0 = int(tid.y) * BQ + row_tile * kU;
  const short rows_here = short(clamp(n_rows - row0, 0, kU));

  const short2 co = frag_coord(simd_lid);
  const short sm = co.y, sn = co.x;

  int q_pos[kElemRows];
#pragma clang loop unroll(full)
  for (short i = 0; i < kElemRows; ++i) {
    const short r = sm + i * kElemRowsJump;
    q_pos[i] = r < rows_here ? position_ids[row0 + r] : -1;
  }

  // The key loop bound must be uniform across the whole THREADGROUP now, not
  // just the simdgroup: staging is a threadgroup-wide cooperative act and a
  // simdgroup that stopped early would leave its peers waiting at a barrier
  // that never completes.
  int my_hi = -1;
  for (short i = 0; i < kElemRows; ++i) my_hi = max(my_hi, q_pos[i]);
  my_hi = simd_max(my_hi);
  if (simd_lid == 0) tg_hi[simd_gid] = my_hi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int kp_hi = -1;
  for (int s = 0; s < QH * RT; ++s) kp_hi = max(kp_hi, tg_hi[s]);

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

  for (int kb = 0; kb < n_blocks; ++kb) {
    const int page = int(kv_page_indices[kb]);
    const short klim = short(min(BK, kp_hi + 1 - kb * BK));

    // ── stage the block once, cooperatively ──
    // Keys past the cache are zeroed rather than skipped: a zero key still
    // scores 0 and exp(0) is 1, so the causal mask below is what actually
    // removes them. Zeroing only keeps the read in bounds.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tidx; e < BK * D; e += NTHREADS) {
      const int key = e / D, d = e % D;
      const size_t slot = size_t(page) * BK + size_t(key);
      const bool live = key < klim;
      sK[e] = live ? k_pages[(slot * n_kv_heads + kv_head) * D + d] : bfloat(0);
      sV[e] = live ? v_pages[(slot * n_kv_heads + kv_head) * D + d] : bfloat(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── S = Q · Kᵀ ──
    ffrag S[TK];
#pragma clang loop unroll(full)
    for (short i = 0; i < TK; ++i) S[i] = ffrag(0);
#pragma clang loop unroll(full)
    for (short id = 0; id < TD; ++id) {
      bfrag qf, kf0, kf1;
      frag_load(qf, Q + id * kU, q_ld, simd_lid);
      frag_load_tg(kf0, sK + id * kU, D, simd_lid);
      frag_load_tg(kf1, sK + kU * D + id * kU, D, simd_lid);
      frag_mma<true>(S[0], S[1], qf, kf0, kf1);
    }

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
        const threadgroup bfloat* Vk = sV + ik * kU * D;
        frag_load_tg(v0, Vk + d * kU, D, simd_lid);
        frag_load_tg(v1, Vk + (d + 1) * kU, D, simd_lid);
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
