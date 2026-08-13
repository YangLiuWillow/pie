// Paged prefill SDPA on the simdgroup matrix unit.
//
// WHY THIS FILE EXISTS. `sdpa_paged_tiled` gives a query ROW to a simdgroup:
// thirty-two rows, thirty-two simdgroups, and every score is a dot product the
// lanes walk by hand -- either thirty-two lanes splitting one key behind a
// shuffle reduction, or (KEY_PER_LANE) one key per lane walking all of D. Both
// are scalar ALU. On a 2048-token gpt-oss prefill that kernel is 35.8% of the
// fire, the second largest term after the quantized GEMM, and it runs at
// roughly 0.5 TFLOP/s against the ~5.6 the GEMM beside it reaches. A tenth of
// the machine, spent on a third of the fire.
//
// The arithmetic was always a matmul -- Q Kᵀ then P V -- and was never issued
// as one. This file issues it as one. The row tile stays 32 so the dispatch
// grid is unchanged; what changes is that a simdgroup now owns EIGHT rows and
// multiplies 8x8 fragments instead of owning one row and adding scalars.
//
// THE ONE THING THIS FILE DEPENDS ON, and the reason it is short: the register
// layout of `simdgroup_matrix<T,8,8>`. `thread_elements()` hands a lane two
// elements, and their coordinates are
//
//     qid = lane / 4;  fm = (qid & 4) + ((lane / 2) % 4);
//     fn  = (qid & 2) * 2 + (lane % 2) * 2;        // owns (fm, fn), (fm, fn+1)
//
// which is `BaseMMAFrag<T,8,8>::get_coord` in quantized_qmm_t.metal, the steel
// fragment this driver's GEMM already trusts. Two consequences carry the whole
// design:
//
//   * A lane's two elements are always in the SAME ROW, and that row is fixed
//     for the lane across every fragment it touches. So a lane's slice of the
//     score tile is a slice of ONE query row, and the online softmax -- row max,
//     row sum, the rescale factor -- is per-lane state, never a threadgroup
//     round trip. Flash-attention implementations that do not know the layout
//     must store S to threadgroup memory, reduce, and load it back every pass.
//     This one does not store S at all.
//
//   * The four lanes sharing a row are {l, l^1, l^8, l^9}: fm depends on
//     (lane/2)%4 and qid&4, both invariant under xor by 1 and by 8. Two
//     `simd_shuffle_xor` steps are the entire row reduction, against the five a
//     `simd_sum` would spend.
//
// WHAT IS NOT BRANCHED, and it matters. Every simdgroup lane must reach every
// `simdgroup_multiply_accumulate`: the instruction is a simdgroup-wide op and
// executing it under a divergent condition is undefined. The row max, the
// rescale factor and the masks are all PER-ROW, hence per-lane, hence
// divergent -- so none of them may gate an MMA. Instead the masked scores are
// driven to -inf, their probabilities come out exactly 0, and the multiply
// runs unconditionally on a tile of zeros. The rescale factor makes this work
// for free in two other places as well:
//
//     factor = (max_score == -inf) ? 0 : exp(max_score - new_max)
//
//   * A pass in which every key of a row is masked leaves new_max == max_score,
//     so factor == 1 and the row's accumulator is untouched. This is what lets
//     a tile that straddles a request boundary stage one run at a time without
//     the rows of the OTHER run losing what they have already accumulated.
//   * A row that has not yet seen a live key has max_score == -inf and an
//     accumulator of zeros, so factor == 0 scales nothing.
//
// KT is per width because the staged tiles are what bound it: at D=64 three
// tiles of KT=64 are 20 KB, and at D=128 the same shape would be 40 KB, over
// the 32 KB a threadgroup gets. The wide head halves KT instead, which doubles
// the staging barriers and is still far ahead of the scalar path.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#include "sdpa_online.h"

template <typename T, int D, int KT, bool WITH_SINK>
[[kernel]] [[max_total_threads_per_threadgroup(128)]] void sdpa_paged_mma(
    const device T* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device T* k_pages     [[buffer(1)]],   // [num_pages, page_size, n_kv_heads, D]
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],   // [N, n_q_heads, D]
    const constant int& gqa_factor             [[buffer(4)]],
    const device int* position_ids             [[buffer(5)]],
    const device int* req_of_token             [[buffer(6)]],
    const device uint* kv_page_indices         [[buffer(7)]],
    const device uint* kv_page_indptr          [[buffer(8)]],
    const constant int& page_size              [[buffer(9)]],
    const constant int& n_kv_heads             [[buffer(10)]],
    const constant float& scale                [[buffer(11)]],
    const device uchar* attention_mask         [[buffer(12)]],
    const device uint& attention_mask_stride   [[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // WITH_SINK only
    const constant int& n_rows                 [[buffer(17)]],  // N; the grid rounds up
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int QT  = 32;        // query rows per threadgroup: the grid's tile
  constexpr int SGS = 4;         // simdgroups; 128 threads
  constexpr int RPS = QT / SGS;  // 8 rows per simdgroup == one fragment row
  constexpr int DF  = D / 8;     // fragments across the head
  constexpr int KF  = KT / 8;    // fragments across a staged key pass
  constexpr float NEG_INF = -3.0e38f;

  static_assert(D % 8 == 0 && KT % 8 == 0, "the matrix unit tiles in eights");
  static_assert(RPS == 8, "a simdgroup owns exactly one fragment row of queries");

  threadgroup half qtile[QT * D];
  threadgroup half ktile[D * KT];  // Kᵀ: [dim][key], written transposed
  threadgroup half vtile[KT * D];  // V:  [key][dim]

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  const int row_lo    = int(tid.y) * QT;
  const uint lid      = simd_gid * 32u + simd_lid;

  // This lane's two elements of every fragment it will touch. `fm` is its row
  // and never changes; `fn` is the first of its two columns.
  const short qid = short(simd_lid) / 4;
  const short fm  = (qid & 4) + ((short(simd_lid) / 2) % 4);
  const short fn  = (qid & 2) * 2 + (short(simd_lid) % 2) * 2;

  const int my_row = row_lo + int(simd_gid) * RPS + int(fm);
  const bool live  = my_row < n_rows;

  // Queries, staged once and scaled once. Reading them straight from device
  // memory into the fragments was tried -- the layout allows it, since a lane's
  // elements are all in the row it already owns -- and it LOSES: 818.7 -> 799.7
  // tok/s at 1024 rows. The tile is 4 KB of the occupancy budget, but the read
  // that fills it is contiguous per thread, where the direct one has each lane
  // strided a whole q_head apart.
  //
  // The scale rides q rather than the score: one rounding, applied where the
  // value is already being converted, and it keeps the half fragments the
  // matrix unit multiplies inside a comfortable exponent range.
  for (uint e = lid; e < uint(QT * D); e += 128u) {
    const int r  = int(e) / D;
    const int d  = int(e) - r * D;
    const int gr = row_lo + r;
    qtile[e] = gr < n_rows
                   ? half(float(queries[(size_t(gr) * n_q_heads + q_head) * D + d]) * scale)
                   : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_matrix<half, 8, 8> Qf[DF];
  for (int i = 0; i < DF; i++) {
    simdgroup_load(Qf[i], qtile, D, ulong2(uint(i * 8), uint(simd_gid) * RPS), false);
  }

  // Per-row bounds. Every one of these is read for THIS lane's row only, which
  // is what makes them cheap: a scalar load, not a broadcast.
  const int q_pos    = live ? position_ids[my_row] : 0;
  const int my_start = (window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0;
  const bool masked  = live && attention_mask_enabled[my_row] != 0;
  const int my_req   = live ? req_of_token[my_row] : -1;

  float Ov[DF * 2];
  for (int i = 0; i < DF * 2; i++) Ov[i] = 0.0f;
  float max_score = NEG_INF;
  float sum_exp = 0.0f;

  // Runs of equal request inside the tile, threadgroup-uniform so the staging
  // barriers stay aligned. A prefill is one run; the loop is here so a tile that
  // straddles a boundary is correct rather than forbidden.
  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int r = req_of_token[row_lo + sub];
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows && req_of_token[row_lo + sub_hi] == r)
      sub_hi++;

    int kp_hi = 0;
    int kp_lo = 0x7fffffff;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
    }
    const int page_base = int(kv_page_indptr[r]);
    const bool mine = live && my_req == r;

    for (int base = kp_lo; base <= kp_hi; base += KT) {
      const int cnt = min(KT, kp_hi + 1 - base);
      // Before the writes as well as after: the previous pass's multiplies must
      // be done with the tile this one overwrites.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lid; e < uint(KT * D); e += 128u) {
        const int kk = int(e) / D;
        const int d  = int(e) - kk * D;
        if (kk < cnt) {
          const int kp = base + kk;
          const int page = int(kv_page_indices[page_base + kp / page_size]);
          const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
          const size_t off = (slot * n_kv_heads + kv_head) * D + d;
          ktile[d * KT + kk] = half(float(k_pages[off]));
          vtile[e] = half(float(v_pages[off]));
        } else {
          // Padding rows of the tile. Zeroed rather than skipped because the
          // multiply below reads the whole tile unconditionally; their scores
          // are masked to -inf anyway, this only keeps them finite.
          ktile[d * KT + kk] = half(0);
          vtile[e] = half(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ── S = (q·scale) Kᵀ ──
      //
      // Chunked at DCH fragments, and that is a PRECISION bound rather than a
      // register one. `simdgroup_multiply_accumulate` carries its accumulator
      // in the fragment type, so a score is a chain of DF half roundings --
      // one per fragment down the head. At d=64 that chain is 8 long and the
      // numerics test passes on it. Widening to d=128 doubled it, and the
      // dense attention case's worst rel_l2 went 0.0281 -> 0.0992 while a
      // borderline mixture case flipped from pass to fail outright: the extra
      // error was enough to change which experts a row routed to.
      //
      // So the chain is capped and the partial sums are folded into float
      // between chunks. At d=64 the cap is the whole head -- one chunk, the
      // arithmetic gpt-oss shipped with, bit for bit, which is the point of
      // making this width-dependent rather than uniform.
      //
      // At d=128 the cap is 4, and it does not have to be argued from accuracy
      // alone because it is also the fastest. Worst rel_l2 in
      // `llama_numerics_test` (scalar path: 0.0281 attention, 0.1145 mixture,
      // tolerance 0.12), and marginal prefill on Qwen3-0.6B:
      //
      //     DCH          2        4        8       16
      //     attention    0.0249   0.0251   0.0237  0.0992
      //     mixture      0.1427   0.0724   0.1619  0.1910
      //     tok/s                 3531             2593
      //
      // The 36% between 4 and 16 is the surprise, and it is why this is not a
      // precision tax to be minimised: a shorter chain is FASTER here as well
      // as more accurate. A 16-deep chain keeps a fragment accumulator live
      // across the whole head where a 4-deep one does not, and at d=128 that
      // is the difference between fitting the register file and not.
      //
      // Read the accuracy columns with care. Among 2, 4 and 8 the attention
      // numbers differ by less than a thousandth in no consistent direction --
      // noise, not a trend. The mixture column swings by a factor of two and is
      // NOT monotone: 2 is strictly more accurate arithmetic than 4 and scores
      // worse. A metric that moves the wrong way when the arithmetic improves
      // is not measuring the arithmetic; it is a routed model amplifying a tiny
      // input change into a different expert. So the mixture number is not
      // evidence for 4 over 8, and tuning against it would be fitting to that
      // amplifier. The throughput is what settles it, and 4 also happens to
      // leave the whole suite at parity with the scalar baseline.
      constexpr int DCH = D <= 64 ? 8 : 4;
      float sv[KF * 2];
      for (int c = 0; c < KF * 2; c++) sv[c] = 0.0f;
      for (int i0 = 0; i0 < DF; i0 += DCH) {
        const int i1 = i0 + DCH < DF ? i0 + DCH : DF;
        simdgroup_matrix<half, 8, 8> S[KF];
        for (int c = 0; c < KF; c++) S[c] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (int i = i0; i < i1; i++) {
          for (int c = 0; c < KF; c++) {
            // ktile IS Kᵀ -- the staging wrote it transposed -- so this is an
            // ordinary load. Transposing here instead would put the whole result
            // on `simdgroup_load`'s transpose argument, whose orientation is not
            // something to guess at when the staging can settle it for free.
            simdgroup_matrix<half, 8, 8> Bk;
            simdgroup_load(Bk, ktile, KT, ulong2(uint(c * 8), uint(i * 8)), false);
            simdgroup_multiply_accumulate(S[c], Qf[i], Bk, S[c]);
          }
        }
        for (int c = 0; c < KF; c++) {
          thread auto& e = S[c].thread_elements();
          sv[c * 2 + 0] += float(e[0]);
          sv[c * 2 + 1] += float(e[1]);
        }
      }

      // ── mask, row max, probabilities, row sum ──
      //
      // The masked score has to be driven to -inf AFTER the chunks are joined,
      // not inside them: a chunk holds a partial dot product, and there is no
      // value to reject until the last one has landed.
      float lmax = NEG_INF;
      for (int c = 0; c < KF; c++) {
        for (int j = 0; j < 2; j++) {
          const int kk = c * 8 + int(fn) + j;
          bool keep = mine && kk < cnt;
          if (keep) {
            const int kp = base + kk;
            keep = kp <= q_pos && kp >= my_start;
            if (keep && masked) {
              keep = !(uint(kp) >= attention_mask_stride ||
                       attention_mask[size_t(my_row) * attention_mask_stride + uint(kp)] == 0);
            }
          }
          const float s = keep ? sv[c * 2 + j] : NEG_INF;
          sv[c * 2 + j] = s;
          lmax = s > lmax ? s : lmax;
        }
      }
      // The four lanes of a row are {l, l^1, l^8, l^9}.
      lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
      lmax = max(lmax, simd_shuffle_xor(lmax, 8u));

      const float new_max = max(max_score, lmax);
      const float factor = max_score == NEG_INF ? 0.0f : fast::exp(max_score - new_max);
      float lsum = 0.0f;
      // P is its own array now that the score accumulator lives inside the
      // chunk loop. Same lane layout -- `thread_elements()` hands this lane the
      // two columns of `fm` it has owned throughout -- so writing the
      // probabilities straight in is the same store that used to overwrite S.
      simdgroup_matrix<half, 8, 8> P[KF];
      for (int c = 0; c < KF; c++) {
        thread auto& e = P[c].thread_elements();
        for (int j = 0; j < 2; j++) {
          const float p = sv[c * 2 + j] == NEG_INF ? 0.0f : fast::exp(sv[c * 2 + j] - new_max);
          e[j] = half(p);
          lsum += p;
        }
      }
      lsum += simd_shuffle_xor(lsum, 1u);
      lsum += simd_shuffle_xor(lsum, 8u);

      max_score = new_max;
      sum_exp = sum_exp * factor + lsum;
      for (int i = 0; i < DF * 2; i++) Ov[i] *= factor;

      // ── O += P V ──
      simdgroup_matrix<half, 8, 8> PV[DF];
      for (int n = 0; n < DF; n++) PV[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
      for (int c = 0; c < KF; c++) {
        for (int n = 0; n < DF; n++) {
          simdgroup_matrix<half, 8, 8> Bv;
          simdgroup_load(Bv, vtile, D, ulong2(uint(n * 8), uint(c * 8)), false);
          simdgroup_multiply_accumulate(PV[n], P[c], Bv, PV[n]);
        }
      }
      // The accumulator is float and the fragments are half: a pass accumulates
      // at most KT terms inside the matrix unit, but a row accumulates the whole
      // sequence, and that sum does not belong in a half.
      for (int n = 0; n < DF; n++) {
        thread auto& e = PV[n].thread_elements();
        Ov[n * 2 + 0] += float(e[0]);
        Ov[n * 2 + 1] += float(e[1]);
      }
    }
    sub = sub_hi;
  }

  if (!live) return;

  float orescale = 1.0f;
  if (WITH_SINK) {
    orescale = sdpa_merge_sink(float(sinks[q_head]), max_score, sum_exp);
  }
  device T* op = out + (size_t(my_row) * n_q_heads + q_head) * D;
  for (int n = 0; n < DF; n++) {
    for (int j = 0; j < 2; j++) {
      const float x = Ov[n * 2 + j] * orescale;
      op[n * 8 + int(fn) + j] = static_cast<T>(sum_exp == 0.0f ? x : x / sum_exp);
    }
  }
}

#define instantiate_sdpa_paged_mma(sfx, name, itype, d, kt, sink)            \
  template [[host_name("sdpa_paged_mma" sfx "_" #name "_d_" #d)]]            \
  [[kernel]] void sdpa_paged_mma<itype, d, kt, sink>(                        \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*, const constant int&,         \
      uint3, uint3, uint, uint);

// KT is what the 32 KB of threadgroup memory allows. The three staged tiles are
// `QT*D` (queries) + `D*KT` (Kᵀ) + `KT*D` (V) halves, so at KT=16 the cost is
// linear in the head width and the cap is a long way off until the head is very
// wide:
//
//     d=64   8 KB      d=128  16 KB      d=256  32 KB  <- exactly the cap
//
// d=256 is therefore NOT a matter of picking a KT -- at KT=16 it already spends
// the entire allowance on staging, and the register file is the second wall
// (`Ov[D/4]` floats is 64 registers a lane before Q and PV). It stays on the
// scalar path until someone restructures the accumulator for it.
//
// `sdpa_mma_head_dim_supported` is the list, and it must agree with this file:
// a width named there and not instantiated here fails to build a pipeline at
// load, by name, which is the good outcome.
instantiate_sdpa_paged_mma("", bfloat16, bfloat, 64, 16, false)      // llama / qwen d=64
instantiate_sdpa_paged_mma("_sink", bfloat16, bfloat, 64, 16, true)  // gpt-oss
// llama, mistral, qwen2, qwen3 and the Qwen MoEs -- the width this driver's
// dense families overwhelmingly use, and the one the prefill profile was taken
// on. KT stays 16 rather than the 32 that also fits: 16 KB of threadgroup
// memory leaves room for a second resident threadgroup where 24 KB does not,
// and the deeper pass buys only half the staging barriers.
instantiate_sdpa_paged_mma("", bfloat16, bfloat, 128, 16, false)
