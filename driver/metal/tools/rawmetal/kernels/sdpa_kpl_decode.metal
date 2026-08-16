// Decode attention with a KEY PER LANE: the same arithmetic, 32x fewer
// reductions.
//
// ## What is actually being attacked
//
// The shipped decode kernel gives a KEY to the simdgroup. Thirty-two lanes each
// multiply `D/32 = 4` dimensions and a five-step `simd_sum` folds them into one
// score, which then feeds a dependent online-softmax update. So the serial
// chain per key is:
//
//     load -> 4 FMA -> 5-step shuffle reduce -> exp/max/sum update -> rescale
//
// At 7424 context each of the 32 simdgroups owns 232 keys, so it walks 232 of
// those chains end to end. **The reduction and the update cost more than the
// four multiply-adds they sit on**, which is the same observation
// `sdpa_paged_tiled_body`'s KEY_PER_LANE comment makes at d=64, and it is why
// unrolling the LOADS bought nothing: the loads were never the constraint.
//
// ## The shape
//
// Give the key to the LANE. Lane `l` owns key `base + l` and walks the whole
// 128-long dot product itself, so thirty-two keys are scored with **no
// reduction at all**. The softmax then needs one `simd_max` and one `simd_sum`
// per THIRTY-TWO keys instead of one of each per key:
//
//     per lane, ctx 7424       FMAs   reductions   dependent updates
//     shipped (key/simdgroup)   928          232                 232
//     this   (key/lane)         928          7.2                 7.2
//
// The multiply-adds are identical. Only the serial reduction and softmax chains
// shrink, and those are what a latency-bound kernel is waiting on.
//
// ## The three costs, and how each is paid
//
// **q.** A lane no longer owns a slice of q, it needs the whole row. It goes to
// threadgroup memory once per fire; every lane then reads the SAME address per
// dimension, which is a broadcast rather than 32 accesses.
//
// **The probabilities.** After scoring, accumulating V wants the original
// layout back -- there the lane owns a slice of V and the accumulation needs no
// reduction at all. So each lane's score has to reach every other lane.
// `simd_shuffle` does that in registers, which is why this needs no probability
// TILE and no barrier where the tiled kernel needs both.
//
// **K's access pattern.** Lane `l` reads key `base+l`, so at a given dimension
// the 32 lanes are 256 B apart -- a scatter where the shipped kernel is one
// coalesced 256 B line. The bytes and the cache lines touched are identical
// (each lane consumes all four 64 B lines of its own key); what changes is that
// they arrive as 32 independent transactions instead of one. For a kernel short
// of latency rather than bandwidth that is not obviously bad, and it is the
// main thing this probe is measuring.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef HEADS
#define HEADS 2
#endif

// The SHIPPED `bind::SdpaPaged` signature, not a prototype one, so
// `sdpa_paged_probe`'s split arm can time this with the same code that times
// the kernel it would replace -- warmed clocks, an interleaved baseline and a
// monotonicity guard, none of which a second bespoke harness would inherit.
kernel void sdpa_kpl_decode(
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
  constexpr int v_per_thread = V / BD;      // 4, the ORIGINAL layout for V
  constexpr int QH = HEADS;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  const int group     = int(tid.x);
  const int n_q_heads = int(tpg.x) * QH;
  const int head_base = group * QH;
  const int kv_head   = head_base / gqa_factor;

  // q for every head this threadgroup owns, whole rows, SCALED once here so the
  // inner loop is a bare multiply-add.
  threadgroup U qtile[QH][D];
  threadgroup U red[BN * BD];
  threadgroup U tg_max[BN];
  threadgroup U tg_sum[BN];

  {
    const uint lid = simd_gid * 32u + simd_lid;   // 0..1023
    for (uint i = lid; i < uint(QH * D); i += 1024u) {
      const int h = int(i) / D, d = int(i) % D;
      qtile[h][d] = U(scale) * U(queries[size_t(head_base + h) * D + d]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U o[QH][v_per_thread];
  U row_max[QH], row_sum[QH];
  for (int h = 0; h < QH; ++h) {
    for (int j = 0; j < v_per_thread; ++j) o[h][j] = 0;
    row_max[h] = NEG_INF;
    row_sum[h] = 0;
  }

  const int r         = req_of_token[0];
  const int q_pos     = position_ids[0];
  const int page_base = int(kv_page_indptr[r]);
  const int total     = q_pos + 1;

  // Each simdgroup takes a BLOCK of 32 keys; lane l owns key `base + l`. The
  // blocks stride by the whole threadgroup, so the 32 simdgroups cover 1024
  // keys per pass exactly as the shipped kernel covers 32.
  for (int base = int(simd_gid) * BN; base < total; base += BN * BN) {
    const int key = base + int(simd_lid);
    const bool live = key < total;

    // This lane's key, once, for both the score and the page walk below.
    const int kr   = live ? key : 0;
    const int page = int(kv_page_indices[page_base + (kr >> 5)]);
    const size_t slot = size_t(page) * 32 + size_t(kr & 31);
    const device bfloat* kp = k_pages + (slot * n_kv_heads + kv_head) * D;

    for (int h = 0; h < QH; ++h) {
      // THE WHOLE DOT PRODUCT, IN ONE LANE. No shuffle, no reduction.
      U score = 0;
#pragma clang loop unroll(full)
      for (int d = 0; d < D; ++d) score += qtile[h][d] * U(kp[d]);
      if (!live) score = NEG_INF;

      // One block max and one block sum for thirty-two keys, where the shipped
      // kernel does one of each per key.
      const U bmax = simd_max(score);
      const U new_max = max(row_max[h], bmax);
      const U factor = (row_max[h] == NEG_INF && new_max == NEG_INF)
                           ? U(1) : fast::exp(row_max[h] - new_max);
      const U p = (new_max == NEG_INF) ? U(0) : fast::exp(score - new_max);
      row_sum[h] = row_sum[h] * factor + simd_sum(p);
      row_max[h] = new_max;

      // Back to the ORIGINAL layout to accumulate V: the lane owns a slice, so
      // this needs no reduction either, and `simd_shuffle` moves each key's
      // probability to every lane without touching threadgroup memory.
      for (int j = 0; j < v_per_thread; ++j) o[h][j] *= factor;
      for (int u = 0; u < BN; ++u) {
        const U pu = simd_shuffle(p, uint(u));
        const int ku = base + u;
        if (ku >= total) break;          // uniform across the simdgroup
        const int pg = int(kv_page_indices[page_base + (ku >> 5)]);
        const size_t sl = size_t(pg) * 32 + size_t(ku & 31);
        const device bfloat* vp =
            v_pages + (sl * n_kv_heads + kv_head) * V + simd_lid * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) o[h][j] += pu * U(vp[j]);
      }
    }
  }

  // The same cross-simdgroup epilogue the shipped kernel has.
  for (int h = 0; h < QH; ++h) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) { tg_max[simd_gid] = row_max[h]; tg_sum[simd_gid] = row_sum[h]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U m = tg_max[simd_lid];
    U new_max = simd_max(m);
    U fac = (m == NEG_INF) ? U(0) : fast::exp(m - new_max);
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
