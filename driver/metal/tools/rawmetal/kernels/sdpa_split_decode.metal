// Split-K decode attention: the SAME work, spread over more threadgroups.
//
// ## Why, and why NOT for the usual reason
//
// Flash-decoding is normally motivated by KV traffic: splitting the key range
// lets a threadgroup cover a whole GQA group without the grid collapsing, so
// each KV head is read once instead of once per query head. **That is not the
// motivation here, and the traffic argument does not hold on this machine.**
// `sdpa_paged_probe`'s head-sharing sweep shows the decode kernel achieving
// 18-26% of the 296 GB/s roof on unique bytes, while the traffic it would need
// if the redundant GQA reads were NOT cache-served works out at 121-146% of the
// roof above 8k context -- impossible, so they are already cached.
//
// The kernel is LATENCY-bound. Its per-key chain is load, dot, cross-lane
// reduce, update the running max and sum, rescale the accumulator, accumulate V
// -- entirely serial, one memory latency deep per key. Unrolling the loads to
// overlap them was measured and REJECTED: 7% in isolation at 12k/16k and a 19%
// regression end to end at short context.
//
// What is left is more independent chains in flight. At QH=2 over 32 query
// heads the grid is 16 threadgroups; splitting the key range S ways makes it
// 16*S, and each one's chain is 1/S as long. That is the whole idea.
//
// ## The cost, and the reason this is two kernels
//
// Each split produces a PARTIAL softmax -- its own running max, its own sum,
// its own unnormalized accumulator. They cannot simply be added: the maxima
// differ. `sdpa_split_combine` rescales each partial to the global max before
// summing, which is the same merge the online softmax does within one loop,
// done once across splits at the end.
//
// Partials are float, not bfloat. They are an intermediate that gets rescaled
// and summed, so rounding them to bfloat here would lose precision the single
// kernel never loses.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef SPLIT_HEADS
#define SPLIT_HEADS 2
#endif
#ifndef NSPLIT
#define NSPLIT 4
#endif

// Partial layout: [q_head][split][D] for o, [q_head][split][2] for (max, sum).
kernel void sdpa_split_decode(
    const device bfloat* queries      [[buffer(0)]],
    const device bfloat* k_pages      [[buffer(1)]],
    const device bfloat* v_pages      [[buffer(2)]],
    device float* partial_o           [[buffer(3)]],
    device float* partial_ms          [[buffer(4)]],
    const constant int& gqa_factor    [[buffer(5)]],
    const device int* position_ids    [[buffer(6)]],
    const device uint* kv_page_indices[[buffer(7)]],
    const constant int& page_size     [[buffer(8)]],
    const constant int& n_kv_heads    [[buffer(9)]],
    const constant float& scale       [[buffer(10)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int D = 128, BN = 32, BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread  = D / BD;
  constexpr int QH = SPLIT_HEADS;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  const int group     = int(tid.x);
  const int split     = int(tid.y);
  const int head_base = group * QH;
  const int kv_head   = head_base / gqa_factor;

  threadgroup U red[BN * BD];
  threadgroup U tg_max[BN];
  threadgroup U tg_sum[BN];

  U q[QH][qk_per_thread], o[QH][v_per_thread];
  U row_max[QH], row_sum[QH];
  for (int h = 0; h < QH; ++h) {
    const device bfloat* qp =
        queries + size_t(head_base + h) * D + simd_lid * qk_per_thread;
    for (int j = 0; j < qk_per_thread; ++j) q[h][j] = U(scale) * U(qp[j]);
    for (int j = 0; j < v_per_thread; ++j) o[h][j] = 0;
    row_max[h] = NEG_INF;
    row_sum[h] = 0;
  }

  const int q_pos = position_ids[0];
  const int total = q_pos + 1;
  // Ceiling division, so the last split takes the short piece rather than
  // leaving keys uncovered -- an uncovered key is a silently wrong softmax.
  const int chunk = (total + NSPLIT - 1) / NSPLIT;
  const int lo = split * chunk;
  const int hi = min(total, lo + chunk);   // exclusive

  for (int kp = lo + int(simd_gid); kp < hi; kp += BN) {
    const int page = int(kv_page_indices[kp / page_size]);
    const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
    const device bfloat* kptr =
        k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
    const device bfloat* vptr =
        v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;
    U kv[qk_per_thread], vv[v_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) kv[j] = U(kptr[j]);
    for (int j = 0; j < v_per_thread; ++j) vv[j] = U(vptr[j]);

    for (int h = 0; h < QH; ++h) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; ++j) score += q[h][j] * kv[j];
      score = simd_sum(score);
      U factor, exp_score;
      sdpa_online_update(score, row_max[h], row_sum[h], factor, exp_score);
      for (int j = 0; j < v_per_thread; ++j) o[h][j] = o[h][j] * factor + exp_score * vv[j];
    }
  }

  // Cross-simdgroup reduction, then write this split's PARTIAL -- unnormalized,
  // with its own max and sum, for the combine to merge.
  for (int h = 0; h < QH; ++h) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) { tg_max[simd_gid] = row_max[h]; tg_sum[simd_gid] = row_sum[h]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U m = tg_max[simd_lid];
    U new_max = simd_max(m);
    U fac = fast::exp(m - new_max);
    U tot = simd_sum(tg_sum[simd_lid] * fac);
    const int qh = head_base + h;
    for (int i = 0; i < v_per_thread; ++i) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      red[simd_lid * BD + simd_gid] = o[h][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      U acc = simd_sum(red[simd_gid * BD + simd_lid] * fac);
      if (simd_lid == 0) {
        // NOT divided by `tot`: the sum is per split and the combine needs the
        // unnormalized accumulator to weight it against the others.
        partial_o[(size_t(qh) * NSPLIT + split) * D + simd_gid * v_per_thread + i] = acc;
      }
    }
    if (simd_lid == 0 && simd_gid == 0) {
      // A split with no keys at all (short context, many splits) must record a
      // sum of zero and a max of -inf so the combine skips it rather than
      // folding in an uninitialized accumulator.
      partial_ms[(size_t(qh) * NSPLIT + split) * 2 + 0] = (hi > lo) ? new_max : NEG_INF;
      partial_ms[(size_t(qh) * NSPLIT + split) * 2 + 1] = (hi > lo) ? tot : 0.0f;
    }
  }
}

/// Merge the per-split partials into the final attention output.
///
/// One threadgroup per query head, one lane per output dimension pair. The
/// merge is the online softmax's own rule applied across splits: rescale each
/// partial to the global max, sum the weights, divide once.
kernel void sdpa_split_combine(
    const device float* partial_o  [[buffer(0)]],
    const device float* partial_ms [[buffer(1)]],
    device bfloat* out             [[buffer(2)]],
    // All scalar: Metal refuses a signature that mixes `uint3` and `uint`
    // position attributes ("expecting input declarations with either all scalar
    // types or all vector types with the same number of elements").
    uint tid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
  constexpr int D = 128;
  constexpr float NEG_INF = -3.0e38f;
  const int qh = int(tid);

  float gmax = NEG_INF;
  for (int s = 0; s < NSPLIT; ++s)
    gmax = max(gmax, partial_ms[(size_t(qh) * NSPLIT + s) * 2 + 0]);

  float denom = 0;
  float w[NSPLIT];
  for (int s = 0; s < NSPLIT; ++s) {
    const float m = partial_ms[(size_t(qh) * NSPLIT + s) * 2 + 0];
    const float t = partial_ms[(size_t(qh) * NSPLIT + s) * 2 + 1];
    // An empty split has max -inf and sum 0; `exp(-inf - gmax)` is 0, so it
    // contributes nothing without a branch.
    w[s] = (t > 0.0f && m > NEG_INF) ? fast::exp(m - gmax) : 0.0f;
    denom += w[s] * t;
  }
  const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;

  for (uint d = lid; d < uint(D); d += nthreads) {
    float acc = 0;
    for (int s = 0; s < NSPLIT; ++s)
      acc += w[s] * partial_o[(size_t(qh) * NSPLIT + s) * D + d];
    out[size_t(qh) * D + d] = bfloat(acc * inv);
  }
}
