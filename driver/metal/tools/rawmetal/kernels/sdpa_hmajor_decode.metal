// The shipped decode attention, with ONE line changed: where a key sits inside
// a page.
//
// ## The hypothesis this exists to test
//
// KV pages are `[slot][kv_head][dim]`. A threadgroup serves one kv head, so it
// reads 256 B of every 1024 B: a 32-key page is 32 KB, of which one head uses
// 8 KB as **32 separate 256 B runs** rather than one 8 KB run. No bytes are
// wasted at cache-line granularity — every 256 B run is whole lines — but the
// stream is 32 small regions where it could be one large one.
//
// That matters because of what the LANES_PER_KEY sweep measured. Halving the
// lanes per key doubles the number of separate regions one load instruction
// touches, and it cost ~1.4x per doubling, monotonically, all the way from 1
// region to 32. The kernel is bound by memory latency and coalescing sets how
// many transactions it must wait on. The page layout is that same effect one
// level up, and it is worth 32 regions per page.
//
// ## The candidate
//
//     shipped      (page*32 + kp%32) * n_kv_heads * D + kv_head * D
//     head-major   ((page * n_kv_heads + kv_head) * 32 + kp%32) * D
//
// Same page size, same allocator, same page table, same bytes resident. Only
// the offset WITHIN a page changes, so a real landing is this one line in every
// kernel that walks a page plus `kv_append_paged` writing to the new offset.
//
// Everything else here is `sdpa_paged_decode_hshare` unchanged — same QH head
// sharing, same online softmax, same epilogue — so the measured difference is
// the layout and nothing else.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef HEADS
#define HEADS 2
#endif
// 0 = the shipped [slot][kv_head][dim]; 1 = [page][kv_head][slot][dim].
// Both compiled from ONE source so the comparison cannot pick up an unrelated
// difference between two files.
#ifndef HEAD_MAJOR
#define HEAD_MAJOR 1
#endif

kernel void sdpa_hmajor_decode(
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
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread  = V / BD;
  constexpr int QH = HEADS;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  const int group     = int(tid.x);
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

  const int r         = req_of_token[0];
  const int q_pos     = position_ids[0];
  const int page_base = int(kv_page_indptr[r]);

  for (int kp = int(simd_gid); kp <= q_pos; kp += BN) {
    const int page = int(kv_page_indices[page_base + (kp >> 5)]);
    // THE ONE LINE. Everything above and below is the shipped kernel.
#if HEAD_MAJOR
    const size_t base = (size_t(page) * n_kv_heads + kv_head) * 32 + size_t(kp & 31);
    const device bfloat* kptr = k_pages + base * D + simd_lid * qk_per_thread;
    const device bfloat* vptr = v_pages + base * D + simd_lid * v_per_thread;
#else
    const size_t slot = size_t(page) * 32 + size_t(kp & 31);
    const device bfloat* kptr =
        k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
    const device bfloat* vptr =
        v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;
#endif

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
