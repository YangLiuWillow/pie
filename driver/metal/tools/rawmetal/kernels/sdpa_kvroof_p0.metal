// What bandwidth can a decode's KV read ACTUALLY reach?
//
// ## Why the streaming roof is the wrong number to compare against
//
// Decode attention is priced at 5.05 ms against a 2.47 ms "roofline", which is
// the unique KV bytes at this machine's 296 GB/s streaming roof. That 2.05x gap
// is the largest headroom in a decode step and it has survived five separate
// attacks -- more parallelism (split-K, 1.06x in situ), fewer reductions (the
// LANES_PER_KEY sweep, every variant slower), overlapped loads (the unroll,
// ~0 end to end), and a head-major page layout (1.03x). Nothing that should
// help a bandwidth-bound kernel has moved it.
//
// At some point the honest question stops being "what else can the kernel do"
// and becomes "is 296 GB/s reachable AT ALL by this access pattern". A decode
// reads its KV as a PAGED GATHER: a page-table lookup per 32 keys, then 256 B
// from a page that may be anywhere, with one kv head's keys 1 KB apart. That is
// not a stream, and a stream's roof may simply not apply to it.
//
// ## What this measures
//
// The shipped head-sharing decode kernel with everything but the LOADS removed.
// Same grid, same threadgroup shape, same page walk, same strides, same bytes
// -- no dot product, no softmax, no reduction. Whatever this reaches is the
// ceiling the real kernel is working under, and the gap between it and 296 GB/s
// is how much of the "headroom" was never available.
//
// PAGED=0 reads the same bytes from one contiguous run instead, with the page
// table untouched. The difference between the two arms is the INDIRECTION and
// the stride; the difference between PAGED=1 and the streaming roof is what a
// gather costs on this machine.
//
// The accumulator is written to `out` so nothing here can be optimised away: a
// load-only kernel whose result is unused is a kernel that does not load.

#include <metal_stdlib>
using namespace metal;

#ifndef HEADS
#define HEADS 2
#endif
#ifndef PAGED
#define PAGED 0
#endif

kernel void sdpa_kvroof_decode(
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
  (void)attention_mask_enabled; (void)window; (void)sinks; (void)queries;
  (void)scale;
  constexpr int D = 128, V = 128, BN = 32, BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread  = V / BD;
  constexpr int QH = HEADS;

  typedef float U;
  const int head_base = int(tid.x) * QH;
  const int kv_head   = head_base / gqa_factor;

  const int r         = req_of_token[0];
  const int q_pos     = position_ids[0];
  const int page_base = int(kv_page_indptr[r]);

  U acc[v_per_thread];
  for (int j = 0; j < v_per_thread; ++j) acc[j] = 0;

  // EXACTLY the shipped kernel's loop, minus the arithmetic. Same stride, same
  // page lookup per 32 keys, same 256 B per key per head.
  for (int kp = int(simd_gid); kp <= q_pos; kp += BN) {
#if PAGED
    const int page = int(kv_page_indices[page_base + (kp >> 5)]);
    const size_t slot = size_t(page) * 32 + size_t(kp & 31);
#else
    // Same bytes, same order, NO page table: `kp` addresses the run directly.
    // The page indices are still bound and still ignored, so the only thing
    // that changed is where the address comes from.
    const size_t slot = size_t(kp);
#endif
    const device bfloat* kptr =
        k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
    const device bfloat* vptr =
        v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;
    for (int j = 0; j < qk_per_thread; ++j) acc[j] += U(kptr[j]);
    for (int j = 0; j < v_per_thread; ++j) acc[j] += U(vptr[j]);
  }

  // Written, so the loads cannot be dead-code eliminated. One lane per slice,
  // which is the shipped kernel's own output shape.
  if (simd_gid == 0) {
    device bfloat* op = out + size_t(head_base) * V + simd_lid * v_per_thread;
    for (int j = 0; j < v_per_thread; ++j) op[j] = bfloat(acc[j]);
  }
}
