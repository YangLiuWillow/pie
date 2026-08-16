// The rejected key-loop unroll, gated on context INSIDE the kernel.
//
// ## Why this is worth a second look
//
// Unrolling the key loop -- issuing U keys' loads before any of their
// arithmetic, so the loads overlap -- was measured at **+7% at 12k and 16k** and
// **-19% end to end at 5,840 tokens**, and was discarded whole. The reason it
// could not be gated is recorded plainly: `pso_for` and `launch_shape` are
// handed the geometry, the row count and the request count, and NOT the context
// length, so "unroll only above 8k" is not a question either site can ask.
//
// **The kernel can ask it.** `total = position_ids[0] + 1` is right there, it is
// uniform across the threadgroup, and a branch on it is close to free.
//
// ## The reason this might still not work
//
// A branch does not undo register pressure. The compiler allocates for the
// WORST path, so a kernel containing an unrolled loop may spill on every fire
// whichever branch it takes -- and spilling is the most likely explanation for
// a 19% end-to-end regression from a change that only touched long-context
// behaviour. If that is what happened, the short-context arm of this kernel
// will be slower than the shipped one even though it runs the same code, and
// the idea is dead for a reason worth knowing.
//
// So the measurement that matters is NOT the long-context speedup. It is
// whether the 2k column still matches the shipped kernel.

#include <metal_stdlib>
using namespace metal;
#include "../../../src/kernels/sdpa_online.h"

#ifndef HEADS
#define HEADS 2
#endif
#ifndef UNROLL
#define UNROLL 4
#endif
// Contexts at or above this take the unrolled path. 0 forces the plain loop
// always, which makes this kernel a control for its own register pressure:
// same code, same allocation, unroll never taken.
#ifndef UGATE
#define UGATE 0
#endif

kernel void sdpa_ugate_decode(
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
  const int head_base = int(tid.x) * QH;
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
  const int total     = q_pos + 1;

  // THE GATE. Uniform across the threadgroup, so this is one branch, not
  // divergence -- and it is a question the host could not have asked.
  const bool wide = (UGATE > 0) && (total >= UGATE);

  if (wide) {
    for (int kp0 = int(simd_gid); kp0 <= q_pos; kp0 += BN * UNROLL) {
      U kv[UNROLL][qk_per_thread], vv[UNROLL][v_per_thread];
      bool live[UNROLL];
#pragma clang loop unroll(full)
      for (int u = 0; u < UNROLL; ++u) {
        const int kp = kp0 + u * BN;
        live[u] = kp <= q_pos;
        const int kpr = live[u] ? kp : 0;
        const int page = int(kv_page_indices[page_base + (kpr >> 5)]);
        const size_t slot = size_t(page) * 32 + size_t(kpr & 31);
        const device bfloat* kptr =
            k_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * qk_per_thread;
        const device bfloat* vptr =
            v_pages + (slot * n_kv_heads + kv_head) * D + simd_lid * v_per_thread;
        for (int j = 0; j < qk_per_thread; ++j) kv[u][j] = U(kptr[j]);
        for (int j = 0; j < v_per_thread; ++j) vv[u][j] = U(vptr[j]);
      }
#pragma clang loop unroll(full)
      for (int u = 0; u < UNROLL; ++u) {
        for (int h = 0; h < QH; ++h) {
          U score = 0;
          for (int j = 0; j < qk_per_thread; ++j) score += q[h][j] * kv[u][j];
          score = simd_sum(score);
          if (!live[u]) continue;
          U factor, exp_score;
          sdpa_online_update(score, row_max[h], row_sum[h], factor, exp_score);
          for (int j = 0; j < v_per_thread; ++j)
            o[h][j] = o[h][j] * factor + exp_score * vv[u][j];
        }
      }
    }
  } else {
    for (int kp = int(simd_gid); kp <= q_pos; kp += BN) {
      const int page = int(kv_page_indices[page_base + (kp >> 5)]);
      const size_t slot = size_t(page) * 32 + size_t(kp & 31);
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
