// Fused flash attention on the slice API: Q.K^T, online softmax, O += P.V,
// with O held in REGISTERS across key blocks.
//
// The design is forced by one line of the header: "A and B can be
// tensor_handle, tensor_offset, and tensor_inline. C can be ... or
// cooperative_tensor." Operands must be memory-backed; only the destination
// may live in registers. So:
//
//   S = Q.K^T          -> C is a THREADGROUP tensor (S must become an operand)
//   softmax            -> plain threadgroup work, no lane algebra needed
//   O += P.V           -> C is a COOPERATIVE tensor, surviving all key blocks
//
// O is the one that must be in registers: it is 64x128 floats = 32 KB, so a
// memory-backed O would cost 32 KB of device traffic on every block just for
// the per-row rescale. The rescale needs each element's row, which
// `get_multidimensional_index(i)[0]` answers directly -- the accessor that an
// evening of lane-bit derivation failed to find.
//
// Threadgroup budget: S 64x64 float = 16 KB, P 64x64 bf16 = 8 KB, row stats
// 512 B. 24.5 KB against the measured 32 KB cap.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// WHICH INDEX IS THE ROW? `get_multidimensional_index` returns a pair and the
// header does not say which way round it is. The 64x64 calibration printed a
// self-consistent pattern either way, so this was assumed rather than checked
// -- and with a 64x128 destination, guessing wrong scrambles everything.
#ifndef NAX_ROW
#define NAX_ROW 0
#define NAX_COL 1
#endif

kernel void nax_flash(
    device bfloat* qp        [[buffer(0)]],   // [BQ][D]
    device bfloat* kp        [[buffer(1)]],   // [D][ctx] K transposed
    device bfloat* vp        [[buffer(2)]],   // [ctx][D]
    device float* outp       [[buffer(3)]],   // [BQ][D]
    const constant int& ctx  [[buffer(4)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 64, BK = 64, D = 128;
  const uint lid = simd_gid * 32u + simd_lid;

  threadgroup float  s_tile[BQ * BK];
  threadgroup bfloat p_tile[BQ * BK];
  threadgroup float  row_max[BQ];
  threadgroup float  row_sum[BQ];
  threadgroup float  row_fac[BQ];

  for (uint r = lid; r < uint(BQ); r += 128u) {
    row_max[r] = -3.0e38f;
    row_sum[r] = 0.0f;
  }

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Q(qp, dextents<int32_t, 2>(D, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> K(kp, dextents<int32_t, 2>(ctx, D));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> V(vp, dextents<int32_t, 2>(D, ctx));
  tensor<threadgroup float,  dextents<int32_t, 2>, tensor_inline> S(s_tile, dextents<int32_t, 2>(BK, BQ));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> P(p_tile, dextents<int32_t, 2>(BK, BQ));

  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, BK, D, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);   // overwrite, not accumulate
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroups<4>> qk_op;

  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, D, BK, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroups<4>> pv_op;

  auto ct_o = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(V), float>();
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) ct_o[i] = 0.0f;

  for (int kb = 0; kb + BK <= ctx; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto Kb = K.slice(kb, 0);
    qk_op.run(Q, Kb, S);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, one row per thread. No lane algebra: S is in memory.
    for (uint r = lid; r < uint(BQ); r += 128u) {
      float m = row_max[r];
      for (int c = 0; c < BK; ++c) m = max(m, s_tile[r * BK + c]);
      const float fac = row_max[r] == -3.0e38f ? 0.0f : fast::exp(row_max[r] - m);
      float sum = 0.0f;
      for (int c = 0; c < BK; ++c) {
        const float p = fast::exp(s_tile[r * BK + c] - m);
        p_tile[r * BK + c] = bfloat(p);
        sum += p;
      }
      row_max[r] = m;
      row_sum[r] = row_sum[r] * fac + sum;
      row_fac[r] = fac;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rescale O by this block's factor, per ROW -- the reason O is a
    // cooperative tensor and the reason its row index had to be knowable.
    for (ushort i = 0; i < ct_o.get_capacity(); ++i) {
      const auto mi = ct_o.get_multidimensional_index(i);
      ct_o[i] *= row_fac[mi[NAX_ROW]];
    }

    auto Vb = V.slice(0, kb);
    pv_op.run(P, Vb, ct_o);
  }

  // Normalise and store.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) {
    const auto mi = ct_o.get_multidimensional_index(i);
    const float s = row_sum[mi[NAX_ROW]];
    outp[uint(mi[NAX_ROW]) * uint(D) + uint(mi[NAX_COL])] = s == 0.0f ? 0.0f : ct_o[i] / s;
  }
}
