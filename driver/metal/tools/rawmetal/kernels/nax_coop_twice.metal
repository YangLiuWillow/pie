// Does a cooperative tensor survive across two run() calls?
//
// The fused kernel is 7780/8192 wrong and the threadgroup destination was
// cleared by isolation, so the fault is O-as-cooperative-tensor. This removes
// everything else: same P, same V, two accumulating run() calls, no softmax,
// no rescale, no second matmul shape. The answer should be exactly 2*(P.V).
//
// If wrong, a cooperative tensor does not carry across calls the way the fused
// kernel assumes -- MLX re-derives one per call and never loops over it.
// If right, the fault is in the softmax path and this narrows it again.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_coop_twice(
    device bfloat* pp [[buffer(0)]],   // [BQ][BK]
    device bfloat* vp [[buffer(1)]],   // [BK][D]
    device float* op_ [[buffer(2)]],   // [BQ][D]
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 64, BK = 64, D = 128;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> P(pp, dextents<int32_t, 2>(BK, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> V(vp, dextents<int32_t, 2>(D, BK));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, D, BK, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  auto ct_o = op.get_destination_cooperative_tensor<decltype(P), decltype(V), float>();
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) ct_o[i] = 0.0f;

  op.run(P, V, ct_o);
  op.run(P, V, ct_o);        // the whole question, in one line

  // Store through the queried index. Bound-checked: an out-of-range index here
  // is a device hang, not a wrong number -- that is measured, not cautious.
  for (ushort i = 0; i < ct_o.get_capacity(); ++i) {
    const auto mi = ct_o.get_multidimensional_index(i);
    if (mi[0] < BQ && mi[1] < D) op_[uint(mi[0]) * uint(D) + uint(mi[1])] = ct_o[i];
  }
}
