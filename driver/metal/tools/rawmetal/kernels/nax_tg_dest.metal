// Isolate ONE new thing: can the matmul write its destination into THREADGROUP
// memory? The fused kernel introduced that and a 64x128 cooperative
// destination at the same time, and is 7780/8192 wrong; testing them together
// tells you nothing about which.
//
// Identical to the verified Q.K^T kernel except C is a threadgroup tensor,
// copied to device afterwards. If this is wrong, the threadgroup destination
// is the fault. If right, the fault is the cooperative O.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_tg_dest(
    device bfloat* qp [[buffer(0)]],
    device bfloat* kp [[buffer(1)]],
    device float* sp  [[buffer(2)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 64, BK = 32, D = 128;
  const uint lid = simd_gid * 32u + simd_lid;
  threadgroup float st[BQ * BK];

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(qp, dextents<int32_t, 2>(D, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(kp, dextents<int32_t, 2>(BK, D));
  tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> C(st, dextents<int32_t, 2>(BK, BQ));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, BK, D, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  op.run(A, B, C);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = lid; e < uint(BQ * BK); e += 128u) sp[e] = st[e];
}
