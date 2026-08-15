// The cooperative tensor's layout does not need deriving OR calibrating: the
// API answers it. `get_multidimensional_index(i)` returns the (row, col) of a
// lane's i-th element, and `get_capacity()` says how many there are.
//
// An evening went into inferring this mapping from the output of a full matmul
// -- 128 of 2048 wrong, a refuted hypothesis, a lane-bit derivation -- and it
// was queryable the whole time. The lesson is not about Metal: when a layout
// is opaque, look for the accessor before reaching for the microscope.
//
// This matters beyond tidiness. Flash attention rescales O by `factor` per ROW
// on every key block, and O at 64x128 floats is 32 KB -- too big for
// threadgroup memory, so with a memory-backed C it is 32 KB of device traffic
// per block. A cooperative tensor puts O in registers, and per-row rescaling
// needs exactly this index.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_calibrate(
    device bfloat* ap [[buffer(0)]],
    device bfloat* bp [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int M = 64, N = 64, K = 128;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(ap, dextents<int32_t, 2>(K, M));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(bp, dextents<int32_t, 2>(N, K));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      M, N, K, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    // The template takes the OPERAND TENSOR types, not element types.
  auto ct_c = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  for (ushort i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;
  op.run(A, B, ct_c);

  const uint lane = simd_gid * 32u + simd_lid;
  out[lane * 128u] = float(ct_c.get_capacity());
  for (ushort i = 0; i < ct_c.get_capacity() && i < 40; ++i) {
    auto mi = ct_c.get_multidimensional_index(i);
    out[lane * 128u + 1u + i * 2u]      = float(mi[0]);
    out[lane * 128u + 1u + i * 2u + 1u] = float(mi[1]);
  }
}
