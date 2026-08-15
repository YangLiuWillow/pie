// O = P.V on the slice API. The other matmul, and NOT symmetric with Q.K^T:
// V is [key][dim] where K is used transposed, so the access pattern differs and
// there is no reason to assume the rate carries over. That assumption is what
// the ~1.4 ms projection rests on, so it gets measured rather than inherited.
//
//   M = BQ (64 query rows), N = D (128 head dims), K = BK (64 keys)
//
// N is 128 here against 64 in Q.K^T -- a wider output tile, which is its own
// reason the rate may differ.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_slice_pv(
    device bfloat* pp        [[buffer(0)]],   // [BQ][BK]  probabilities
    device bfloat* vp        [[buffer(1)]],   // [ctx][D]  values
    device float* op_        [[buffer(2)]],   // [BQ][D]   output accumulator
    const constant int& ctx  [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 64, D = 128;

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> P(pp, dextents<int32_t, 2>(BK, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> V(vp, dextents<int32_t, 2>(D, ctx));
  tensor<device float,  dextents<int32_t, 2>, tensor_inline> O(op_, dextents<int32_t, 2>(D, BQ));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, D, BK, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  for (int kb = 0; kb + BK <= ctx; kb += BK) {
    auto Vb = V.slice(0, kb);
    op.run(P, Vb, O);
  }
}
