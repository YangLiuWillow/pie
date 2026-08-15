// The whole-kernel number, on the design that actually exists.
//
// The queued staging measurement was aimed at `sdpa_nax_staged` -- the
// hand-filled kernel with manual threadgroup staging. That design is dead: it
// computes the wrong thing AND its 32 KB cap forced the BK=32 tile that cost
// 40% of the throughput. The slice kernel stages nothing, so "the staging
// half" is not a term it has, and the total cannot be composed from parts. It
// has to be walked over a real context and measured.
//
// Q.K^T only, accumulating across key blocks. Softmax and P.V are not here:
// this is the memory-bound term the tile sweep could not see, because that
// swept the same cache-resident block 232 times.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_slice_full(
    device bfloat* qp        [[buffer(0)]],   // [BQ][D]
    device bfloat* kp        [[buffer(1)]],   // [D][ctx]  (K transposed)
    device float* sp         [[buffer(2)]],   // [BQ][BK]  accumulator
    const constant int& ctx  [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 64, D = 128;   // the tile the sweep chose

  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(qp, dextents<int32_t, 2>(D, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> Kf(kp, dextents<int32_t, 2>(ctx, D));
  tensor<device float,  dextents<int32_t, 2>, tensor_inline> C(sp, dextents<int32_t, 2>(BK, BQ));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, BK, D, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  // Walk the context one key block at a time, so every key is read once from
  // device memory -- the traffic a real prefill pays.
  for (int kb = 0; kb + BK <= ctx; kb += BK) {
    auto Bk = Kf.slice(kb, 0);
    op.run(A, Bk, C);
  }
}
