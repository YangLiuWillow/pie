// Q.K^T on the DOCUMENTED tensor-slice API, where the library owns the
// memory-to-register mapping and there is no lane layout to get wrong.
// Correctness first; the hand-filled cooperative-tensor version is 128/2048
// wrong and no amount of index permutation against an undocumented layout is
// the way to fix it.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#ifndef NAX_BQ
#define NAX_BQ 64
#endif
#ifndef NAX_BK
#define NAX_BK 32
#endif

kernel void nax_slice_qk(
    // NOT const: `tensor_inline`'s constructor takes a mutable
    // `data_handle_type`, and a const pointer loses the qualifier.
    device bfloat* qp [[buffer(0)]],
    device bfloat* kp [[buffer(1)]],
    device float* sp        [[buffer(2)]],
    const constant int& reps [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constexpr int BQ = NAX_BQ, BK = NAX_BK, D = 128;

  // `tensor_inline` is the DESCRIPTOR tag for a tensor built in-kernel from a
  // pointer, as against `tensor_handle` which must be bound host-side. pie's
  // context binds plain buffers, so inline is the form that fits.
  // A is BQ x D (queries), B is D x BK (K TRANSPOSED), C is BQ x BK (scores).
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(qp, dextents<int32_t, 2>(D, BQ));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(kp, dextents<int32_t, 2>(BK, D));
  tensor<device float,  dextents<int32_t, 2>, tensor_inline> C(sp, dextents<int32_t, 2>(BK, BQ));

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BQ, BK, D,
      /*transpose_left=*/false, /*transpose_right=*/false,
      /*relaxed_precision=*/false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

  // `reps` is 1 for the correctness run and large for the rate run. Repeating
  // the SAME op is a compute-bound measurement -- the operands stay resident,
  // so it prices the instruction path and not the memory system, which is the
  // only thing measurable while the machine is contended.
  for (int i = 0; i < reps; ++i) {
    op.run(A, B, C);
  }
}
