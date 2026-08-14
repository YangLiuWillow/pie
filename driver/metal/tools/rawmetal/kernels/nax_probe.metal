// Can this driver reach the M5 neural accelerators at all? Probe only.
//
// MLX 0.31.3's metallib contains `mpp::tensor_ops::matmul2d` kernels
// (`gemm_splitk_nax`, `segmented_mm_nax`, `BaseNAXFrag`, and the `bq64`
// attention configs, whose `kU=16` fragment is what lets them satisfy the
// `TQ == 1` assert that an 8x8 fragment cannot). That is a SECOND matrix path,
// distinct from `simdgroup_matrix`, and this machine is an M5 Pro.
//
// pie uses `simdgroup_matrix` exclusively. Before anyone spends another week
// tuning it, this establishes whether the other path is even available through
// pie's runtime shader compiler, which is what `MTLLanguageVersion4_0` and
// `newLibraryWithSource:` give us without a full Xcode install.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void nax_probe(const device bfloat* A [[buffer(0)]],
                      const device bfloat* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, 16, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  (void)op;
  C[tid] = float(A[tid]) + float(B[tid]);
}
