// Raw matrix throughput: M5 neural accelerators vs simdgroup_matrix.
//
// Every structural explanation for pie's 4.4x attention gap has now been
// measured and eliminated -- paging, page size, addressing, memory layout,
// staging, tile shape, accumulator type, chunking. What is left is that MLX
// ships kernels built on `mpp::tensor_ops::matmul2d` (16x16 fragments, the
// neural accelerators) and pie uses `simdgroup_matrix` (8x8) exclusively.
//
// This prices the two instructions against each other with nothing else in the
// picture: no attention, no memory traffic, no MLX. Operands live in registers
// and the loop runs long enough that the only thing being timed is issue rate.
//
// Each kernel runs its unit in its NATIVE configuration rather than a forced
// common one, because the question is "what is the best each unit can do",
// not "what do they do under identical constraints":
//   * NAX  -- bf16 operands, fp32 accumulate, 16x32x16 (MLX's own descriptor)
//   * simdgroup -- half operands, half accumulate, 8x8x8 (pie's shipped form)
//
// TWO INDEPENDENT ACCUMULATOR CHAINS in each. With one chain both kernels
// measure instruction LATENCY, not throughput, and a latency number would make
// the wider instruction look bad for the wrong reason.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// FLOPs per iteration, per simdgroup: 2 runs * 2*16*32*16 = 32768.
kernel void nax_rate(device float* out [[buffer(0)]],
                     const constant int& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16,
      /*transpose_a=*/false, /*transpose_b=*/false, /*relaxed=*/true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c0 = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
  auto ct_c1 = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

  // Seeded from gid so the compiler cannot fold the operands to constants.
  const bfloat seed = bfloat(float(gid & 7) * 0.125f + 0.5f);
  for (short i = 0; i < 8; i++) ct_a[i] = seed;
  for (short i = 0; i < 16; i++) ct_b[i] = seed;
  for (short i = 0; i < 16; i++) { ct_c0[i] = 0.0f; ct_c1[i] = 0.0f; }

  for (int t = 0; t < iters; t++) {
    op.run(ct_a, ct_b, ct_c0);
    op.run(ct_a, ct_b, ct_c1);
  }

  float acc = 0.0f;
  for (short i = 0; i < 16; i++) acc += ct_c0[i] + ct_c1[i];
  out[gid] = acc;
}

// FLOPs per iteration, per simdgroup: 8 mma * 2*8*8*8 = 8192.
kernel void simdgroup_rate(device float* out [[buffer(0)]],
                           const constant int& iters [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
  const half seed = half(float(gid & 7) * 0.125f + 0.5f);
  simdgroup_matrix<half, 8, 8> A = make_filled_simdgroup_matrix<half, 8, 8>(seed);
  simdgroup_matrix<half, 8, 8> B = make_filled_simdgroup_matrix<half, 8, 8>(seed);
  simdgroup_matrix<half, 8, 8> C[8];
  for (short i = 0; i < 8; i++) C[i] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

  // MANUALLY UNROLLED. `C[i]` under a loop index cannot live in registers
  // unless the loop is fully unrolled -- an indexed simdgroup_matrix array
  // spills, and a spilled accumulator measures the stack, not the matrix unit.
  for (int t = 0; t < iters; t++) {
    simdgroup_multiply_accumulate(C[0], A, B, C[0]);
    simdgroup_multiply_accumulate(C[1], A, B, C[1]);
    simdgroup_multiply_accumulate(C[2], A, B, C[2]);
    simdgroup_multiply_accumulate(C[3], A, B, C[3]);
    simdgroup_multiply_accumulate(C[4], A, B, C[4]);
    simdgroup_multiply_accumulate(C[5], A, B, C[5]);
    simdgroup_multiply_accumulate(C[6], A, B, C[6]);
    simdgroup_multiply_accumulate(C[7], A, B, C[7]);
  }

  float acc = 0.0f;
  for (short i = 0; i < 8; i++) {
    thread auto& e = C[i].thread_elements();
    acc += float(e[0]) + float(e[1]);
  }
  out[gid] = acc;
}

// ── Chasing the simdgroup arm ──
//
// `simdgroup_rate` above reads 5.38 TFLOP/s, but pie's shipped attention kernel
// reaches ~6.9 on its multiply half. A real kernel cannot beat its unit's peak,
// so one of the two is wrong. These two variants test the microbenchmark side.
// Manually unrolling the accumulator array already changed nothing.

// Sixteen chains instead of eight. If the 8-chain version was latency-bound --
// not enough independent work in flight to cover the matrix instruction's
// latency -- this is faster. If issue rate is the bound, this is identical.
// FLOPs per iteration, per simdgroup: 16 mma * 2*8*8*8 = 16384.
kernel void simdgroup_rate16(device float* out [[buffer(0)]],
                             const constant int& iters [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
  const half seed = half(float(gid & 7) * 0.125f + 0.5f);
  simdgroup_matrix<half, 8, 8> A = make_filled_simdgroup_matrix<half, 8, 8>(seed);
  simdgroup_matrix<half, 8, 8> B = make_filled_simdgroup_matrix<half, 8, 8>(seed);
  simdgroup_matrix<half, 8, 8> C[16];
#define SGMMA(i) simdgroup_multiply_accumulate(C[i], A, B, C[i])
  for (short i = 0; i < 16; i++) C[i] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
  for (int t = 0; t < iters; t++) {
    SGMMA(0);  SGMMA(1);  SGMMA(2);  SGMMA(3);
    SGMMA(4);  SGMMA(5);  SGMMA(6);  SGMMA(7);
    SGMMA(8);  SGMMA(9);  SGMMA(10); SGMMA(11);
    SGMMA(12); SGMMA(13); SGMMA(14); SGMMA(15);
  }
#undef SGMMA
  float acc = 0.0f;
  for (short i = 0; i < 16; i++) {
    thread auto& e = C[i].thread_elements();
    acc += float(e[0]) + float(e[1]);
  }
  out[gid] = acc;
}

// fp32 operands and accumulator, which is what MLX's simdgroup kernel uses and
// what the neural-accelerator path accumulates in. pie accumulates in half. If
// the matrix unit is not actually faster in half, pie's whole DCH chunking
// tradeoff -- which exists ONLY to bound half-rounding -- was paid for nothing.
// FLOPs per iteration, per simdgroup: 8 mma * 2*8*8*8 = 8192.
kernel void simdgroup_rate_f32(device float* out [[buffer(0)]],
                               const constant int& iters [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
  const float seed = float(gid & 7) * 0.125f + 0.5f;
  simdgroup_matrix<float, 8, 8> A = make_filled_simdgroup_matrix<float, 8, 8>(seed);
  simdgroup_matrix<float, 8, 8> B = make_filled_simdgroup_matrix<float, 8, 8>(seed);
  simdgroup_matrix<float, 8, 8> C[8];
#define SGMMAF(i) simdgroup_multiply_accumulate(C[i], A, B, C[i])
  for (short i = 0; i < 8; i++) C[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int t = 0; t < iters; t++) {
    SGMMAF(0); SGMMAF(1); SGMMAF(2); SGMMAF(3);
    SGMMAF(4); SGMMAF(5); SGMMAF(6); SGMMAF(7);
  }
#undef SGMMAF
  float acc = 0.0f;
  for (short i = 0; i < 8; i++) {
    thread auto& e = C[i].thread_elements();
    acc += e[0] + e[1];
  }
  out[gid] = acc;
}
