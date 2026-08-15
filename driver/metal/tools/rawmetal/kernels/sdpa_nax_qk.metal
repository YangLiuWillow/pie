// STEP 3, STAGE 1: does NAX's 6x survive at attention shapes? Probe only.
//
// `matrix_rate_probe` measured the neural accelerators at 31 TFLOP/s against
// `simdgroup_matrix`'s 5.2 -- but it fills its operands ONCE and then loops,
// so it prices the instruction and not the work around it. An attention kernel
// refills every operand every pass: 8 elements per lane into A, 16 into B, 16
// out of C, all through cooperative tensors. That fill is untested, and every
// number in the step-3 plan assumes the 6x survives it.
//
// So this computes S = Q K^T ONLY -- no softmax, no PV, no correctness -- at
// the real serving shape, and reports TFLOP/s. Near 31 means the fill is free
// and the plan holds. Near 5 means the fill eats the instruction and the plan
// needs re-thinking before anyone writes a flash-attention loop around it.
//
// DECISION RULE, written before the run: the plan assumed 6x. Under 3x
// (i.e. below ~16 TFLOP/s here) stop and re-plan.
//
// Tile is the one `device_caps` picked: BQ=64 forced by the 16x16 fragment
// (TQ = BQ/(4 warps * 16) must be 1), BK=32 because BQ=64/BK=64 needs 35 KB
// against a measured 32 KB cap. Same shape as MLX's `bq64_bk32_bd128`.
//
// Staging is deliberately minimal, matching `sdpa_nostage_mma.metal`: one write
// per thread per pass. Not laziness -- it isolates the multiply, and without
// SOME write the tiles are provably loop-invariant and the compiler hoists the
// operand loads clean out of the loop, which is how the first ablation priced
// pie's multiply above its own unit's ceiling.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void sdpa_nax_qk(
    device float* out            [[buffer(0)]],
    const constant int& ctx_len  [[buffer(1)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 64;    // query rows per threadgroup; forced by the fragment
  constexpr int BK = 32;    // keys per pass; what 32 KB allows with Q staged
  constexpr int D  = 128;   // head dim
  constexpr int TD = D / 16;   // fragments down the head: 8
  constexpr int PAD = 8;       // 16 bytes at bf16, MLX's padding unit

  threadgroup bfloat qtile[BQ * (D + PAD)];    // [row][dim]  17.4 KB
  threadgroup bfloat ktile[D * (BK + PAD)];    // [dim][key]  10.2 KB

  const uint lid = simd_gid * 32u + simd_lid;

  // The 16x16 fragment's lane map. A lane owns EIGHT elements: two rows, eight
  // apart, four columns each. `fm` depends on lane bits {4,2,1} and `fn` on
  // {3,0} -- disjoint, which is why the row-sharing lanes are still
  // {l, l^1, l^8, l^9} and pie's two-shuffle row reduction survives into
  // stage 2. See the plan's fragment-layout section.
  const short qid = short(simd_lid) >> 2;
  const short fm  = (qid & 4) | ((short(simd_lid) >> 1) & 3);
  const short fn  = ((qid & 2) | (short(simd_lid) & 1)) * 4;

  for (uint e = lid; e < uint(BQ * (D + PAD)); e += 128u) qtile[e] = bfloat(0.5h);
  for (uint e = lid; e < uint(D * (BK + PAD)); e += 128u) ktile[e] = bfloat(0.5h);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16,
      /*transpose_a=*/false, /*transpose_b=*/false, /*relaxed=*/true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

  // This simdgroup's 16 query rows of the threadgroup's 64.
  const int q_row = int(simd_gid) * 16;
  float acc = 0.0f;

  // Q IS LOOP-INVARIANT and hoisting it out of the pass loop LOSES: caching
  // the eight A-fragments as `bfloat qfrag[TD][8]` -- 64 bfloats a lane --
  // measured 7.88 TFLOP/s against 16.70 for re-reading them from threadgroup
  // memory. An array that large, indexed by a loop variable, spills to the
  // stack, and a spilled operand is worse than a threadgroup re-read. Same
  // shape of mistake as the accumulator array in `matrix_rate.metal`. Do not
  // retry without a full unroll AND a register-pressure check.

  const int passes = ctx_len / BK;
  for (int kb = 0; kb < passes; ++kb) {
    // One write per thread. See the header: without it the loads hoist.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ktile[lid] = bfloat(float(kb & 7) * 0.125f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short i = 0; i < 16; ++i) ct_c[i] = 0.0f;

    for (int dd = 0; dd < TD; ++dd) {
      // A = Q[16 rows][16 dims] at (q_row, dd*16).
      for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < 4; ++j) {
          ct_a[i * 4 + j] =
              qtile[(q_row + int(fm) + i * 8) * (D + PAD) + dd * 16 + int(fn) + j];
        }
      }
      // B = K^T[16 dims][32 keys], two 16x16 fragments side by side.
      for (short nf = 0; nf < 2; ++nf) {
        for (short i = 0; i < 2; ++i) {
          for (short j = 0; j < 4; ++j) {
            ct_b[nf * 8 + i * 4 + j] =
                ktile[(dd * 16 + int(fm) + i * 8) * (BK + PAD) + nf * 16 + int(fn) + j];
          }
        }
      }
      op.run(ct_a, ct_b, ct_c);
    }
    // Consume S so the whole pass is not dead code. Stage 2 replaces this with
    // the online softmax and the P V multiply.
    for (short i = 0; i < 16; ++i) acc += ct_c[i];
  }

  out[tid.x * 128u + lid] = acc;
}
