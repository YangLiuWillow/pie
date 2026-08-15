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

#ifndef PIE_NAX_STAGE
#define PIE_NAX_STAGE 1
#endif
#ifndef PIE_NAX_KT_STRAIGHT
#define PIE_NAX_KT_STRAIGHT 0
#endif

kernel void sdpa_nax_qk(
    device float* out            [[buffer(0)]],
    const constant int& ctx_len  [[buffer(1)]],
    const device bfloat* q_dev   [[buffer(2)]],
    const device bfloat* k_dev   [[buffer(3)]],
    const device bfloat* v_dev   [[buffer(4)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 64;    // query rows per threadgroup; forced by the fragment
  constexpr int BK = 32;    // keys per pass; what 32 KB allows with Q staged
  constexpr int D  = 128;   // head dim
  constexpr int TD = D / 16;   // fragments down the head: 8
  constexpr int PAD = 8;       // 16 bytes at bf16, MLX's padding unit

  threadgroup bfloat qtile[BQ * (D + PAD)];    // [row][dim]  17.4 KB
  // K AND V SHARE ONE BUFFER. Separate tiles are 10.2 + 8.7 = 18.9 KB, and
  // with Q that is 36.3 KB against a measured 32 KB cap. V is not needed until
  // S has been computed, so the buffer is K-transposed during Q.K^T and then
  // overwritten with V for P.V, with a barrier between. This is MLX's
  // `Ks = KV_smem; Vs = KV_smem` and it is what makes the tile fit at all.
  constexpr int KV_ELEMS = (D * (BK + PAD)) > (BK * (D + PAD))
                               ? (D * (BK + PAD)) : (BK * (D + PAD));
  threadgroup bfloat ktile[KV_ELEMS];          // K^T [dim][key] | V [key][dim]

  const uint lid = simd_gid * 32u + simd_lid;

  // The 16x16 fragment's lane map. A lane owns EIGHT elements: two rows, eight
  // apart, four columns each. `fm` depends on lane bits {4,2,1} and `fn` on
  // {3,0} -- disjoint, which is why the row-sharing lanes are still
  // {l, l^1, l^8, l^9} and pie's two-shuffle row reduction survives into
  // stage 2. See the plan's fragment-layout section.
  const short qid = short(simd_lid) >> 2;
  const short fm  = (qid & 4) | ((short(simd_lid) >> 1) & 3);
  const short fn  = ((qid & 2) | (short(simd_lid) & 1)) * 4;

#if PIE_NAX_STAGE >= 4 || PIE_NAX_STAGE == 5
  // Q staged once from device: [row][dim] -> padded [row][dim].
  for (uint e = lid; e < uint(BQ * D); e += 128u) {
    const int r = int(e) / D, d = int(e) - r * D;
    qtile[r * (D + PAD) + d] = q_dev[size_t(tid.x % 3u) * BQ * D + e];
  }
#else
  for (uint e = lid; e < uint(BQ * (D + PAD)); e += 128u) qtile[e] = bfloat(0.5h);
#endif
  for (uint e = lid; e < uint(KV_ELEMS); e += 128u) ktile[e] = bfloat(0.5h);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // PIE_NAX_KT_STRAIGHT: stage K in [key][dim] like V and let the descriptor
  // transpose it, instead of writing K transposed into threadgroup memory.
  // The transposed write puts consecutive lanes (BK+PAD)=40 halves apart --
  // 20 words, gcd(20,32)=4, so 32 lanes reach 8 banks. The straight write is
  // contiguous. This is the same bank-conflict question the 8x8 kernel lost on
  // when padded, but that experiment moved the fragment LOAD too; this one
  // moves only the write, with the transpose handled by the instruction.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16,
#if PIE_NAX_KT_STRAIGHT
      /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed=*/true,
#else
      /*transpose_a=*/false, /*transpose_b=*/false, /*relaxed=*/true,
#endif
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

  // This simdgroup's 16 query rows of the threadgroup's 64.
  const int q_row = int(simd_gid) * 16;
  float acc = 0.0f;
#if PIE_NAX_STAGE >= 2 && PIE_NAX_STAGE != 5
  float ov[4][16];
  for (short n = 0; n < 4; ++n)
    for (short i = 0; i < 16; ++i) ov[n][i] = 0.0f;
#endif
#if PIE_NAX_STAGE >= 3 && PIE_NAX_STAGE != 5
  float row_max[2] = {-3.0e38f, -3.0e38f};
  float row_sum[2] = {0.0f, 0.0f};
#endif

  // Q IS LOOP-INVARIANT and hoisting it out of the pass loop LOSES: caching
  // the eight A-fragments as `bfloat qfrag[TD][8]` -- 64 bfloats a lane --
  // measured 7.88 TFLOP/s against 16.70 for re-reading them from threadgroup
  // memory. An array that large, indexed by a loop variable, spills to the
  // stack, and a spilled operand is worse than a threadgroup re-read. Same
  // shape of mistake as the accumulator array in `matrix_rate.metal`. Do not
  // retry without a full unroll AND a register-pressure check.

  const int passes = ctx_len / BK;
  for (int kb = 0; kb < passes; ++kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if PIE_NAX_STAGE >= 4 || PIE_NAX_STAGE == 5
    // K staged TRANSPOSED, [dim][key], so the Q.K^T operand fill is a
    // contiguous read. 32 keys x 128 dims by 128 threads: 32 elements each.
    for (uint e = lid; e < uint(BK * D); e += 128u) {
      const int kk = int(e) / D, d = int(e) - kk * D;
#if PIE_NAX_KT_STRAIGHT
      ktile[kk * (D + PAD) + d] = k_dev[(size_t(kb) * BK + kk) * D + d];
#else
      ktile[d * (BK + PAD) + kk] = k_dev[(size_t(kb) * BK + kk) * D + d];
#endif
    }
#else
    // One write per thread. See the header: without it the loads hoist.
    ktile[lid] = bfloat(float(kb & 7) * 0.125f);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short i = 0; i < 16; ++i) ct_c[i] = 0.0f;

#if PIE_NAX_STAGE == 5
    // Staging only. The tiles must be CONSUMED or the writes above are dead
    // and the compiler deletes the thing being timed -- the same trap that
    // priced the shipped kernel's multiply above its own unit's ceiling.
    acc += float(ktile[lid]) + float(qtile[lid]);
#else
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
#if PIE_NAX_KT_STRAIGHT
            // B is K in [key][dim]; the descriptor's transpose_b handles the
            // orientation, so the fill reads along the contiguous axis.
            ct_b[nf * 8 + i * 4 + j] =
                ktile[(nf * 16 + int(fm) + i * 8) * (D + PAD) + dd * 16 + int(fn) + j];
#else
            ct_b[nf * 8 + i * 4 + j] =
                ktile[(dd * 16 + int(fm) + i * 8) * (BK + PAD) + nf * 16 + int(fn) + j];
#endif
          }
        }
      }
      op.run(ct_a, ct_b, ct_c);
    }
#endif
#if PIE_NAX_STAGE >= 3 && PIE_NAX_STAGE != 5
    // ── Online softmax, per lane, no threadgroup round trip ──
    //
    // A lane holds two rows (fm and fm+8), four columns in each of the two
    // C fragments -- so eight columns per row half. The row-sharing lanes are
    // still {l, l^1, l^8, l^9} (verified exhaustively; see the plan), so the
    // row max and row sum are still two `simd_shuffle_xor` steps. What doubles
    // against the 8x8 kernel is the STATE: one max and one sum per row half.
    for (short i = 0; i < 2; ++i) {
      float lmax = -3.0e38f;
      for (short nf = 0; nf < 2; ++nf) {
        for (short j = 0; j < 4; ++j) {
          const float v = ct_c[nf * 8 + i * 4 + j];
          lmax = v > lmax ? v : lmax;
        }
      }
      lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
      lmax = max(lmax, simd_shuffle_xor(lmax, 8u));

      const float new_max = max(row_max[i], lmax);
      const float factor = row_max[i] == -3.0e38f ? 0.0f : fast::exp(row_max[i] - new_max);
      float lsum = 0.0f;
      for (short nf = 0; nf < 2; ++nf) {
        for (short j = 0; j < 4; ++j) {
          const float p = fast::exp(ct_c[nf * 8 + i * 4 + j] - new_max);
          ct_c[nf * 8 + i * 4 + j] = p;
          lsum += p;
        }
      }
      lsum += simd_shuffle_xor(lsum, 1u);
      lsum += simd_shuffle_xor(lsum, 8u);
      row_max[i] = new_max;
      row_sum[i] = row_sum[i] * factor + lsum;
      for (short n = 0; n < 4; ++n)
        for (short j = 0; j < 4; ++j) ov[n][i * 4 + j] *= factor;
    }
#endif

#if PIE_NAX_STAGE >= 2 && PIE_NAX_STAGE != 5
    // ── O += P V ──
    //
    // O is 16 rows x 128 dims per simdgroup: FOUR 16x32 accumulators, 64
    // floats a lane. That is double the 8x8 kernel's 32, and it is the main
    // thing this stage measures -- a spill here costs more than the matmul
    // saves, which is exactly what killed the Q hoist.
    //
    // V overwrites K in the shared tile. The barrier is not optional.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if PIE_NAX_STAGE >= 4 || PIE_NAX_STAGE == 5
    // V staged straight, [key][dim], overwriting K in the shared buffer.
    for (uint e = lid; e < uint(BK * D); e += 128u) {
      const int kk = int(e) / D, d = int(e) - kk * D;
      ktile[kk * (D + PAD) + d] = v_dev[(size_t(kb) * BK + kk) * D + d];
    }
#else
    ktile[lid] = bfloat(float(kb & 3) * 0.25f);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short n = 0; n < 4; ++n) {
      auto ct_o = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
      for (short i = 0; i < 16; ++i) ct_o[i] = ov[n][i];
      for (short kc = 0; kc < 2; ++kc) {
        // A = P[16 rows][16 keys], the kc-th half of this pass's 32 keys.
        for (short i = 0; i < 8; ++i) ct_a[i] = bfloat(ct_c[kc * 8 + i]);
        // B = V[16 keys][32 dims] at (kc*16, n*32).
        for (short nf = 0; nf < 2; ++nf)
          for (short i = 0; i < 2; ++i)
            for (short j = 0; j < 4; ++j)
              ct_b[nf * 8 + i * 4 + j] =
                  ktile[(kc * 16 + int(fm) + i * 8) * (D + PAD) + n * 32 + nf * 16 + int(fn) + j];
        op.run(ct_a, ct_b, ct_o);
      }
      for (short i = 0; i < 16; ++i) ov[n][i] = ct_o[i];
    }
#else
    // Stage 1 only: consume S so the pass is not dead code.
    for (short i = 0; i < 16; ++i) acc += ct_c[i];
#endif
  }

#if PIE_NAX_STAGE >= 2 && PIE_NAX_STAGE != 5
  for (short n = 0; n < 4; ++n)
    for (short i = 0; i < 16; ++i) acc += ov[n][i];
#endif
#if PIE_NAX_STAGE >= 3 && PIE_NAX_STAGE != 5
  acc = acc / (row_sum[0] + row_sum[1] + 1.0f) + row_max[0];
#endif
  out[tid.x * 128u + lid] = acc;
}
