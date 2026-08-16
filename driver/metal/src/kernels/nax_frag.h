// NAX 16x16 register fragments: the form a FUSED attention needs.
//
// ## Why this exists, when `nax_slice_qk.metal` already works
//
// There are two ways to reach `mpp::tensor_ops::matmul2d`, and the difference
// between them is the whole reason the previous fused attempt failed.
//
//   * The **tensor-slice** form (`nax_slice_qk.metal`): build a `tensor` over
//     device or threadgroup memory and call `op.run(A, B, C)`. The library owns
//     the memory-to-register mapping, so there is no lane layout to get wrong.
//     Correct on the first try, and measured at 0.627 ms/layer for Q.Kᵀ.
//   * The **cooperative-tensor** form, here: operands live in ordinary `thread`
//     registers and a cooperative tensor is materialized for the INSTANT of the
//     multiply.
//
// The slice form cannot fuse. `matrix_rate_probe`'s isolation arm measured it:
// a cooperative destination does NOT carry across `run()` calls -- P.V twice
// into one cooperative O gave 7316 of 8192 elements wrong, worst relative
// exactly 1.0000 (zero where the reference is not). A flash attention has to
// accumulate O across every key block, so O must live somewhere that survives,
// and that means plain registers.
//
// **The design is MLX's** (`steel/attn/nax.h`), which is where the shape of
// this file comes from: O and S are `thread` arrays of fragments, and every
// cooperative tensor is created, filled, run and read back inside one function
// call. This is a reimplementation of that structure against the same public
// MetalPerformancePrimitives API, not a copy of their code -- pie compiles its
// own kernels and cannot include theirs.
//
// ## The layout, which is the one thing everything else depends on
//
// A 16x16 fragment over 32 lanes is 8 elements per lane, held as 2 rows x 4
// columns. For lane `l`, with `qid = l >> 2`:
//
//     fm = (qid & 4) | ((l >> 1) & 3)      // first row; the other is fm + 8
//     fn = ((qid & 2) | (l & 1)) * 4       // first of four consecutive columns
//
// Two consequences carry the flash-attention design, and they are the same two
// that `sdpa_paged_mma.metal` already relies on for the 8x8 simdgroup fragment:
//
//   * **A lane's elements sit in exactly two rows**, fixed for that lane across
//     every fragment it touches. So the online softmax's per-row state -- row
//     max, row sum, the rescale factor -- is per-lane state. S is never stored
//     to threadgroup memory and never read back.
//   * **The four lanes sharing a row are {l, l^1, l^8, l^9}**: `fm` depends on
//     bits 1, 2 and 4 of the lane and `fn` on bits 0 and 3, so xor by 1 and by 8
//     reach exactly the lanes with the same `fm`. Two shuffles are the entire
//     cross-lane row reduction, against the five a `simd_sum` would spend.


#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace pie_nax {

constant constexpr int kFragRows = 16;
constant constexpr int kFragCols = 16;
constant constexpr int kElemsPerFrag = 8;   // (16*16)/32
constant constexpr int kElemRows = 2;
constant constexpr int kElemCols = 4;
constant constexpr int kElemRowsJump = 8;

typedef vec<float, kElemsPerFrag> ffrag;    // accumulator fragment (S, O)
typedef vec<bfloat, kElemsPerFrag> bfrag;   // operand fragment (Q, K, V)

/// This lane's first row (`.y`) and first column (`.x`) within a fragment.
inline short2 frag_coord(uint simd_lid) {
    const short l = short(simd_lid);
    const short qid = l >> 2;
    const short fm = (qid & 4) | ((l >> 1) & 3);
    const short fn = ((qid & 2) | (l & 1)) * 4;
    return short2{fn, fm};
}

/// Load a 16x16 fragment straight from device memory.
///
/// NO THREADGROUP STAGING, deliberately. `matrix_rate_probe` measured a staged
/// NAX attention at 7.692 ms/layer against 1.492 unstaged -- the staging alone
/// was 2.921 ms, more than the whole multiply. MLX stages nothing here either:
/// the fragment loads are device reads, and the cache is what makes them cheap.
///
/// `row_stride` is in elements; the column stride is 1, which every operand
/// this kernel loads satisfies (Q, K and V are all head-dim-contiguous).
template <typename T>
inline void frag_load(thread vec<T, kElemsPerFrag>& dst, const device T* src,
                      int row_stride, uint simd_lid) {
    const short2 c = frag_coord(simd_lid);
    src += c.y * row_stride + c.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j)
            dst[i * kElemCols + j] = src[i * kElemRowsJump * row_stride + j];
}

/// The same, but refusing rows past `lim` -- the tail key block of a cache whose
/// length is not a multiple of the block. Reading them would be a load past the
/// end of the page array; zeroing them is not enough on its own, which is why
/// the caller ALSO drives their scores to -inf (a zero key still scores 0, and
/// exp(0) is 1, not 0).
template <typename T>
inline void frag_load_rows(thread vec<T, kElemsPerFrag>& dst, const device T* src,
                           int row_stride, short lim, uint simd_lid) {
    const short2 c = frag_coord(simd_lid);
    src += c.y * row_stride + c.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) {
        const short r = c.y + i * kElemRowsJump;
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j)
            dst[i * kElemCols + j] =
                r < lim ? src[i * kElemRowsJump * row_stride + j] : T(0);
    }
}

/// Load a fragment from a THREADGROUP-memory staging buffer.
///
/// The device-memory form above is right when a block is read once. It is wrong
/// when the same block is read by several simdgroups of the same threadgroup,
/// which is what a query tile wider than one simdgroup does: four simdgroups
/// each issue the same loads, and eight query heads sharing a KV head issue
/// them again. Staging once and reading from here is the alternative, and which
/// one wins is a measurement, not a principle -- see
/// `results-prefill-experiments.md`.
template <typename T>
inline void frag_load_tg(thread vec<T, kElemsPerFrag>& dst, const threadgroup T* src,
                         int row_stride, uint simd_lid) {
    const short2 c = frag_coord(simd_lid);
    src += c.y * row_stride + c.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i)
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j)
            dst[i * kElemCols + j] = src[i * kElemRowsJump * row_stride + j];
}

/// Store a float fragment to device memory, converting.
template <typename T>
inline void frag_store(const thread ffrag& src, device T* dst, int row_stride,
                       short lim, uint simd_lid) {
    const short2 c = frag_coord(simd_lid);
    dst += c.y * row_stride + c.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) {
        const short r = c.y + i * kElemRowsJump;
        if (r >= lim) continue;
#pragma clang loop unroll(full)
        for (short j = 0; j < kElemCols; ++j)
            dst[i * kElemRowsJump * row_stride + j] = T(src[i * kElemCols + j]);
    }
}

/// C[16x32] += A[16x16] * B[16x32], on the neural accelerators.
///
/// N is 32 because that is what one `matmul2d` issues, so C and B each arrive
/// as a PAIR of 16-wide fragments. The cooperative tensors are created here and
/// die here: that lifetime is the correctness condition this whole file is
/// arranged around.
template <bool TRANSPOSE_B>
inline void frag_mma(thread ffrag& c0, thread ffrag& c1,
                     const thread bfrag& a,
                     const thread bfrag& b0, const thread bfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_left=*/false, /*transpose_right=*/TRANSPOSE_B,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c =
        op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b),
                                                     float>();

#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) ct_a[i] = a[i];
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) {
        ct_b[i] = b0[i];
        ct_b[kElemsPerFrag + i] = b1[i];
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) {
        ct_c[i] = c0[i];
        ct_c[kElemsPerFrag + i] = c1[i];
    }

    op.run(ct_a, ct_b, ct_c);

#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[kElemsPerFrag + i];
    }
}

/// C[16x16] += A[16x32] * B[16x32]ᵀ — the N=16 case, done by widening K.
///
/// `frag_mma` issues N=32, and a simdgroup owning an ODD number of 16-wide
/// column fragments cannot use it. The dense projections are exactly that: the
/// driver's tile is bm=64/bn=32 at WM=WN=2, so TN=1.
///
/// The obvious fix — a `matmul2d_descriptor(16, 16, 16, ...)` — is refused by
/// the API, and the refusal is explicit:
///
///     "At least one of M, N, or K must be 32 if both inputs are cooperative
///      tensors"
///
/// So the 32 goes on K instead of N. One call consumes TWO K-halves of A and of
/// B and produces one 16x16 accumulator, which suits the dense tile exactly:
/// BK is 32, so this covers a whole K-block in one instruction where the N=32
/// form takes two.
///
/// The alternative was re-shaping the warps, and it is worse: the dense GEMM's
/// threadgroup is dispatched as {32, WM, WN}, so changing WM/WN moves a LAUNCH
/// shape — the class of change this repo keeps getting wrong. This keeps the
/// launch identical and makes the swap a matter of the entrypoint name again.
template <bool TRANSPOSE_B>
inline void frag_mma_k32(thread ffrag& c,
                         const thread bfrag& a0, const thread bfrag& a1,
                         const thread bfrag& b0, const thread bfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 32,
        /*transpose_left=*/false, /*transpose_right=*/TRANSPOSE_B,
        /*relaxed_precision=*/true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c =
        op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b),
                                                      float>();
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) {
        ct_a[i] = a0[i];
        ct_a[kElemsPerFrag + i] = a1[i];
        ct_b[i] = b0[i];
        ct_b[kElemsPerFrag + i] = b1[i];
        ct_c[i] = c[i];
    }
    op.run(ct_a, ct_b, ct_c);
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemsPerFrag; ++i) c[i] = ct_c[i];
}

/// Reduce each of a lane's two rows across the four lanes that share it, then
/// combine into `acc`. `{l, l^1, l^8, l^9}` -- see the header note.
template <typename Op>
inline void frag_row_reduce(const thread ffrag& f, thread float* acc) {
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; ++i) {
        float t = Op::apply(Op::apply(f[i * kElemCols + 0], f[i * kElemCols + 1]),
                            Op::apply(f[i * kElemCols + 2], f[i * kElemCols + 3]));
        t = Op::apply(t, simd_shuffle_xor(t, ushort(1)));
        t = Op::apply(t, simd_shuffle_xor(t, ushort(8)));
        acc[i] = Op::apply(acc[i], t);
    }
}

struct MaxOp { static float apply(float a, float b) { return metal::max(a, b); } };
struct SumOp { static float apply(float a, float b) { return a + b; } };

}  // namespace pie_nax
