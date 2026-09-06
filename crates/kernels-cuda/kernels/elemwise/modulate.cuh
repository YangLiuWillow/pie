#pragma once

#include "prelude/device.cuh"
#include "elemwise/norm.cuh"

namespace pie::elemwise {

/// **THE MODULATION A DiT BLOCK IS BUILT OUT OF** (`.wiki/imagegen/design.md`
/// D6). An adaLN block does nothing to a row that is not one of three
/// shapes: `x·(1+s)+b`, `x·(1+s)`, and `tanh(g)·x`. All three read the same
/// `m` rectangle and differ only in what they take out of it, so they are one
/// kernel under a form flag rather than three files.
///
/// **WHERE `m`'S ROW COMES FROM IS THE OP'S ONE REAL DECISION.** A modulation
/// vector is per SAMPLE (`[lanes, k·D]`, the adaLN MLP's output for the
/// lane's timestep) or per TOKEN (`[rows, k·D]`, Wan-TI2V/LTX per-token
/// timesteps, MiniMax's modality tags). `lane_of_row` is the difference: a
/// `[rows]` i32 plane holding each row's lane — `RequestOfToken` — and its
/// absence means "`m` is indexed by the row itself". The lane it yields is an
/// ABSOLUTE lane id, so a body replayed above lane zero reads the right
/// vector without the lane words.
constexpr int kModScaleShift = 0;
constexpr int kModScale = 1;
constexpr int kModTanhGate = 2;

/// Which row of `m` this row reads: the lane the map names, or the row
/// itself when no map is bound.
__device__ __forceinline__ int modulation_row(
    const i32* __restrict__ lane_of_row, int row)
{
    return lane_of_row != nullptr ? lane_of_row[row] : row;
}

/// `o = x·(1+s)+b` / `x·(1+s)` / `tanh(g)·x`, one block per row.
///
/// The scale and the shift are ONE `fmaf` — the fused multiply-add is the
/// arithmetic this op states, not an accident of `--fmad=true`, so the host
/// reference is `mul_add` and the two agree to the bit. Everything else is
/// f32 in, f32 out, one rounding at the store. `o` may alias `x`: every
/// thread reads its own columns before it writes them.
/// `TM` is the modulation plane's element: `T` itself, or `float` when the
/// vector arrives from a lane chain kept in f32 (the timestep embedding's
/// linear lands f32). The arithmetic is f32 either way; only the read
/// changes, so `m` in f32 is exact and `m` in `T` rounds once at its store.
template <class T, class TM, int FORM>
__global__ void modulate(
    const T* __restrict__ x,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (`rmsnorm_row`'s idiom, one block per row): a
    // replay whose grid was carved at a bucket retires its padded rows here,
    // off a word the fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: `x`, `o` and the `[rows]`
    // lane map arrive at their plane base. `m` does NOT move — it is keyed by
    // the lane id that map yields, or by the row in the same frame.
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;
    const TM* mr = m + static_cast<long long>(modulation_row(lane_of_row, row)) * m_width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float xv = Elem<T>::to_f32(xr[i]);
        float v;
        if constexpr (FORM == kModScaleShift) {
            v = fmaf(xv, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width]));
        } else if constexpr (FORM == kModScale) {
            v = xv * (1.f + Elem<TM>::to_f32(mr[i]));
        } else {
            v = tanhf(Elem<TM>::to_f32(mr[i])) * xv;
        }
        orow[i] = Elem<T>::from_f32(v);
    }
}

/// **THE GATED RESIDUAL**: `r_out = r + g·y`, the write every DiT sub-block
/// ends with (`x += gate_msa · attn`). One `fmaf` and one rounding, so a
/// chain of them accumulates no more error than the reference's.
///
/// `g` reads its row the way `modulate`'s `m` does — per lane through
/// `lane_of_row`, or per token without it. `r_out` may alias `r`; that is the
/// in-place form the IR spells by aliasing the output onto the input.
template <class T, class TM>
__global__ void gated_residual_add(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ r_out,
    int width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* rr = r + static_cast<long long>(row) * width;
    const T* yr = y + static_cast<long long>(row) * width;
    T* orow = r_out + static_cast<long long>(row) * width;
    const TM* gr = g + static_cast<long long>(modulation_row(lane_of_row, row)) * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        orow[i] = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                         Elem<T>::to_f32(yr[i]),
                                         Elem<T>::to_f32(rr[i])));
    }
}

/// Which norm the fused pass runs. `kNormLayer` is LayerNorm with no affine
/// at all (the modulation IS the affine), `kNormRmsNoScale` the same for the
/// rms family, `kNormRmsWeight` the rms norm that still carries a weight
/// bank in front of the modulation (LTX's QK-norm shape).
constexpr int kNormLayer = 0;
constexpr int kNormRmsNoScale = 1;
constexpr int kNormRmsWeight = 2;

/// **NORM THEN MODULATE, ONE ROW REDUCTION AND ONE PASS** — and, when
/// `GATED`, the residual fold in front of it (the deferred-residual form the
/// FLUX.2/LTX references write as `x += gate·y; h = norm(x)·(1+s)+b`).
///
/// **THE RESIDUAL IS BIT-EQUAL TO THE UNFUSED WRITE AND THE MOMENTS ARE
/// TAKEN OVER IT** — `residual_add_rmsnorm`'s discipline one file over: the
/// sum is rounded to `T`, stored, and reduced in that rounded form, so a
/// trace that fuses lands what the two launches land rather than something
/// nearby.
///
/// The NORMED row, by contrast, stays in f32 all the way to the modulation's
/// single rounding — `layernorm`'s argument: the intermediate the unfused
/// chain rounds is a storage type this kernel does not have, and reproducing
/// its rounding would be reproducing an artifact. The two therefore agree to
/// about one bf16 ulp on `o`, not to the bit.
///
/// LayerNorm reduces TWICE (mean, then centred squares) and not once through
/// `E[x²]-E[x]²`, for `layernorm_row`'s reason: a row whose mean is large
/// against its spread cancels catastrophically in f32.
template <class T, class TM, int BLOCK, int NORM, bool GATED>
__device__ __forceinline__ void norm_modulate_row(
    const T* __restrict__ src,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;
    const int mrow = modulation_row(lane_of_row, row);
    const TM* mr = m + static_cast<long long>(mrow) * m_width;
    T* orow = o + static_cast<long long>(row) * width;

    // The normed row's storage: the residual plane when this pass folds one
    // (it must be written anyway), the source plane otherwise.
    const T* normed = GATED ? residual + static_cast<long long>(row) * width
                            : src + static_cast<long long>(row) * width;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    if constexpr (GATED) {
        const T* rr = src + static_cast<long long>(row) * width;
        const T* yr = y + static_cast<long long>(row) * width;
        const TM* gr = g + static_cast<long long>(mrow) * width;
        T* rout = residual + static_cast<long long>(row) * width;
        for (int i = tid; i < width; i += BLOCK) {
            const T summed = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                                    Elem<T>::to_f32(yr[i]),
                                                    Elem<T>::to_f32(rr[i])));
            rout[i] = summed;
            const float v = Elem<T>::to_f32(summed);
            local += NORM == kNormLayer ? v : v * v;
        }
    } else {
        for (int i = tid; i < width; i += BLOCK) {
            const float v = Elem<T>::to_f32(normed[i]);
            local += NORM == kNormLayer ? v : v * v;
        }
    }

    float mean = 0.f;
    float inv = 0.f;
    if constexpr (NORM == kNormLayer) {
        mean = block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(width);
        __syncthreads();
        float spread = 0.f;
        for (int i = tid; i < width; i += BLOCK) {
            const float c = Elem<T>::to_f32(normed[i]) - mean;
            spread += c * c;
        }
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(spread, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    } else {
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    }

    for (int i = tid; i < width; i += BLOCK) {
        float c = (Elem<T>::to_f32(normed[i]) - mean) * inv;
        if constexpr (NORM == kNormRmsWeight) c *= Elem<T>::to_f32(weight[i]);
        orow[i] = Elem<T>::from_f32(
            fmaf(c, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width])));
    }
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void norm_modulate(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, false>(
        x, nullptr, nullptr, nullptr, weight, m, lane_of_row, o, width, m_width,
        norm_eps, win);
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void gated_residual_norm_modulate(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ r_out,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, true>(
        r, g, y, r_out, weight, m, lane_of_row, o, width, m_width, norm_eps, win);
}

}
