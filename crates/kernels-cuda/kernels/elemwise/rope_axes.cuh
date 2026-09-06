#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

/// **MULTI-AXIS ROTARY: UP TO FOUR AXES, EACH WITH ITS OWN THETA, THREE
/// PAIRINGS** (`.wiki/imagegen/design.md` D7).
///
/// `rope_mrope` next door turns a row by an `(t, h, w)` TRIPLE of integers
/// under ONE theta and hands the axes out by a section rule. This op is the
/// generalisation the image and video DiTs want:
///
/// - axis `a` owns `dims[a]` CONTIGUOUS rotary channels, concatenated in axis
///   order, `Σ dims = rotary_dim`; channels `[rotary_dim, head_dim)` pass
///   through untouched;
/// - each axis carries its OWN theta, and its `i`-th angle turns at
///   `theta_a^(-2i / dims[a])` — its own full ladder, computed in f32;
/// - positions are f32 and may be FRACTIONAL (LTX's physical coordinates in
///   seconds and pixels are not integers);
/// - and it applies to every head of the row, out of place (`o` may alias
///   `x`), so one call turns a whole `[rows, heads·head_dim]` rectangle.
///
/// **THE FORM IS THE PAIRING, AND NOTHING ELSE.** All three assign the same
/// angle to the same axis; they disagree only about which two channels that
/// angle rotates:
constexpr int kRopeInterleaved = 0;  ///< GPT-J: `(x[2i], x[2i+1])`.
constexpr int kRopeNeox = 1;         ///< rotate-half: `(x[p], x[p + rotary_dim/2])`.
constexpr int kRopeSplit = 2;        ///< within the axis block: `(x[b+i], x[b+s+i])`.
constexpr int kRopeSplitLadder = 3;  ///< one ladder across the row, axes round-robin.

/// **`kRopeInterleaved` IS NOT `MropeForm::Interleaved`.** That name, one
/// file over, describes how SECTIONS are handed out (pairs alternating
/// `t, h, w`) under rotate-half pairing; this one names the PAIRING, the way
/// `rope_full`'s `interleaved` flag and sglang's `is_neox=False` do. Z-Image
/// applies its 3-axis rope as `view_as_complex` pairs, which is this.
///
/// `kRopeSplit` is `MropeForm::Split` transcribed: each axis owns a
/// contiguous block of `2s` channels and pair `i` of the block is
/// `(x[b+i], x[b+s+i])` — rotate-half WITHIN the block, never across axes —
/// which is Gemma's tower and LTX's `x = [x1 | x2]` halves.
///
/// `kRopeNeox` is rotate-half across the WHOLE rotary span, the layout every
/// `torch.cat([freqs, freqs], -1)` reference produces; with one axis and
/// `rotary_dim == head_dim` it is `rope_full`'s non-interleaved arm, angle for
/// angle.
///
/// `kRopeSplitLadder` is the odd one out and the reason the form is not just
/// a pairing: LTX-2 builds ONE frequency ladder over the whole
/// `[rows, heads·head_dim]` row and hands the axes out round-robin ALONG it,
/// so head `h`'s angles are neither one axis's nor one band of the ladder.
/// Slot `g = head·angles + i` (`angles = rotary_dim/2`) is identity while
/// `g < pad`, and otherwise belongs to axis `(g − pad) mod axes` at ladder
/// index `f = (g − pad) / axes`, turning by `positions[a] ·
/// thetas[a]^(f/(F_a − 1))` — a POSITIVE, endpoint-inclusive exponent
/// (`torch.linspace(0, 1, F_a)`), `F_a = dims[a]/2` the ROW's frequency
/// count for that axis and `pad = (heads·rotary_dim − Σ dims)/2`. Its
/// pairing is rotate-half within the head, which is `kRopeNeox`'s at
/// `rotary_dim == head_dim`.
///
/// `__sincosf` and `powf` are `rope.cuh`'s, transcribed, so a one-axis call
/// answers what the scalar kernel answers to the bit. THE LADDER FORM USES
/// `sincosf` INSTEAD: its positions are pre-scaled by `π/2` and its top
/// frequency is `theta` itself, so an angle reaches ~1.6e4 radians, past
/// where the SFU's reduction holds; the reference reduces in fp32 with a
/// full-precision π, and so does this.
template <class T, int FORM>
__global__ void rope_axes(
    const T* __restrict__ x,
    const float* __restrict__ positions,
    T* __restrict__ o,
    int axes,
    int d0, int d1, int d2, int d3,
    float t0, float t1, float t2, float t3,
    int rotary_dim,
    int head_dim,
    int heads,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (`rope_mrope`'s idiom, one block per row): a
    // replay whose grid was carved at a bucket retires its padded rows here.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: `x`, `o` and the
    // `[rows, axes]` position stream are row planes handed at their base. The
    // angles are keyed by the position VALUES, never by the row.
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int dims[4] = {d0, d1, d2, d3};
    const float thetas[4] = {t0, t1, t2, t3};
    const int angles = rotary_dim / 2;
    const int width = heads * head_dim;

    const float* pos = positions + static_cast<long long>(row) * axes;
    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;

    if constexpr (FORM == kRopeSplitLadder) {
        // One ladder over the whole row. `pad` identity slots in front, then
        // axis-major round robin; the pairing is rotate-half within a head.
        int span = 0;
        for (int a = 0; a < axes; ++a) span += dims[a];
        const int pad = (heads * rotary_dim - span) / 2;
        for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
            const int head = idx / angles;
            const int angle = idx % angles;
            float cos_v = 1.f;
            float sin_v = 0.f;
            if (idx >= pad) {
                const int slot = idx - pad;
                const int axis = slot % axes;
                const int f = slot / axes;
                const int ladder = dims[axis] / 2;
                const float exponent =
                    ladder > 1 ? static_cast<float>(f) / static_cast<float>(ladder - 1) : 0.f;
                sincosf(pos[axis] * powf(thetas[axis], exponent), &sin_v, &cos_v);
            }
            const int lo = angle;
            const int hi = angle + angles;
            const T* xh = xr + static_cast<long long>(head) * head_dim;
            T* oh = orow + static_cast<long long>(head) * head_dim;
            const float a = Elem<T>::to_f32(xh[lo]);
            const float b = Elem<T>::to_f32(xh[hi]);
            oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
            oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
        return;
    }

    for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
        const int head = idx / angles;
        const int angle = idx % angles;

        // Which axis owns this angle, and where its block starts. Four steps
        // at most, and the same four for every thread of the warp.
        int axis = 0;
        int first_angle = 0;
        int first_channel = 0;
        while (axis < axes && angle >= first_angle + dims[axis] / 2) {
            first_angle += dims[axis] / 2;
            first_channel += dims[axis];
            ++axis;
        }
        // Rotary channels no axis claimed: left alone, like the tail.
        if (axis >= axes) continue;
        const int within = angle - first_angle;

        const float freq = powf(thetas[axis],
            -2.f * static_cast<float>(within) / static_cast<float>(dims[axis]));
        float cos_v, sin_v;
        __sincosf(pos[axis] * freq, &sin_v, &cos_v);

        int lo;
        int hi;
        if constexpr (FORM == kRopeInterleaved) {
            lo = first_channel + 2 * within;
            hi = lo + 1;
        } else if constexpr (FORM == kRopeNeox) {
            lo = angle;
            hi = angle + angles;
        } else {
            lo = first_channel + within;
            hi = first_channel + dims[axis] / 2 + within;
        }

        const T* xh = xr + static_cast<long long>(head) * head_dim;
        T* oh = orow + static_cast<long long>(head) * head_dim;
        const float a = Elem<T>::to_f32(xh[lo]);
        const float b = Elem<T>::to_f32(xh[hi]);
        oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
        oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
    }

    // The unrotated tail of each head. A no-op when `o` aliases `x`, and the
    // difference between a rotation and a corrupt row when it does not.
    if (o != x) {
        const int tail = head_dim - rotary_dim;
        for (int idx = threadIdx.x; idx < heads * tail; idx += blockDim.x) {
            const int head = idx / tail;
            const int i = rotary_dim + idx % tail;
            orow[head * head_dim + i] = xr[head * head_dim + i];
        }
    }
}

}
