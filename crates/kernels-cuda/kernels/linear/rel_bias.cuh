#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

// The learned relative-position profile (Inkling's `rel_logits_proj`): for
// every (row, head), `d_rel` features mixed through a `[d_rel, extent]` bank
// into one bias per backward distance, landed f32 as
// `bias[(row * heads + h) * extent + d]`. One thread per distance; the
// `d_rel` features of a (row, head) are re-read by every thread of its
// block, which the cache absorbs (sixteen bf16 values).
template <class T>
__global__ void rel_bias(
    const T* __restrict__ r,
    const T* __restrict__ proj,
    float* __restrict__ bias,
    int rows, int heads, int d_rel, int extent,
    const u32* __restrict__ win)
{
    const int row = blockIdx.z;
    // The staged-geometry seat: padded rows retire off the fire's live count,
    // and `win[1]` is where the live rows start on the planes.
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    if (row >= rows) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int h = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads || d >= extent) return;
    const T* rr = r + (static_cast<long long>(plane_row) * heads + h) * d_rel;
    float acc = 0.f;
    for (int j = 0; j < d_rel; ++j) {
        acc += Elem<T>::to_f32(rr[j]) *
               Elem<T>::to_f32(proj[static_cast<long long>(j) * extent + d]);
    }
    bias[(static_cast<long long>(plane_row) * heads + h) * extent + d] = acc;
}

}  // namespace pie::linear
