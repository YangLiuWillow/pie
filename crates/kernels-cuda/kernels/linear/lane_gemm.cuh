#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

/// **THE LANE-AXIS PROJECTION** (`.wiki/imagegen/design.md` D6): `y = act ·
/// w^T` for the few rows a lane chain has — one per request — where `act`
/// stays f32 (the timestep embedding, its `silu`) and the tensor-core gemm's
/// bf16 activation contract does not apply. `[rows, k] × [n, k]^T → [rows,
/// n]`, every element f32-accumulated with `fmaf` and rounded once at the
/// store, so a host `mul_add` chain over the same operands agrees.
///
/// One warp per `(row, column)`: the lanes stride over `k`, coalesced on
/// both the activation row and the weight row, and the partials shuffle down
/// to lane zero. The grid is `(ceil(n / warps), rows)`, a plain function of
/// the lane count, and nothing here reads a seat: a lane-shaped launch grids
/// at the fire's lane carve and computes every lane the carve admits.
template <class TA, class TW, class TY>
__global__ void lane_gemm(
    const TA* __restrict__ act,
    const TW* __restrict__ w,
    TY* __restrict__ y,
    int rows,
    int n,
    int k)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) + warp;
    const int r = static_cast<int>(blockIdx.y);
    if (c >= n || r >= rows) return;
    const TA* a = act + static_cast<long long>(r) * k;
    const TW* ww = w + static_cast<long long>(c) * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) {
        acc = fmaf(Elem<TA>::to_f32(a[i]), Elem<TW>::to_f32(ww[i]), acc);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[static_cast<long long>(r) * n + c] = Elem<TY>::from_f32(acc);
    }
}

}
