#pragma once

// **THE VOXEL-AXIS RESHAPES: NO ARITHMETIC, ONLY ADDRESSES.** Each kernel
// is one thread per output element: find the output row's lane in the
// output table, unravel its `(t, h, w)`, name the input voxel and channel
// under the input table, copy. A row no lane claims lands zero.
//
// **CHANNEL ORDER OF THE SHUFFLES** is einops
// `'b (c r1 r2 r3) t h w -> b c (t r1) (h r2) (w r3)'`, which is
// `torch.pixel_shuffle`'s in two dimensions and the DiT patchify's
// `'b c (h ph) (w pw) -> b (h w) (c ph pw)'` inverted: the block offsets
// `(i1, i2, i3)` are the fast index under the channel,
// `c_in = c * (r1*r2*r3) + (i1 * r2 + i2) * r3 + i3`.

#include "prelude/device.cuh"
#include "spatial/grid.cuh"

namespace pie::spatial {

/// Nearest-neighbour upsample by `(ft, fh, fw)`. `keep_first` is the causal
/// video VAEs' time rule: frame 0 is emitted once and every later frame
/// `ft` times, so `t_out = 1 + (t - 1) * ft`; otherwise `t_out = t * ft`.
template <class T>
__global__ __launch_bounds__(256) void upsample_nearest(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int ft,
    int fh,
    int fw,
    int keep_first,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int ti = keep_first ? (o.t == 0 ? 0 : (o.t - 1) / ft + 1) : o.t / ft;
    const int src = ravel(ig, ti, o.h / fh, o.w / fw);
    y[e] = x[static_cast<long long>(src) * c + col];
}

/// Depth to space: `[rows, C * r1*r2*r3]` over `(t, h, w)` to `[rows *
/// r1*r2*r3, C]` over `(t*r1, h*r2, w*r3)`. `c` is the OUTPUT width.
template <class T>
__global__ __launch_bounds__(256) void pixel_shuffle(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int r1,
    int r2,
    int r3,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int src = ravel(ig, o.t / r1, o.h / r2, o.w / r3);
    const int block = ((o.t % r1) * r2 + (o.h % r2)) * r3 + (o.w % r3);
    const int c_in = c * (r1 * r2 * r3);
    y[e] = x[static_cast<long long>(src) * c_in + col * (r1 * r2 * r3) + block];
}

/// Space to depth, the inverse: `[rows, C]` over `(t, h, w)` to `[rows /
/// (r1*r2*r3), C * r1*r2*r3]` over `(t/r1, h/r2, w/r3)`. `c` is the INPUT
/// width.
template <class T>
__global__ __launch_bounds__(256) void pixel_unshuffle(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int r1,
    int r2,
    int r3,
    long long total)
{
    const int r = r1 * r2 * r3;
    const int c_out = c * r;
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c_out);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c_out);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int cin = col / r;
    const int block = col - cin * r;
    const int i1 = block / (r2 * r3);
    const int i2 = (block / r3) % r2;
    const int i3 = block % r3;
    const int src = ravel(ig, o.t * r1 + i1, o.h * r2 + i2, o.w * r3 + i3);
    y[e] = x[static_cast<long long>(src) * c + cin];
}

}
