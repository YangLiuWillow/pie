#pragma once

#include "prelude/device.cuh"

// **THE VOXEL AXIS, AS EVERY SPATIAL KERNEL READS IT.** An activation is
// `[rows, channels]` row-major; a row is one voxel; the voxels of one lane
// (an image or a clip) are one contiguous row range in `(t, h, w)` order
// with `w` fastest. The per-lane box lives in an `i32` table `grid[lanes][4]
// = {t, h, w, row_offset}`. Nothing here knows what a channel means.

namespace pie::spatial {

/// One lane's box and where its rows start.
struct Lane {
    int t;
    int h;
    int w;
    int off;

    __device__ __forceinline__ int voxels() const { return t * h * w; }
    __device__ __forceinline__ int plane() const { return h * w; }
};

__device__ __forceinline__ Lane lane_at(const int* __restrict__ grid, int l) {
    const int* row = grid + 4 * l;
    return Lane{__ldg(row), __ldg(row + 1), __ldg(row + 2), __ldg(row + 3)};
}

/// The lane whose row range holds `row`, its box in `out`; `-1` when no lane
/// claims the row (a padded row past the live voxels).
__device__ __forceinline__ int lane_of(const int* __restrict__ grid, int lanes, int row, Lane& out) {
    for (int l = 0; l < lanes; ++l) {
        const Lane g = lane_at(grid, l);
        const int local = row - g.off;
        if (local >= 0 && local < g.voxels()) {
            out = g;
            return l;
        }
    }
    return -1;
}

/// The row range `[begin, end)` of the ATTENTION BLOCK a query row sits
/// in. `seg_frames` is `0` for one block per lane (the image VAEs' mid
/// block) and `n > 0` for one block per run of `n` frames inside the lane
/// (Wan 2.2's mid block attends one frame at a time, `n = 1`); a lane whose
/// frame count is not a multiple leaves a short run at the end. `local` is
/// the row's offset inside the lane.
__device__ __forceinline__ void segment_of(
    const Lane& g, int local, int seg_frames, int& begin, int& end)
{
    if (seg_frames <= 0) {
        begin = g.off;
        end = g.off + g.voxels();
        return;
    }
    const int plane = g.plane();
    const int frame = plane > 0 ? local / plane : 0;
    const int first = frame / seg_frames * seg_frames;
    const int last = min(first + seg_frames, g.t);
    begin = g.off + first * plane;
    end = g.off + last * plane;
}

/// A voxel's `(t, h, w)` inside its lane.
struct Voxel {
    int t;
    int h;
    int w;
};

__device__ __forceinline__ Voxel unravel(const Lane& g, int local) {
    const int w = local % g.w;
    const int rest = local / g.w;
    return Voxel{rest / g.h, rest % g.h, w};
}

__device__ __forceinline__ int ravel(const Lane& g, int t, int h, int w) {
    return g.off + (t * g.h + h) * g.w + w;
}

}
