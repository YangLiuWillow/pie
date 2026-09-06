#pragma once

// **THE OUTPUT GRID OF ONE OP, ON THE DEVICE.** A `[lanes, 4]` table in,
// the same out: each clip's box mapped by one rule (a convolution's
// `(n + front + back - k) / s + 1`, a nearest upsample's factors, a
// shuffle's or unshuffle's block), with the row offsets prefix-summed in
// clip order so the output rectangle is packed the way the input was. One
// block, one thread: a fire carries hundreds of clips at most, and the
// prefix sum is the whole work. A box the rule cannot map (smaller than
// the kernel, not divisible by the block) lands `{0, 0, 0, off}` — no
// rows, so nothing downstream reads it.

#include "prelude/device.cuh"

namespace pie::spatial {

/// `GridRule`, flattened: `kind` 0 conv (`a` k, `b` stride, `c` pad, `flag`
/// causal_t), 1 upsample (`a` factor, `flag` keep_first_frame), 2 shuffle
/// (`a` r), 3 unshuffle (`a` r).
struct RuleGeom {
    int kind;
    int a0, a1, a2;
    int b0, b1, b2;
    int c0, c1, c2;
    int flag;
    int lanes;
};

__device__ __forceinline__ int conv_axis(int n, int k, int s, int front, int back, bool& ok) {
    const int span = n + front + back - k;
    if (span < 0) { ok = false; return 0; }
    return span / (s > 0 ? s : 1) + 1;
}

__global__ void grid_rule(const int* __restrict__ grid, int* __restrict__ out, RuleGeom g)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long off = 0;
    for (int l = 0; l < g.lanes; ++l) {
        const int t = grid[4 * l], h = grid[4 * l + 1], w = grid[4 * l + 2];
        int ot = 0, oh = 0, ow = 0;
        bool ok = true;
        switch (g.kind) {
            case 0: {
                const int back_t = g.flag ? 0 : g.c0;
                ot = conv_axis(t, g.a0, g.b0, g.c0, back_t, ok);
                oh = conv_axis(h, g.a1, g.b1, g.c1, g.c1, ok);
                ow = conv_axis(w, g.a2, g.b2, g.c2, g.c2, ok);
                break;
            }
            case 1:
                ot = (g.flag && t > 0) ? 1 + (t - 1) * g.a0 : t * g.a0;
                oh = h * g.a1;
                ow = w * g.a2;
                break;
            case 2:
                ot = t * g.a0; oh = h * g.a1; ow = w * g.a2;
                break;
            default:
                ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
                    && t % g.a0 == 0 && h % g.a1 == 0 && w % g.a2 == 0;
                if (ok) { ot = t / g.a0; oh = h / g.a1; ow = w / g.a2; }
                break;
        }
        if (!ok) { ot = 0; oh = 0; ow = 0; }
        out[4 * l] = ot;
        out[4 * l + 1] = oh;
        out[4 * l + 2] = ow;
        out[4 * l + 3] = static_cast<int>(off);
        off += static_cast<long long>(ot) * oh * ow;
    }
}

}
