#pragma once

// **THE CONV VAE'S MID-BLOCK ATTENTION: ONE HEAD AS WIDE AS THE ROW, PER
// BLOCK.** `y = softmax(q·kᵀ · scale) · v` over `[rows, C]` bf16 rectangles
// on the voxel axis, every query attending the voxels of its own block and
// nothing else. A block is a whole lane (`seg_frames == 0`, the image
// VAEs) or a run of `seg_frames` consecutive frames inside one (Wan 2.2's
// mid block attends one frame at a time); the lane table says where a
// lane's rows are and `segment_of` cuts the run out of it. Not the ragged
// FA2 template: that is stamped at head widths
// 64/128/256, and a VAE's head is its whole channel row (512 on the FLUX
// VAE) at its lowest resolution (a few thousand voxels), where a plain
// online-softmax walk over the keys is enough.
//
// **THE WALK.** One warp owns `QPW` consecutive query rows; lane `l` of the
// warp holds channels `[l·CPL, (l+1)·CPL)` of each query in fp32. For every
// key of the queries' block the warp reads the key row once (each lane its
// CPL channels as 16-byte words), reduces the `QPW` partial dots with
// shuffles, updates each query's running max and sum (fp32, `exp2` on
// pre-scaled logits), rescales the accumulators and folds the value row in.
// A group of `QPW` rows that straddles two blocks is walked once per block
// with the other rows masked. Rows no lane claims land zeros. fp32 scores,
// fp32 softmax and accumulation, one rounding at the store.
//
// **REGISTERS.** `qreg` and `acc` are `QPW · C/32` fp32 each, so the caller
// picks `QPW` against `C`: four queries a warp at C 256, two at 512, one at
// 640 and 1024 — at most 64 registers of state either way, which is what
// keeps the 1024-wide head (Wan's decoder) off the local-memory spill path.
//
// **THE VECTOR PATH AND THE SCALAR ONE.** A lane reads its `CPL = C/32`
// channels as 16-byte words when `CPL` divides by 8 (C 256, 512, 1024).
// Wan 2.2's ENCODER mid block is 640 wide: `CPL` is 20, so a lane's slice
// is neither a whole number of words nor 16-byte aligned, and that
// instantiation reads and writes bf16 scalars instead. The arithmetic is
// the same and the fast widths keep the same instructions they had.

#include "prelude/device.cuh"
#include "spatial/grid.cuh"

namespace pie::spatial {

namespace detail {

/// Eight bf16 of one 16-byte word into fp32.
__device__ __forceinline__ void unpack8(uint4 word, float out[8]) {
    const unsigned int u[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = __uint_as_float(u[i] << 16);
        out[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

}  // namespace detail

/// `C` channels per row (`C % 32 == 0`; a lane's `CPL = C / 32` channels
/// move as 16-byte words when `CPL % 8 == 0` and as scalars otherwise);
/// `QPW` queries per warp; four warps a block. `scale_log2` is
/// `sm_scale · log2(e)`.
template <int C, int QPW>
__global__ __launch_bounds__(128) void attention(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const int* __restrict__ grid,
    bf16* __restrict__ y,
    int lanes,
    int rows,
    int seg_frames,
    float scale_log2)
{
    constexpr int CPL = C / 32;
    constexpr bool VEC = (CPL % 8 == 0);
    constexpr int WORDS = CPL / 8;
    static_assert(C % 32 == 0, "every lane holds a whole slice of the row");

    // One row's `CPL` channels into fp32, vector or scalar.
    auto read_slice = [](const bf16* __restrict__ base, int row, int col, float (&out)[CPL]) {
        const bf16* at = base + static_cast<size_t>(row) * C + col;
        if constexpr (VEC) {
            const uint4* src = reinterpret_cast<const uint4*>(at);
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                float tmp[8];
                detail::unpack8(__ldg(src + w), tmp);
#pragma unroll
                for (int e = 0; e < 8; ++e) out[w * 8 + e] = tmp[e];
            }
        } else {
#pragma unroll
            for (int e = 0; e < CPL; ++e) out[e] = bf16_to_f32(ldg(at + e));
        }
    };

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * 4 + warp) * QPW;
    if (first >= rows) return;
    const int col = lane * CPL;

    // Which lane (clip) each query row belongs to, and which rows of it
    // its attention block spans.
    int owner[QPW];
    int seg_begin[QPW], seg_end[QPW];
    Lane box[QPW];
    float qreg[QPW][CPL];
#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        const int r = first + i;
        owner[i] = r < rows ? lane_of(grid, lanes, r, box[i]) : -2;
        seg_begin[i] = 0;
        seg_end[i] = 0;
        if (owner[i] >= 0) {
            segment_of(box[i], r - box[i].off, seg_frames, seg_begin[i], seg_end[i]);
            read_slice(q, r, col, qreg[i]);
#pragma unroll
            for (int e = 0; e < CPL; ++e) qreg[i][e] *= scale_log2;
        } else {
#pragma unroll
            for (int c = 0; c < CPL; ++c) qreg[i][c] = 0.f;
        }
    }

    float acc[QPW][CPL];
    float m[QPW], l[QPW];
#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        m[i] = __int_as_float(0xff800000);  // -inf
        l[i] = 0.f;
#pragma unroll
        for (int c = 0; c < CPL; ++c) acc[i][c] = 0.f;
    }

    // Walk each distinct BLOCK among the group's rows once. Blocks are
    // disjoint contiguous row ranges, so the begin row names one.
    bool done[QPW];
#pragma unroll
    for (int i = 0; i < QPW; ++i) done[i] = owner[i] < 0;
    for (;;) {
        int pick = -1;
#pragma unroll
        for (int i = 0; i < QPW; ++i) if (pick < 0 && !done[i]) pick = i;
        if (pick < 0) break;
        const int who = seg_begin[pick];
        bool active[QPW];
#pragma unroll
        for (int i = 0; i < QPW; ++i) {
            active[i] = !done[i] && seg_begin[i] == who;
            if (active[i]) done[i] = true;
        }
        const int begin = who;
        const int end = seg_end[pick];
        for (int j = begin; j < end; ++j) {
            float kk[CPL];
            read_slice(k, j, col, kk);
            float s[QPW];
#pragma unroll
            for (int i = 0; i < QPW; ++i) s[i] = 0.f;
#pragma unroll
            for (int e = 0; e < CPL; ++e) {
#pragma unroll
                for (int i = 0; i < QPW; ++i) s[i] = fmaf(qreg[i][e], kk[e], s[i]);
            }
            float p[QPW];
#pragma unroll
            for (int i = 0; i < QPW; ++i) {
                s[i] = detail::warp_sum(s[i]);
                if (active[i]) {
                    const float m_new = fmaxf(m[i], s[i]);
                    const float rescale = exp2f(m[i] - m_new);
                    p[i] = exp2f(s[i] - m_new);
                    l[i] = l[i] * rescale + p[i];
                    m[i] = m_new;
#pragma unroll
                    for (int c = 0; c < CPL; ++c) acc[i][c] *= rescale;
                } else {
                    p[i] = 0.f;
                }
            }
            float vv[CPL];
            read_slice(v, j, col, vv);
#pragma unroll
            for (int e = 0; e < CPL; ++e) {
#pragma unroll
                for (int i = 0; i < QPW; ++i) acc[i][e] = fmaf(p[i], vv[e], acc[i][e]);
            }
        }
    }

    // Store: the normalised accumulator for a claimed row, zeros for an
    // unclaimed one inside the rectangle, nothing past it.
#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        const int r = first + i;
        if (r >= rows) continue;
        const float inv = owner[i] >= 0 && l[i] > 0.f ? 1.f / l[i] : 0.f;
        bf16* at = y + static_cast<size_t>(r) * C + col;
        if constexpr (VEC) {
            uint4* dst = reinterpret_cast<uint4*>(at);
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                uint4 word;
                word.x = pack_bf16x2(acc[i][w * 8 + 0] * inv, acc[i][w * 8 + 1] * inv);
                word.y = pack_bf16x2(acc[i][w * 8 + 2] * inv, acc[i][w * 8 + 3] * inv);
                word.z = pack_bf16x2(acc[i][w * 8 + 4] * inv, acc[i][w * 8 + 5] * inv);
                word.w = pack_bf16x2(acc[i][w * 8 + 6] * inv, acc[i][w * 8 + 7] * inv);
                dst[w] = word;
            }
        } else {
#pragma unroll
            for (int e = 0; e < CPL; ++e) at[e] = f32_to_bf16(acc[i][e] * inv);
        }
    }
}

}  // namespace pie::spatial
