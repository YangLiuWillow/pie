#pragma once

// **IMPLICIT-GEMM CONVOLUTION OVER THE VOXEL AXIS.** `o[m][n] = bias[n] +
// sum_{tap, c} x[in_row(m, tap)][c] * w[n][tap * C_in + c]` — a GEMM whose
// A operand (`[rows_out, K]`, `K = kt*kh*kw*C_in`) is never materialised:
// each K-step gathers the `BK` channels of one tap's input voxel per output
// row straight into shared memory. Two kernels compute the same thing:
//
//   `conv3d_direct` — fp32 FMA tiles, 64 voxels x 64 channels a block, every
//   channel count and alignment (the scalar-gather instantiation), the
//   reference the tensor-core kernel is measured against;
//   `conv3d_mma`    — `mma.sync.m16n8k16` bf16 tiles with fp32 accumulators,
//   128 x 128 a block, `cp.async` gathers three stages deep; asks for
//   `C_in % 8 == 0` and 16-byte-aligned planes.
//
// **THE WEIGHT IS TAP-MAJOR, CHANNEL-FASTEST.** `w` is `[C_out, kt*kh*kw*C_in]`
// with `K = ((it*kh + ih)*kw + iw)*C_in + c_in` — PyTorch's
// `weight.permute(0, 2, 3, 4, 1).reshape(C_out, -1)`, cuDNN's KRSC. The
// natural `[C_out, C_in, kt, kh, kw]` flattening would put the `BK`
// channels of one tap `kt*kh*kw` elements apart: every B tile a strided
// two-byte gather, which is what `conv_weight_taps_major` exists to undo
// once at load.
//
// **TIME.** Symmetric: `pt` frames each side, zeros or — under `replicate`
// — the clip's own first frame in front and its last frame behind (LTX-2.5's
// non-causal decoder pads that way). Causal: `pt` frames in front and none
// behind; a frame before the clip reads `cache` (the previous tile's last
// `pt` frames, per lane `[pt * h * w, C_in]` rows in lane order) when one is
// given, else frame 0 (`replicate`) or zero. `h`/`w` are always symmetric
// zero padding. Output rows no lane claims land zeros.
//
// Numerics: bf16 in, fp32 accumulate over all of K, bias added in fp32, one
// rounding at the store. The two kernels sum K in different orders and agree
// to fp32 rounding, not bit-for-bit.

#include "prelude/device.cuh"
#include "prelude/mma.cuh"
#include "spatial/grid.cuh"

#include <cuda_pipeline.h>

namespace pie::spatial {

/// The static shape of one convolution, one launch argument by value.
/// Mirrors `spatial::conv::Geom` in the Rust entry field for field.
struct ConvGeom {
    int c_in;
    int c_out;
    int kt;
    int kh;
    int kw;
    int st;
    int sh;
    int sw;
    /// Front time pad (causal) or the symmetric time pad.
    int pt;
    int ph;
    int pw;
    /// 1: time pad in front only; `cache` frames stand in for it.
    int causal;
    /// 1: a padded frame reads the clip's own end frame instead of zeros —
    /// frame 0 in front (with no cache), and, when not causal, the last
    /// frame behind.
    int replicate;
    int lanes;
    int rows_out;
};

/// One output row's view of its lane: the input box, the output voxel, and
/// where this lane's cache frames start.
struct OutRow {
    Lane in;
    Voxel o;
    int cache_base;
    bool live;
};

__device__ __forceinline__ OutRow out_row(
    const ConvGeom& g,
    const int* __restrict__ grid,
    const int* __restrict__ o_grid,
    int m)
{
    OutRow r;
    r.live = false;
    r.cache_base = 0;
    if (m >= g.rows_out) return r;
    Lane og;
    const int l = lane_of(o_grid, g.lanes, m, og);
    if (l < 0) return r;
    r.in = lane_at(grid, l);
    r.o = unravel(og, m - og.off);
    // The cache is `pt` frames per lane, lane after lane, each in its own
    // lane's plane size.
    int planes = 0;
    for (int j = 0; j < l; ++j) planes += lane_at(grid, j).plane();
    r.cache_base = planes * g.pt;
    r.live = true;
    return r;
}

/// The row one tap of one output voxel reads: a row of `x` (`from_cache`
/// false), a row of `cache` (true), or `-1` for a zero.
__device__ __forceinline__ int tap_row(
    const ConvGeom& g,
    const OutRow& r,
    int it,
    int ih,
    int iw,
    bool has_cache,
    bool& from_cache)
{
    from_cache = false;
    const int hi = r.o.h * g.sh - g.ph + ih;
    const int wi = r.o.w * g.sw - g.pw + iw;
    if (hi < 0 || hi >= r.in.h || wi < 0 || wi >= r.in.w) return -1;
    int ti = r.o.t * g.st - g.pt + it;
    if (ti < 0) {
        if (g.causal && has_cache) {
            from_cache = true;
            return r.cache_base + ((ti + g.pt) * r.in.h + hi) * r.in.w + wi;
        }
        if (!g.replicate) return -1;
        ti = 0;
    }
    if (ti >= r.in.t) {
        // Behind the clip: a symmetric replicating convolution reads the
        // last frame; a causal one never pads behind, and a zero-padded
        // symmetric one reads zeros.
        if (g.causal || !g.replicate) return -1;
        ti = r.in.t - 1;
    }
    return ravel(r.in, ti, hi, wi);
}

/// The `(it, ih, iw)` of tap index `tap` in the weight's tap-major order.
__device__ __forceinline__ void tap_of(const ConvGeom& g, int tap, int& it, int& ih, int& iw) {
    const int plane = g.kh * g.kw;
    it = tap / plane;
    const int rest = tap - it * plane;
    ih = rest / g.kw;
    iw = rest - ih * g.kw;
}

// ---------------------------------------------------------------------------
// Version A: fp32 FMA tiles.
// ---------------------------------------------------------------------------

constexpr int kDirectTM = 64;
constexpr int kDirectTN = 64;
constexpr int kDirectBK = 32;
constexpr int kDirectThreads = 256;

/// Eight consecutive channels of one row into `out`, zero past `c_in`.
/// `VEC` reads them as one 16-byte word (the caller guaranteed `c_in % 8 ==
/// 0` and alignment); otherwise element by element under a channel guard.
template <bool VEC>
__device__ __forceinline__ void gather8(const bf16* __restrict__ row, int c, int c_in, float (&out)[8]) {
    if constexpr (VEC) {
        if (c < c_in) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(row + c));
            const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[2 * j] = bf16_to_f32(bf16{static_cast<unsigned short>(words[j] & 0xffffu)});
                out[2 * j + 1] = bf16_to_f32(bf16{static_cast<unsigned short>(words[j] >> 16)});
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (c + j < c_in) out[j] = bf16_to_f32(ldg(row + c + j));
        }
    }
}

template <bool VEC>
__global__ __launch_bounds__(kDirectThreads) void conv3d_direct(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    const bf16* __restrict__ w,
    const float* __restrict__ bias,
    const bf16* __restrict__ cache,
    bf16* __restrict__ o,
    const int* __restrict__ o_grid,
    ConvGeom g)
{
    __shared__ __align__(16) float As[kDirectBK][kDirectTM];
    __shared__ __align__(16) float Bs[kDirectBK][kDirectTN];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * kDirectTM;
    const int n0 = blockIdx.y * kDirectTN;

    // Staging map: a warp stages 32 consecutive rows (of A) and 32
    // consecutive weight rows (of B), one 8-wide channel chunk each, so the
    // transposed shared-memory writes land on 32 distinct banks.
    const int s_row = tid % 64;
    const int s_k = (tid / 64) * 8;
    const OutRow r = out_row(g, grid, o_grid, m0 + s_row);
    const bool has_cache = cache != nullptr;
    const int taps = g.kt * g.kh * g.kw;
    const int cchunks = (g.c_in + kDirectBK - 1) / kDirectBK;
    const int steps = taps * cchunks;
    const bool n_live = (n0 + s_row) < g.c_out;
    const bf16* wrow = w + static_cast<long long>(n_live ? n0 + s_row : 0) * taps * g.c_in;

    // Compute map: 16 x 16 threads, four rows by four columns each.
    const int ty = tid / 16;
    const int tx = tid % 16;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    }

    int src_row = -1;
    bool from_cache = false;
    for (int s = 0; s < steps; ++s) {
        const int tap = s / cchunks;
        const int c0 = (s - tap * cchunks) * kDirectBK;
        if (c0 == 0 && r.live) {
            int it, ih, iw;
            tap_of(g, tap, it, ih, iw);
            src_row = tap_row(g, r, it, ih, iw, has_cache, from_cache);
        }

        float a[8];
        float b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            a[j] = 0.f;
            b[j] = 0.f;
        }
        if (r.live && src_row >= 0) {
            const bf16* row = (from_cache ? cache : x) + static_cast<long long>(src_row) * g.c_in;
            gather8<VEC>(row, c0 + s_k, g.c_in, a);
        }
        if (n_live) {
            gather8<VEC>(wrow + static_cast<long long>(tap) * g.c_in, c0 + s_k, g.c_in, b);
        }

        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            As[s_k + j][s_row] = a[j];
            Bs[s_k + j][s_row] = b[j];
        }
        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < kDirectBK; ++k) {
            const float4 av = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            const float ar[4] = {av.x, av.y, av.z, av.w};
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= g.rows_out) continue;
        Lane box;
        const bool live = lane_of(o_grid, g.lanes, m, box) >= 0;
        bf16* orow = o + static_cast<long long>(m) * g.c_out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= g.c_out) continue;
            float v = 0.f;
            if (live) {
                v = acc[i][j];
                if (bias != nullptr) v += __ldg(bias + n);
            }
            orow[n] = f32_to_bf16(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Version B: bf16 tensor-core tiles.
// ---------------------------------------------------------------------------

constexpr int kMmaTM = 128;
constexpr int kMmaTN = 128;
constexpr int kMmaBK = 32;
constexpr int kMmaThreads = 256;
/// The staged row pitch in bf16: eight of padding so the eight rows of one
/// `ldmatrix` phase land in eight different 16-byte bank groups.
constexpr int kMmaLd = kMmaBK + 8;
constexpr int kMmaStageElems = (kMmaTM + kMmaTN) * kMmaLd;
/// The epilogue tile's row pitch (bf16), padded the same way.
constexpr int kMmaLdC = kMmaTN + 8;

/// Dynamic shared memory one launch of `conv3d_mma<kStages>` asks for.
template <int kStages>
__host__ __device__ constexpr int conv3d_mma_smem() {
    constexpr int ring = kStages * kMmaStageElems * 2;
    constexpr int tile = kMmaTM * kMmaLdC * 2;
    return ring > tile ? ring : tile;
}

/// One 16-byte `cp.async`, zero-filled when `bytes` is 0 (the gather's
/// out-of-range taps): `src` must still be a valid address, so the caller
/// hands the plane base.
__device__ __forceinline__ void conv_cp_async16(void* dst, const void* src, int bytes) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(bytes)
                 : "memory");
}

/// `ldmatrix.x4` over a row-major 16x16 bf16 tile: lane `l` addresses row
/// `l & 15`, column `(l >> 4) * 8`; the registers come back as the (rows
/// 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15)
/// 8x8 tiles — the mma A fragment as is, and off a `[n][k]` weight tile the
/// B fragments of two 8-row subtiles as {r0, r2} and {r1, r3}.
__device__ __forceinline__ void conv_ldmatrix_x4(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

template <int kStages>
__global__ __launch_bounds__(kMmaThreads) void conv3d_mma(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    const bf16* __restrict__ w,
    const float* __restrict__ bias,
    const bf16* __restrict__ cache,
    bf16* __restrict__ o,
    const int* __restrict__ o_grid,
    ConvGeom g)
{
    static_assert(kStages >= 2, "one stage in flight while one is multiplied, at least");
    extern __shared__ __align__(16) unsigned char conv_smem[];
    bf16* ring = reinterpret_cast<bf16*>(conv_smem);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m0 = blockIdx.x * kMmaTM;
    const int n0 = blockIdx.y * kMmaTN;
    const bool has_cache = cache != nullptr;
    const int taps = g.kt * g.kh * g.kw;
    const int cchunks = (g.c_in + kMmaBK - 1) / kMmaBK;
    const int steps = taps * cchunks;
    const long long wpitch = static_cast<long long>(taps) * g.c_in;

    // Staging map: chunk `tid + i * 256` is row `chunk / 4`, channels
    // `(chunk % 4) * 8`; two A rows and two B rows a thread, fixed for the
    // whole K loop, so the voxel arithmetic is done once and the tap
    // arithmetic once per tap.
    const int s_kc = (tid & 3) * 8;
    OutRow rows[2];
    int src_row[2] = {-1, -1};
    bool from_cache[2] = {false, false};
    const bf16* wrows[2];
    bool n_live[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int s_row = (tid >> 2) + 64 * i;
        rows[i] = out_row(g, grid, o_grid, m0 + s_row);
        n_live[i] = (n0 + s_row) < g.c_out;
        wrows[i] = w + static_cast<long long>(n_live[i] ? n0 + s_row : 0) * wpitch;
    }

    auto stage = [&](int buf, int step) {
        const int tap = step / cchunks;
        const int c0 = (step - tap * cchunks) * kMmaBK;
        if (c0 == 0) {
            int it, ih, iw;
            tap_of(g, tap, it, ih, iw);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                src_row[i] = rows[i].live ? tap_row(g, rows[i], it, ih, iw, has_cache, from_cache[i]) : -1;
            }
        }
        bf16* adst = ring + buf * kMmaStageElems;
        bf16* bdst = adst + kMmaTM * kMmaLd;
        const int c = c0 + s_kc;
        const bool k_live = c < g.c_in;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int s_row = (tid >> 2) + 64 * i;
            const bool a_live = k_live && src_row[i] >= 0;
            const bf16* asrc = a_live
                ? (from_cache[i] ? cache : x) + static_cast<long long>(src_row[i]) * g.c_in + c
                : x;
            conv_cp_async16(adst + s_row * kMmaLd + s_kc, asrc, a_live ? 16 : 0);
            const bool b_live = k_live && n_live[i];
            const bf16* bsrc = b_live ? wrows[i] + static_cast<long long>(tap) * g.c_in + c : w;
            conv_cp_async16(bdst + s_row * kMmaLd + s_kc, bsrc, b_live ? 16 : 0);
        }
        __pipeline_commit();
    };

    // Warp tile: 2 (rows) x 4 (columns) warps, 64 x 32 each — four m16
    // tiles by four n8 tiles.
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
        }
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s);
        } else {
            __pipeline_commit();
        }
    }

    const int frag_row = lane & 15;
    const int frag_col = (lane >> 4) << 3;

    for (int step = 0; step < steps; ++step) {
        const int fetch = step + kStages - 1;
        if (fetch < steps) {
            stage(fetch % kStages, fetch);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const bf16* asrc = ring + (step % kStages) * kMmaStageElems + (warp_m * 64 + frag_row) * kMmaLd + frag_col;
        const bf16* bsrc = ring + (step % kStages) * kMmaStageElems + kMmaTM * kMmaLd + (warp_n * 32 + frag_row) * kMmaLd + frag_col;
#pragma unroll
        for (int kk = 0; kk < kMmaBK / 16; ++kk) {
            unsigned a[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                conv_ldmatrix_x4(a[i], asrc + i * 16 * kMmaLd + kk * 16);
            }
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                unsigned b[4];
                conv_ldmatrix_x4(b, bsrc + pair * 16 * kMmaLd + kk * 16);
                const unsigned b_lo[2] = {b[0], b[2]};
                const unsigned b_hi[2] = {b[1], b[3]};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ::nvcuda::wmma::detail::mma_m16n8k16(acc[i][2 * pair], a[i], b_lo, acc[i][2 * pair]);
                    ::nvcuda::wmma::detail::mma_m16n8k16(acc[i][2 * pair + 1], a[i], b_hi, acc[i][2 * pair + 1]);
                }
            }
        }
        __syncthreads();
    }

    // **THE EPILOGUE GOES THROUGH SHARED MEMORY.** One lane's accumulator
    // holds two rows (`g`, `g + 8`) by two adjacent columns; staged as a
    // `[row][col]` bf16 tile it drains as whole 16-byte row pieces. The bias
    // joins in fp32 before the one rounding.
    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = ring;
    {
        const int gq = lane >> 2;
        const int tq = lane & 3;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = warp_n * 32 + j * 8 + 2 * tq;
                const int n = n0 + col;
                const float b0 = (bias != nullptr && n < g.c_out) ? __ldg(bias + n) : 0.f;
                const float b1 = (bias != nullptr && n + 1 < g.c_out) ? __ldg(bias + n + 1) : 0.f;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = warp_m * 64 + i * 16 + gq + 8 * half;
                    *reinterpret_cast<unsigned*>(c_tile + row * kMmaLdC + col) =
                        pack_bf16x2(acc[i][j][2 * half] + b0, acc[i][j][2 * half + 1] + b1);
                }
            }
        }
    }
    __syncthreads();

    constexpr int kChunksOfRow = kMmaTN / 8;
    constexpr int kDrain = kMmaTM * kChunksOfRow;
    const bool whole_chunks = (g.c_out & 7) == 0;
#pragma unroll
    for (int i = 0; i < kDrain / kMmaThreads; ++i) {
        const int chunk = tid + i * kMmaThreads;
        const int r = chunk / kChunksOfRow;
        const int c = (chunk % kChunksOfRow) * 8;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m >= g.rows_out || n >= g.c_out) continue;
        Lane box;
        const bool live = lane_of(o_grid, g.lanes, m, box) >= 0;
        bf16* dst = o + static_cast<long long>(m) * g.c_out + n;
        const bf16* src = c_tile + r * kMmaLdC + c;
        if (whole_chunks) {
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (live) v = *reinterpret_cast<const uint4*>(src);
            *reinterpret_cast<uint4*>(dst) = v;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (n + j < g.c_out) dst[j] = live ? src[j] : bf16{static_cast<unsigned short>(0)};
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The load-time weight relabelling.
// ---------------------------------------------------------------------------

/// `dst[n][tap * c_in + c] = src[n][c * taps + tap]`: PyTorch's
/// `[C_out, C_in, kt, kh, kw]` rectangle, flattened as stored, into the
/// tap-major channel-fastest order the two kernels read. One thread per
/// element of the destination.
__global__ void conv_weight_taps_major(
    const bf16* __restrict__ src,
    bf16* __restrict__ dst,
    int c_out,
    int c_in,
    int taps)
{
    const long long total = static_cast<long long>(c_out) * c_in * taps;
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const long long pitch = static_cast<long long>(c_in) * taps;
    const int n = static_cast<int>(e / pitch);
    const int k = static_cast<int>(e - n * pitch);
    const int tap = k / c_in;
    const int c = k - tap * c_in;
    dst[e] = src[n * pitch + static_cast<long long>(c) * taps + tap];
}

}
