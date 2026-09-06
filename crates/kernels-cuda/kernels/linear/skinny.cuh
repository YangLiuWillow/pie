// **THE SKINNY bf16 PROJECTION WITH ITS EPILOGUE FOLDED IN**: `y = act x
// w^T` for a few activation rows (m <= 64) against a big weight — the decode
// GEMM — finished in registers by what the trace would otherwise run as a
// second pass over the result.
//
// The kernel itself is a weight-streaming tensor-core GEMM that runs the
// decode shapes at the same rate cuBLAS does (the card's ~700 GB/s; see
// `linear/dense.rs`'s tuner, which it does NOT enter). Its point is the
// epilogue: the traced `mlp_geglu_tanh_packed` re-read the up/gate output
// (64 x 20480 bf16) and the traced `logit_softcap` re-read the logits
// (64 x 262144), ~0.3 ms a step at 64 lanes on gemma-4-E4B, and both are a
// few flops on values the GEMM already holds.
//
//   - one block owns `kWarps * 16` rows of `w` and ALL 64 activation rows
//     (rows past `m` are zeros in shared memory and never stored);
//   - each stage is a `kBK`-wide contraction step, the block's weight rows
//     and the 64 activation rows for that k range `cp.async`ed into a
//     `kStages`-deep ring;
//   - each warp holds its 16 weight rows as the mma A fragment and walks
//     the eight 8-row activation subtiles as B;
//   - the accumulator is staged through shared memory as `[act row][weight
//     col]` and drained as whole 128-byte rows, the epilogue applied on
//     the way.
//
// Epilogues (`kEpilogue`):
//   0  store:   `y[m][n]`, the plain projection;
//   1  softcap: `y = cap * tanh(y / cap)`, the head's `logit_softcap`, on
//               the bf16-rounded product exactly as the traced pass saw it;
//   2  geglu:   `w` is the packed `[2I x k]` up/gate weight, `n = I`, and a
//               block owns `kWarps * 8` GATE rows and their `kWarps * 8` UP
//               rows (`I` rows apart), so `y[m][I] = gelu_tanh(gate) * up`
//               is a whole-row product in the epilogue, again off the bf16
//               rounded halves so it lands what `mlp_geglu_tanh_packed`
//               lands.
//
// Layouts: `w` row-major `[n][k]` (`[2I][k]` for geglu), `act` row-major
// `[m][k]`, `y` row-major `[m][n]`, bf16 throughout, f32 accumulation.
// Preconditions the wrapper enforces: `1 <= m <= 64`, `n` a whole number of
// the block's output columns, `k % kBK == 0`, 16-byte aligned pointers.
#pragma once

#include "prelude/device.cuh"
#include "prelude/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {

/// Activation rows a block covers; the mma B extent times eight.
constexpr int kSkinnyM = 64;

/// The staged row stride for a `kBK`-wide step, in bf16: eight elements of
/// padding so the eight addresses of one `ldmatrix` land in eight different
/// bank segments (see `kTiledLdA` in `linear/tiled.cuh`).
template <int kBK>
__host__ __device__ constexpr int skinny_ld() {
    return kBK + 8;
}

/// `ldmatrix.x4` over a row-major 16x16 bf16 tile: lane `l` addresses row
/// `l & 15`, column `(l >> 4) * 8`; the four registers come back as the
/// (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15), (rows 8-15,
/// k 8-15) 8x8 tiles — the mma A fragment, and, read off the activation
/// tile, the B fragments of two 8-row subtiles as {r0, r2} and {r1, r3}.
__device__ __forceinline__ void skinny_ldmatrix_x4(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

/// An L2 policy: keep (the activation tile every block re-reads) or stream
/// (the weight, read once).
__device__ __forceinline__ unsigned long long skinny_policy(bool keep) {
    unsigned long long policy;
    if (keep) {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(policy));
    } else {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    }
    return policy;
}

/// One 16-byte `cp.async` under an L2 policy; grouped by the same
/// `__pipeline_commit` / `__pipeline_wait_prior` as the intrinsic form.
__device__ __forceinline__ void skinny_cp_async16(void* dst, const void* src, unsigned long long policy) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "l"(policy)
                 : "memory");
}

/// Round through bf16: the value the traced second pass would have read.
__device__ __forceinline__ float skinny_round(float v) {
    return Elem<bf16>::to_f32(Elem<bf16>::from_f32(v));
}

/// `gelu_tanh(g) * u`, the arithmetic of `mlp_geglu_tanh_packed` verbatim.
__device__ __forceinline__ float skinny_geglu(float g, float u) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    return gelu * u;
}

template <int kWarps, int kStages, int kBK, int kEpilogue>
__global__ __launch_bounds__(kWarps * 32) void skinny_bf16_kernel(
    const bf16* __restrict__ w,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int m,
    int n,
    int k,
    float cap)
{
    constexpr bool kGeglu = kEpilogue == 2;
    constexpr int kThreads = kWarps * 32;
    constexpr int kSkinnyLd = skinny_ld<kBK>();
    static_assert(kBK % 64 == 0, "a step is whole 128-byte row pieces");
    static_assert(!kGeglu || kWarps % 2 == 0, "geglu halves the block's rows between gate and up");
    /// Weight rows a block owns: one mma A tile per warp.
    constexpr int kN = kWarps * 16;
    /// Output columns a block lands: every weight row, or the gate half's.
    constexpr int kCols = kGeglu ? kN / 2 : kN;
    /// Activation subtiles of eight rows: the mma B extent.
    constexpr int kNSubs = kSkinnyM / 8;
    /// k16 steps in one stage.
    constexpr int kQuad = kBK / 16;
    constexpr int kWStage = kN * kSkinnyLd;
    constexpr int kAStage = kSkinnyM * kSkinnyLd;
    constexpr int kStageElems = kWStage + kAStage;
    constexpr int kChunksPerRow = kBK / 8;
    constexpr int kWChunks = kN * kChunksPerRow;
    constexpr int kAChunks = kSkinnyM * kChunksPerRow;
    /// The epilogue tile's row stride, padded like the staging rows.
    constexpr int kLdC = kN + 8;
    static_assert(kStages >= 2, "one stage in flight while one is multiplied, at least");
    static_assert(kWChunks % kThreads == 0 && kAChunks % kThreads == 0,
                  "the staging map is a whole number of 16-byte chunks per thread");
    static_assert(kSkinnyM * kLdC <= kStageElems, "the epilogue tile fits in one stage of the ring");
    static_assert(kCols % 8 == 0, "the drain stores whole 16-byte chunks");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    /// The block's first output column.
    const int n0 = static_cast<int>(blockIdx.x) * kCols;

    extern __shared__ __align__(16) unsigned char skinny_smem[];
    bf16* ring = reinterpret_cast<bf16*>(skinny_smem);

    // **ACTIVATION ROWS PAST `m` ARE ZEROS, WRITTEN ONCE.** cp.async never
    // touches such a row, so zeroing the ring is the whole edge handling;
    // a full tile skips it.
    if (m < kSkinnyM) {
        u32* z = reinterpret_cast<u32*>(skinny_smem);
        constexpr int kWords = kStages * kStageElems / 2;
        for (int i = tid; i < kWords; i += kThreads) z[i] = 0u;
        __syncthreads();
    }

    // Which weight row the block's staged row `r` is: its own run, or for
    // geglu the gate run then the up run `n` (= I) rows further on.
    auto weight_row = [&](int r) -> int {
        if constexpr (kGeglu) {
            return r < kCols ? n0 + r : n + n0 + (r - kCols);
        } else {
            return n0 + r;
        }
    };

    // The weight streams through L2 once; the activation tile is what
    // every block re-reads, so it is the one to keep resident.
    const unsigned long long stream_policy = skinny_policy(false);
    const unsigned long long keep_policy = skinny_policy(true);

    // One stage: the block's weight rows and the live activation rows for
    // the k range, 16 bytes a chunk, eight adjacent threads per row so a
    // warp's fetch is four contiguous row pieces.
    auto stage = [&](int buf, int k0) {
        bf16* wdst = ring + buf * kStageElems;
        bf16* adst = wdst + kWStage;
#pragma unroll
        for (int i = 0; i < kWChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            skinny_cp_async16(
                wdst + r * kSkinnyLd + c,
                w + static_cast<long long>(weight_row(r)) * k + k0 + c,
                stream_policy);
        }
#pragma unroll
        for (int i = 0; i < kAChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            if (r < m) {
                skinny_cp_async16(
                    adst + r * kSkinnyLd + c,
                    act + static_cast<long long>(r) * k + k0 + c,
                    keep_policy);
            }
        }
        __pipeline_commit();
    };

    const int steps = k / kBK;

    float acc[kNSubs][4];
#pragma unroll
    for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[s][i] = 0.f;
    }

    // The prologue is `kStages - 1` groups, real or empty, so every
    // iteration's `wait_prior(kStages - 1)` means the same thing.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s * kBK);
        } else {
            __pipeline_commit();
        }
    }

    const int a_row = lane & 15;
    const int a_col = (lane >> 4) << 3;

    for (int step = 0; step < steps; ++step) {
        // Refill the buffer the previous iteration finished with (the
        // trailing barrier below is what makes that safe).
        const int fetch = step + kStages - 1;
        if (fetch < steps) {
            stage(fetch % kStages, fetch * kBK);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const bf16* wsrc = ring + (step % kStages) * kStageElems + warp * 16 * kSkinnyLd;
        const bf16* asrc = ring + (step % kStages) * kStageElems + kWStage;
#pragma unroll
        for (int kk = 0; kk < kQuad; ++kk) {
            unsigned a[4];
            skinny_ldmatrix_x4(a, wsrc + a_row * kSkinnyLd + kk * 16 + a_col);
#pragma unroll
            for (int pair = 0; pair < kNSubs / 2; ++pair) {
                unsigned b[4];
                skinny_ldmatrix_x4(b, asrc + (pair * 16 + a_row) * kSkinnyLd + kk * 16 + a_col);
                const unsigned b_lo[2] = {b[0], b[2]};
                const unsigned b_hi[2] = {b[1], b[3]};
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair], a, b_lo, acc[2 * pair]);
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair + 1], a, b_hi, acc[2 * pair + 1]);
            }
        }
        __syncthreads();
    }

    // **THE EPILOGUE GOES THROUGH SHARED MEMORY.** The accumulator's lane
    // map puts one lane's four values on two weight rows (`g`, `g + 8`) and
    // two adjacent activation rows (`2t`, `2t + 1`): scattered as direct
    // stores, whole rows once staged as `[act row][weight col]`. Staged
    // already rounded to bf16 — the value the traced second pass read.
    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = ring;
    {
        const int g = lane >> 2;
        const int t = lane & 3;
#pragma unroll
        for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int col = warp * 16 + g + 8 * half;
                const int row = s * 8 + 2 * t;
                c_tile[row * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half]);
                c_tile[(row + 1) * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half + 1]);
            }
        }
    }
    __syncthreads();

    constexpr int kChunksOfRow = kCols / 8;
    constexpr int kDrain = kSkinnyM * kChunksOfRow;
#pragma unroll
    for (int i = 0; i < (kDrain + kThreads - 1) / kThreads; ++i) {
        const int chunk = tid + i * kThreads;
        if (chunk < kDrain) {
            const int r = chunk / kChunksOfRow;
            const int c = (chunk % kChunksOfRow) * 8;
            if (r < m) {
                bf16* dst = out + static_cast<long long>(r) * n + n0 + c;
                const bf16* src = c_tile + r * kLdC + c;
                if constexpr (kEpilogue == 0) {
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                } else {
                    __align__(16) bf16 piece[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float v;
                        if constexpr (kEpilogue == 1) {
                            v = cap * tanhf(Elem<bf16>::to_f32(src[e]) * (1.f / cap));
                        } else {
                            v = skinny_geglu(
                                Elem<bf16>::to_f32(src[e]),
                                Elem<bf16>::to_f32(src[kCols + e]));
                        }
                        piece[e] = Elem<bf16>::from_f32(v);
                    }
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(piece);
                }
            }
        }
    }
#else
    (void)w;
    (void)act;
    (void)out;
    (void)m;
    (void)n;
    (void)k;
    (void)cap;
    __trap();
#endif
}

}  // namespace pie::linear
