#pragma once

// topk.cuh — a row's argmax into one column of an i32 plane, and a row's k
// largest entries sorted with their indices beside. The transcription of
// `kernels-metal/kernels/layout/{argmax,topk}.metal`, with the same rule
// throughout: ties go to the LOWEST column and a NaN never wins — the
// epilogue's `reduce_argmax` rule, kept here so a draft the head chained on
// is the token the verifier reads back from the same logits.

#include "prelude/device.cuh"

namespace pie::layout {

// The row's elements widened: bf16 through the prelude's `Elem`, f32 as is
// (the plane stamps both; `Elem<float>` is not a thing).
__device__ __forceinline__ float topk_load(bf16 v) { return ::pie::Elem<bf16>::to_f32(v); }
__device__ __forceinline__ float topk_load(float v) { return v; }

constexpr unsigned TOPK_NONE = 0xFFFFFFFFu;

// `a` beats `b` when it is larger, or equal at a lower index.
__device__ __forceinline__ bool topk_beats(float av, unsigned ai, float bv, unsigned bi)
{
    return av > bv || (av == bv && ai < bi);
}

// One warp's best `(value, index)` under `topk_beats`, answered on every lane.
__device__ __forceinline__ void warp_best(float& v, unsigned& i)
{
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (oi != TOPK_NONE && (i == TOPK_NONE || topk_beats(ov, oi, v, i))) {
            v = ov;
            i = oi;
        }
    }
}

/// **ONE ROW'S ARGMAX, WRITTEN INTO ONE COLUMN OF AN I32 PLANE.**
///
/// `y[row * depth + column] = argmax_c x[row, c]`, one block per row. Every
/// thread scans a strided share of the row keeping its best `(value, index)`,
/// the warps fold with shuffles, thread 0 folds the warps.
template <class T>
__global__ void argmax_rows(
    const T* __restrict__ x,
    int* __restrict__ y,
    int width,
    int depth,
    int column,
    const u32* __restrict__ win)
{
    const unsigned row = blockIdx.x;
    if (win != nullptr && row >= win[0]) return;
    const T* src = x + static_cast<size_t>(row) * static_cast<size_t>(width);

    float best = neg_inf();
    unsigned best_i = TOPK_NONE;
    for (unsigned c = threadIdx.x; c < static_cast<unsigned>(width); c += blockDim.x) {
        const float v = topk_load(src[c]);
        if (!isnan(v) && (best_i == TOPK_NONE || topk_beats(v, c, best, best_i))) {
            best = v;
            best_i = c;
        }
    }

    __shared__ float part_v[32];
    __shared__ unsigned part_i[32];
    warp_best(best, best_i);
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    if (lane == 0) {
        part_v[warp] = best;
        part_i[warp] = best_i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned warps = (blockDim.x + 31u) >> 5;
        float top = neg_inf();
        unsigned top_i = TOPK_NONE;
        for (unsigned w = 0; w < warps; ++w) {
            const unsigned i = part_i[w];
            if (i != TOPK_NONE && (top_i == TOPK_NONE || topk_beats(part_v[w], i, top, top_i))) {
                top = part_v[w];
                top_i = i;
            }
        }
        y[static_cast<size_t>(row) * static_cast<size_t>(depth) + static_cast<size_t>(column)] =
            static_cast<int>(top_i == TOPK_NONE ? 0u : top_i);
    }
}

constexpr unsigned TOPK_THREADS = 128;
constexpr unsigned TOPK_WARPS = TOPK_THREADS / 32;

/// **THE K LARGEST ENTRIES OF EVERY ROW, SORTED, INDICES BESIDE.**
///
/// One block of `TOPK_THREADS` per row. Every thread walks a strided share
/// of the row keeping its own sorted list of K `(value, index)` pairs — most
/// candidates fail the list's floor and cost one compare — then the lists
/// meet in shared memory and the block pops the global maximum K times: each
/// thread offers its list's head, the warps fold, thread 0 picks across
/// warps, the owner advances. `values` is `[rows, K]` f32, `indices`
/// `[rows, K]` i32. `TOPK_THREADS x K x 8` bytes of shared memory: 16 KB at
/// K = 16.
template <class T, int K>
__global__ void __launch_bounds__(TOPK_THREADS) topk_rows(
    const T* __restrict__ x,
    float* __restrict__ values,
    int* __restrict__ indices,
    int width,
    const u32* __restrict__ win)
{
    const unsigned row = blockIdx.x;
    if (win != nullptr && row >= win[0]) return;
    const unsigned lid = threadIdx.x;
    const T* src = x + static_cast<size_t>(row) * static_cast<size_t>(width);

    // This thread's sorted list, best first.
    float lv[K];
    unsigned li[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        lv[j] = neg_inf();
        li[j] = TOPK_NONE;
    }
    for (unsigned c = lid; c < static_cast<unsigned>(width); c += TOPK_THREADS) {
        const float v = topk_load(src[c]);
        if (isnan(v)) continue;
        if (li[K - 1] != TOPK_NONE && !topk_beats(v, c, lv[K - 1], li[K - 1])) continue;
        // Insert, shifting the tail down.
        int at = K - 1;
        while (at > 0 && (li[at - 1] == TOPK_NONE || topk_beats(v, c, lv[at - 1], li[at - 1]))) {
            lv[at] = lv[at - 1];
            li[at] = li[at - 1];
            --at;
        }
        lv[at] = v;
        li[at] = c;
    }

    __shared__ float lists_v[TOPK_THREADS * K];
    __shared__ unsigned lists_i[TOPK_THREADS * K];
    __shared__ float part_v[TOPK_WARPS];
    __shared__ unsigned part_i[TOPK_WARPS];
    __shared__ unsigned part_owner[TOPK_WARPS];
    __shared__ unsigned winner;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        lists_v[lid * K + j] = lv[j];
        lists_i[lid * K + j] = li[j];
    }
    unsigned head = 0;
    const unsigned warp = lid >> 5;
    const unsigned lane = lid & 31;
    __syncthreads();

    for (int j = 0; j < K; ++j) {
        // Offer this list's head.
        float hv = head < static_cast<unsigned>(K) ? lists_v[lid * K + head] : neg_inf();
        unsigned hi = head < static_cast<unsigned>(K) ? lists_i[lid * K + head] : TOPK_NONE;
        const unsigned my_i = hi;
        warp_best(hv, hi);
        // The lane whose head won owns the pop; the lowest such lane if two
        // hold one index (they cannot: every column lives on one thread).
        unsigned owner = (my_i != TOPK_NONE && my_i == hi) ? lid : TOPK_NONE;
        for (int off = 16; off > 0; off >>= 1) {
            owner = min(owner, __shfl_xor_sync(0xffffffffu, owner, off));
        }
        if (lane == 0) {
            part_v[warp] = hv;
            part_i[warp] = hi;
            part_owner[warp] = owner;
        }
        __syncthreads();
        if (lid == 0) {
            float top = neg_inf();
            unsigned top_i = TOPK_NONE;
            unsigned top_owner = TOPK_NONE;
            for (unsigned w = 0; w < TOPK_WARPS; ++w) {
                if (part_i[w] != TOPK_NONE
                    && (top_i == TOPK_NONE || topk_beats(part_v[w], part_i[w], top, top_i))) {
                    top = part_v[w];
                    top_i = part_i[w];
                    top_owner = part_owner[w];
                }
            }
            values[static_cast<size_t>(row) * K + j] = top_i == TOPK_NONE ? 0.0f : top;
            indices[static_cast<size_t>(row) * K + j] = static_cast<int>(top_i == TOPK_NONE ? 0u : top_i);
            winner = top_owner;
        }
        __syncthreads();
        if (winner == lid) {
            ++head;
        }
    }
}

}  // namespace pie::layout
