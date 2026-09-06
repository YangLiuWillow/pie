#pragma once

// selector_walk.cuh — DFlash2's candidate selector, walked. The
// transcription of `kernels-metal/kernels/attn/selector_walk.metal`.
//
// The reference (`mlx_dspark.dflash_model.CandidateSelector`) scores every
// `(predecessor, candidate)` pair of adjacent slots,
//
//     scores[s, p, c] = unary[s, c] + < A[pred[s, p]] * hp[s], B[cand[s, c]] >
//
// and `walk_greedy` follows the best successor from the anchor: only the ROW
// of the predecessor actually chosen is ever read, so a walk is `slots x K`
// dot products of `rank` terms, not `slots x K x K`. One block per request:
// 256 threads are sixteen lanes a candidate, the lanes stride the rank and
// fold with shuffles inside their aligned sixteen (two candidates a warp),
// thread 0 takes the argmax (ties to the lower candidate) and the pick
// becomes the next slot's predecessor. Rows are the request's span in order:
// the first is the anchor (its pick is its first candidate, unread by any
// guest), the rest are mask slots.
//
// bf16 in, f32 accumulation; the reference is bf16 bilinear plus f32 unary.

#include "prelude/device.cuh"

namespace pie::attn {

template <class T>
using Elem = ::pie::Elem<T>;

constexpr unsigned WALK_THREADS = 256;
constexpr unsigned WALK_LANES = 16;  // lanes a candidate
constexpr unsigned WALK_MAX_K = WALK_THREADS / WALK_LANES;

template <class T>
__global__ void __launch_bounds__(WALK_THREADS) selector_walk(
    const i32* __restrict__ cand,     // [rows, k]
    const i32* __restrict__ indptr,   // [lanes + 1]
    const float* __restrict__ unary,  // [rows, k]
    const T* __restrict__ hp,         // [rows, rank], read when has_hp
    const i32* __restrict__ tokens,   // [rows]
    const T* __restrict__ pred,       // [vocab, rank]
    const T* __restrict__ succ,       // [vocab, rank]
    i32* __restrict__ picks,          // [rows]
    int k,
    int rank,
    int vocab,
    int has_hp,  // 0: a plain bigram lattice
    int first,   // the span's first slot row
    const u32* __restrict__ win)
{
    const int r = static_cast<int>(blockIdx.x);
    if (win != nullptr && blockIdx.x >= win[2]) return;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid / WALK_LANES;     // this thread's candidate
    const unsigned lane = tid % WALK_LANES;  // its lane inside the candidate
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    if (end <= begin) return;

    __shared__ float score[WALK_MAX_K];
    __shared__ int prev_id;
    if (tid == 0) {
        // The predecessor of the first slot is the anchor's own token. When
        // the anchor row is not a slot (`first == 1`) it proposes nothing,
        // and its pick is its own first candidate.
        if (first > 0) {
            picks[begin] = cand[static_cast<size_t>(begin) * static_cast<size_t>(k)];
        }
        prev_id = tokens[begin];
    }
    __syncthreads();

    for (int row = begin + first; row < end; ++row) {
        const int my_prev = prev_id;
        float partial = 0.0f;
        if (static_cast<int>(c) < k) {
            const int cid = cand[static_cast<size_t>(row) * static_cast<size_t>(k) + c];
            const bool live = my_prev >= 0 && my_prev < vocab && cid >= 0 && cid < vocab;
            if (live) {
                const T* a = pred + static_cast<size_t>(my_prev) * static_cast<size_t>(rank);
                const T* b = succ + static_cast<size_t>(cid) * static_cast<size_t>(rank);
                if (has_hp) {
                    const T* h = hp + static_cast<size_t>(row) * static_cast<size_t>(rank);
                    for (int d = static_cast<int>(lane); d < rank; d += static_cast<int>(WALK_LANES)) {
                        partial += Elem<T>::to_f32(a[d]) * Elem<T>::to_f32(h[d]) * Elem<T>::to_f32(b[d]);
                    }
                } else {
                    for (int d = static_cast<int>(lane); d < rank; d += static_cast<int>(WALK_LANES)) {
                        partial += Elem<T>::to_f32(a[d]) * Elem<T>::to_f32(b[d]);
                    }
                }
            }
        }
        // Fold the sixteen lanes of this candidate; the xor tree stays inside
        // the aligned sixteen, so the two candidates sharing a warp do not
        // mix.
        partial += __shfl_xor_sync(0xffffffffu, partial, 8);
        partial += __shfl_xor_sync(0xffffffffu, partial, 4);
        partial += __shfl_xor_sync(0xffffffffu, partial, 2);
        partial += __shfl_xor_sync(0xffffffffu, partial, 1);
        if (lane == 0 && static_cast<int>(c) < k) {
            score[c] = unary[static_cast<size_t>(row) * static_cast<size_t>(k) + c] + partial;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            float best_v = score[0];
            for (int j = 1; j < k; ++j) {
                if (score[j] > best_v) {
                    best_v = score[j];
                    best = j;
                }
            }
            const int pick = cand[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(best)];
            picks[row] = pick;
            prev_id = pick;
        }
        __syncthreads();
    }
}

}  // namespace pie::attn
