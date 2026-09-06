#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

/// **THE T5 RELATIVE-POSITION BUCKET, EXACTLY AS HUGGING FACE'S
/// `T5Attention._relative_position_bucket` COMPUTES IT**, for one signed
/// distance `d = memory_position − context_position` (`kj − qi`):
///
/// ```text
/// bucket = 0
/// if bidirectional: num_buckets /= 2; bucket += (d > 0) · num_buckets; n = |d|
/// else:             n = −min(d, 0)
/// max_exact = num_buckets / 2
/// if n < max_exact: return bucket + n
/// large = max_exact + trunc( ln(n / max_exact) / ln(max_distance / max_exact)
///                            · (num_buckets − max_exact) )        -- f32, truncated toward zero
/// return bucket + min(large, num_buckets − 1)
/// ```
///
/// The logarithm's ratio is f32 like torch's (`torch.log` of an f32 tensor
/// over a Python `math.log` cast to f32 at the divide): `log_ratio` is the
/// host's `f32(ln(max_distance / max_exact))` and the numerator is the
/// accurate `logf`, so an exact boundary (`n = 16, 32, 64` at T5's 32/128)
/// truncates to the integer torch lands, not one below it.
__device__ __forceinline__ int relative_position_bucket(
    int d, bool bidirectional, int num_buckets, float log_ratio)
{
    int bucket = 0;
    int n;
    if (bidirectional) {
        num_buckets /= 2;
        if (d > 0) bucket += num_buckets;
        n = d < 0 ? -d : d;
    } else {
        n = d < 0 ? -d : 0;
    }
    const int max_exact = num_buckets / 2;
    if (n < max_exact) return bucket + n;
    const float x = logf(static_cast<float>(n) / static_cast<float>(max_exact));
    const float scaled = x / log_ratio * static_cast<float>(num_buckets - max_exact);
    int large = max_exact + static_cast<int>(scaled);
    if (large > num_buckets - 1) large = num_buckets - 1;
    return bucket + large;
}

/// **THE DENSE RELATIVE BIAS TABLE A LAYER'S ATTENTION READS**:
/// `y[h][c] = embedding[bucket(c − (max_len − 1))][h]` for `c` in
/// `0 .. 2·max_len − 1` — one row per head, the column of distance `d` at
/// `d + max_len − 1`, which is where `attention.ragged`'s `RelativeBias`
/// arm looks it up. `embedding` is the checkpoint's `[num_buckets, heads]`
/// plane as stored (bf16 or f32, row-major, `stride` elements per bucket
/// row); the table is f32. One thread per cell, the bucket recomputed per
/// cell — the table is `heads · (2·max_len − 1)` cells, 64 · 1023 for umT5,
/// so nothing here is worth sharing.
template <class T>
__global__ void relative_bucket_bias(
    const T* __restrict__ embedding,
    float* __restrict__ y,
    int heads,
    int span,
    int max_len,
    int stride,
    int num_buckets,
    int bidirectional,
    float log_ratio)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(heads) * span) return;
    const int h = static_cast<int>(i / span);
    const int c = static_cast<int>(i % span);
    const int d = c - (max_len - 1);
    const int b = relative_position_bucket(d, bidirectional != 0, num_buckets, log_ratio);
    y[i] = Elem<T>::to_f32(embedding[static_cast<long long>(b) * stride + h]);
}

}
