#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

/// **THE TIMESTEP EMBEDDING, EXACTLY AS `diffusers.get_timestep_embedding`
/// WRITES IT** (`.wiki/imagegen/design.md` D6's prerequisite: the denoiser's
/// only input the guest cannot hand over as an activation).
///
/// `half = dim / 2` frequencies `exp(-ln(max_period) · i / half)`; the angle
/// is `scale · t · freq`; the row is `[sin | cos]`, or `[cos | sin]` when
/// `flip_sin_cos`; an odd `dim` leaves the last column zero. That denominator
/// is `half` and not `half - 1`, which is `get_timestep_embedding`'s
/// `downscale_freq_shift = 0` — the value every DiT in the set passes, and
/// the openai/`timestep_embedding` original.
///
/// **f32 IN, f32 OUT, AND THE ACCURATE TRANSCENDENTALS.** `expf`/`sincosf`,
/// not `__expf`/`__sincosf`: this embedding is the head of the adaLN MLP, one
/// row per lane per step, so its cost is nothing and its error is multiplied
/// by every modulation vector in the block. `t` is f32 because the schedule's
/// sigma is a fraction, not a token index.
__global__ void sinusoid(
    const float* __restrict__ t,
    float* __restrict__ o,
    int dim,
    float max_period,
    int flip_sin_cos,
    float scale,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (`rmsnorm_row`'s idiom, one block per row): a
    // replay whose grid was carved at a bucket retires its padded rows here.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: both `t` and `o` are row
    // planes handed at their base.
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int half = dim / 2;
    const float tv = t[row];
    const float log_period = logf(max_period);
    float* orow = o + static_cast<long long>(row) * dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq =
            expf(-log_period * static_cast<float>(i) / static_cast<float>(half));
        float sin_v, cos_v;
        // `scale · (t · freq)`, in the reference's own order.
        sincosf(scale * (tv * freq), &sin_v, &cos_v);
        orow[i] = flip_sin_cos ? cos_v : sin_v;
        orow[i + half] = flip_sin_cos ? sin_v : cos_v;
    }
    // The zero pad an odd width takes: `F.pad(emb, (0, 1, 0, 0))`.
    if ((dim & 1) != 0 && threadIdx.x == 0) orow[dim - 1] = 0.f;
}

}
