#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

/// **THE BARE POINTWISE OPS A DiT GRAPH NEEDS AND THE TRUNKS NEVER DID**
/// (`.wiki/imagegen/design.md` D6). A language trunk's every add is a
/// residual (`residual_add`, in place on the stream) and its every activation
/// is half of a gated MLP (`mlp_swiglu`, `mlp_geglu_tanh`), so the tree grew
/// no plain `x + y` and no plain `silu(x)`. A DiT graph has both: two
/// conditioning streams summed into one, a gate built out of a `silu` between
/// two linears, an embedding folded by `tanh`.
///
/// Only what is missing is here. `gelu_tanh` already exists as
/// `linear::mlp_gelu_tanh`, and `elemwise::activation::gelu_tanh` names that
/// entry rather than transcribing its polynomial a third time.
constexpr int kBinAdd = 0;
constexpr int kBinMul = 1;

/// `o = x + y` / `x · y`, one thread per element. `o` may alias either input.
template <class T, int OP>
__global__ void binary(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // The staged-geometry seat in the ELEMENT form this flat launch needs
    // (`mlp_gelu_tanh`'s idiom): a lane is not a row here, so the live-rows
    // word bounds `win[0] * width` elements and `win[1] * width` is where they
    // begin.
    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float a = Elem<T>::to_f32(x[at]);
    const float b = Elem<T>::to_f32(y[at]);
    o[at] = Elem<T>::from_f32(OP == kBinAdd ? a + b : a * b);
}

constexpr int kActSilu = 0;
constexpr int kActTanh = 1;

/// `o = silu(x)` / `tanh(x)`, one thread per element, out of place (`o` may
/// alias `x`). `silu_scaled` next door is the same arithmetic in place on one
/// plane with a scale in front; this is the two-plane shape a graph that does
/// not consume its input needs.
template <class T, int ACT>
__global__ void activation(
    const T* __restrict__ x,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float v = Elem<T>::to_f32(x[at]);
    // `silu_scaled`'s expression, transcribed: `__expf` and one rounding.
    o[at] = Elem<T>::from_f32(ACT == kActSilu ? v / (1.f + __expf(-v)) : tanhf(v));
}

}
