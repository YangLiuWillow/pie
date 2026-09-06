//! The skinny bf16 projection with its epilogue folded in
//! (`linear/skinny.cuh`): a few activation rows against a big weight,
//! streamed by our own tensor-core kernel, finished in registers by the pass
//! the trace would otherwise run over the result. Not a dense tactic the
//! tuner races — its speed matches cuBLAS's on the decode shapes, and the
//! point is the pass it saves.

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};

/// What the drain does with each bf16-rounded product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epilogue {
    /// `y = act x w^T`.
    Store,
    /// `y = cap * tanh(y / cap)`: the head's logit softcap.
    Softcap(f32),
    /// `w` is the packed `[2I x k]` up/gate weight and `n = I`:
    /// `y = gelu_tanh(gate) * up`.
    Geglu,
}

/// Activation rows the kernel covers; `kSkinnyM` in the header.
pub const ROWS: u32 = 64;

/// Warps per block: a block owns 64 weight rows (32 gate + 32 up for geglu).
/// With two stages of a 128-wide step this instantiation ran the decode
/// shapes fastest of the six tried, in a clean microbench and in situ alike
/// (4x4x64, 8x3x64, 4x3x64, 2x3x128, 1x2x256 were the others).
const WARPS: u32 = 4;
/// cp.async ring depth.
const STAGES: u32 = 2;
/// The contraction step: 256 bytes of one weight row per stage.
const STEP: u32 = 128;

/// Dynamic shared memory: `STAGES` of (64 weight rows + 64 activation rows)
/// at the padded `STEP + 8` stride, bf16. Mirrors `skinny_smem_bytes`.
const SMEM_BYTES: u32 = STAGES * (WARPS * 16 + ROWS) * (STEP + 8) * 2;

/// Output columns one block lands.
const fn cols(epilogue: Epilogue) -> u32 {
    match epilogue {
        Epilogue::Geglu => WARPS * 8,
        Epilogue::Store | Epilogue::Softcap(_) => WARPS * 16,
    }
}

/// Whether the kernel covers `m` rows of `k`-wide activations against an
/// `n`-column output (`n = I` for geglu): the wrapper's own refusals,
/// answered without firing so a caller can pick its other road.
#[must_use]
pub fn covers(m: i32, n: i32, k: i32, epilogue: Epilogue) -> bool {
    let cols = cols(epilogue) as i32;
    m >= 1 && m <= ROWS as i32 && n > 0 && n % cols == 0 && k > 0 && k % STEP as i32 == 0
}

/// `y[m][n] = epilogue(act[m][k] x w[n][k]^T)`, bf16, `1 <= m <= 64`. For
/// [`Epilogue::Geglu`], `w` is `[2n][k]` and `n` is the intermediate width.
///
/// # Errors
///
/// [`Error::Refused`] for a shape [`covers`] refuses, a null or misaligned
/// pointer, or a launch the runtime refused.
pub fn skinny_bf16(
    ctx: &Ctx,
    weight: u64,
    act: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    epilogue: Epilogue,
) -> Result<(), Error> {
    if !covers(m, n, k, epilogue) {
        return Err(refuse(
            "linear.skinny",
            format!(
                "covers 1 <= m <= 64, n in whole {}s and k in whole {STEP}s, and was handed \
                 m={m}, n={n}, k={k} for {epilogue:?}",
                cols(epilogue)
            ),
        ));
    }
    for (address, what) in [
        (weight, "the weight"),
        (act, "the activation"),
        (out, "the output"),
    ] {
        if address == 0 {
            return Err(refuse("linear.skinny", format!("{what} is null")));
        }
        if !aligned16(address) {
            return Err(refuse(
                "linear.skinny",
                format!("{what} is not 16-byte aligned, and cp.async demands it"),
            ));
        }
    }
    let cap = match epilogue {
        Epilogue::Softcap(cap) => {
            if !(cap.is_finite() && cap > 0.0) {
                return Err(refuse(
                    "linear.skinny",
                    format!("a softcap of {cap} is not a positive number"),
                ));
            }
            cap
        }
        Epilogue::Store | Epilogue::Geglu => 0.0,
    };
    let entrypoint = match epilogue {
        Epilogue::Store => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 0>",
        Epilogue::Softcap(_) => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 1>",
        Epilogue::Geglu => "::pie::linear::skinny_bf16_kernel<4, 2, 128, 2>",
    };
    let values = [
        ArgValue::Ptr(weight),
        ArgValue::Ptr(act),
        ArgValue::Ptr(out),
        ArgValue::I32(m),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(cap),
    ];
    ctx.fire(
        "linear.skinny",
        Fire::at("linear/skinny.cuh", entrypoint).apply(
            Launch::grid(
                [(n / cols(epilogue) as i32) as u32, 1, 1],
                [32 * WARPS, 1, 1],
            )
            .smem(SMEM_BYTES),
        ),
        &values,
    )
}
