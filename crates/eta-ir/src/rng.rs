//! The canonical ETA RNG contract.
//!
//! The deterministic CUDA/C++ and MSL projections of this formula are emitted
//! by `eta-compiler`; this module is the formula itself plus the host
//! implementation the reference interpreter runs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One `value ^= value >> xor_shift; value *= multiplier` step of
/// [`splitmix64`].
pub struct SplitMix64Round {
    /// Right-shift distance of the xor-fold.
    pub xor_shift: u32,
    /// Odd multiplier applied after the fold, or `None` for a final fold
    /// with no multiply.
    pub multiplier: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Every constant the ETA RNG is made of, in one value. These numbers are ABI: a backend that reproduces the ops but not the constants produces a different token stream from the same seed. Emitters must project this struct rather than transcribe the literals.
pub struct RngFormula {
    /// The three mixing rounds, applied in order.
    pub splitmix64_rounds: [SplitMix64Round; 3],
    /// Multiplied by the lane index to decorrelate neighbouring lanes.
    pub lane_stride: u64,
    /// Added to the lane index before scaling, so lane `0` is not the
    /// unmixed seed.
    pub lane_index_bias: u64,
    /// How far to shift a mixed word down before taking mantissa bits.
    pub uniform_mantissa_shift: u32,
    /// How many bits of the mixed word become the uniform draw.
    pub uniform_mantissa_bits: u32,
    /// Added to the integer draw before scaling, centring it in its bucket
    /// so neither `0.0` nor `1.0` can be produced.
    pub uniform_midpoint: f32,
    /// Folded into an ambient seed so seed `0` is not the unmixed state.
    pub ambient_seed_xor: u64,
    /// How far the key is shifted above the counter when the two pack into
    /// one keyed seed word.
    pub keyed_word_bits: u32,
}

/// The one instance of [`RngFormula`] that defines ETA's RNG.
pub const RNG_FORMULA: RngFormula = RngFormula {
    splitmix64_rounds: [
        SplitMix64Round {
            xor_shift: 27,
            multiplier: Some(0x3C79_AC49_2BA7_B653),
        },
        SplitMix64Round {
            xor_shift: 33,
            multiplier: Some(0x1C69_B3F7_4AC4_AE35),
        },
        SplitMix64Round {
            xor_shift: 27,
            multiplier: None,
        },
    ],
    lane_stride: 0x9E37_79B9_7F4A_7C15,
    lane_index_bias: 1,
    uniform_mantissa_shift: 40,
    uniform_mantissa_bits: 24,
    uniform_midpoint: 0.5,
    ambient_seed_xor: 0xA5A5_A5A5,
    keyed_word_bits: 32,
};

#[inline]
/// Mixes `value` through [`RNG_FORMULA`]'s rounds.
pub fn splitmix64(mut value: u64) -> u64 {
    for round in RNG_FORMULA.splitmix64_rounds {
        value ^= value >> round.xor_shift;
        if let Some(multiplier) = round.multiplier {
            value = value.wrapping_mul(multiplier);
        }
    }
    value
}

#[inline]
/// The effective 64-bit seed for an ambient draw from `seed`.
pub fn seed_eff(seed: u32) -> u64 {
    seed as u64 ^ RNG_FORMULA.ambient_seed_xor
}

#[inline]
/// The per-stream offset that keeps two [`Op::Rng`](crate::op::Op::Rng) ops
/// in one fire from drawing the same numbers.
pub fn stream_salt(stream: u32) -> u64 {
    splitmix64((stream as u64).wrapping_mul(RNG_FORMULA.lane_stride))
}

#[inline]
/// The effective seed for stream `stream` of ambient seed `seed`.
pub fn seed_eff_stream(seed: u32, stream: u32) -> u64 {
    seed_eff(seed) ^ stream_salt(stream)
}

#[inline]
/// The effective seed for a keyed draw from an `[key, ctr]` state tensor.
///
/// Nothing ambient enters here: the draw is a pure function of the state,
/// which is what makes a replay of the same program bit-identical.
pub fn keyed_seed(key: u32, counter: u32) -> u64 {
    splitmix64(((key as u64) << RNG_FORMULA.keyed_word_bits) | counter as u64)
}

/// The largest `f32` strictly below `1.0`.
///
/// For the top mantissa value, `(bits + 0.5) / 2^24` rounds to `1.0` exactly in `f32`, which breaks any consumer assuming a half-open range: `gumbel = -log(-log(u))` evaluates to `+inf` at `u = 1`, and `+inf` unconditionally wins `argmax(logits + gumbel)`.
pub const UNIFORM_MAX: f32 = 1.0 - f32::EPSILON / 2.0;

#[inline]
/// The uniform draw for lane `index` under effective seed `seed_eff`.
///
/// The result is in `(0, 1)` — never `0.0`, and never above
/// [`UNIFORM_MAX`].
pub fn hash_uniform(seed_eff: u64, index: u32) -> f32 {
    let x = seed_eff.wrapping_add(
        RNG_FORMULA
            .lane_stride
            .wrapping_mul(index as u64 + RNG_FORMULA.lane_index_bias),
    );
    let bits = (splitmix64(x) >> RNG_FORMULA.uniform_mantissa_shift) as u32;
    let denominator = (1u32 << RNG_FORMULA.uniform_mantissa_bits) as f32;
    let raw = (bits as f32 + RNG_FORMULA.uniform_midpoint) * (1.0 / denominator);
    if raw < UNIFORM_MAX { raw } else { UNIFORM_MAX }
}

/// How many uniform lanes one [`RngKind::Normal`](crate::types::RngKind::Normal)
/// draw consumes.
///
/// Box-Muller turns a pair of uniforms into a pair of independent normals;
/// ETA keeps only the cosine branch and spends the sine one, because the
/// alternative - carrying the second variate to the next element - would make
/// a draw depend on which lanes a backend happens to schedule together.
/// Element `i` reads lanes `NORMAL_PAIR_STRIDE * i` and
/// `NORMAL_PAIR_STRIDE * i + 1`, a pure function of `i` alone, so a row block,
/// a whole plane and the host interpreter all write the same numbers.
pub const NORMAL_PAIR_STRIDE: u32 = 2;

/// The `2*pi` of the Box-Muller angle, as the `f32` every backend spells.
///
/// A number here, not a `core::f32::consts::TAU` reference, for the reason
/// the rest of [`RngFormula`] is one: an emitter printing its own constant
/// draws a different - still marginally correct - normal, and nothing would
/// catch the drift. `6.2831855` is the nearest `f32` to `2*pi`.
pub const NORMAL_TWO_PI: f32 = 6.283_185_5;

/// The standard-normal draw for lane `index` under effective seed `seed_eff`.
///
/// `z = sqrt(-2*ln(u0)) * cos(2*pi*u1)` over `u0 = hash_uniform(seed, 2*index)`
/// and `u1 = hash_uniform(seed, 2*index + 1)`. Finite for every input:
/// [`hash_uniform`] never returns `0.0`, so the log is never `-inf`, and never
/// returns `1.0`, so `-2*ln(u0)` is never negative.
///
/// The device projections in `eta-compiler`'s emitters spell this expression
/// for expression, in this operand order; the only slack between host and
/// device is the platform's own `logf`/`cosf` rounding, the same slack
/// [`RngKind::Gumbel`](crate::types::RngKind::Gumbel) has always carried.
///
/// `std`-gated because the transform needs `ln`/`sqrt`/`cos`, which `core`
/// does not have. The guest half of this crate builds the IR and never
/// evaluates it, so the `no_std` surface loses nothing.
#[cfg(feature = "std")]
#[inline]
pub fn hash_normal(seed_eff: u64, index: u32) -> f32 {
    let lane = index.wrapping_mul(NORMAL_PAIR_STRIDE);
    let u0 = hash_uniform(seed_eff, lane);
    let u1 = hash_uniform(seed_eff, lane.wrapping_add(1));
    let radius = (-2.0f32 * u0.ln()).sqrt();
    radius * (NORMAL_TWO_PI * u1).cos()
}
