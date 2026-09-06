//! Deterministic backend projections of the canonical ETA RNG contract,
//! printed from [`eta_ir::rng::RNG_FORMULA`] so device and host cannot drift.

use core::fmt::Write;

use eta_ir::rng::{NORMAL_PAIR_STRIDE, NORMAL_TWO_PI, RNG_FORMULA};

/// The nearest `f32` to this text is exactly `UNIFORM_MAX`, so host and
/// device agree bit for bit.
const UNIFORM_MAX_LITERAL: &str = "0.99999994";

/// The nearest `f32` to this text is exactly [`NORMAL_TWO_PI`], for the same
/// reason [`UNIFORM_MAX_LITERAL`] is spelled out: a shortest-round-trip
/// print of the constant is what host and device must both parse.
const NORMAL_TWO_PI_LITERAL: &str = "6.2831855";

/// The Box-Muller projection: `sqrt(-2*ln(u0)) * cos(2*pi*u1)` over the two
/// uniform lanes `NORMAL_PAIR_STRIDE * index` and `+ 1`, expression for
/// expression with `eta_ir::rng::hash_normal`. `sqrt`/`log`/`cos` are the
/// backend's spellings of the three library calls.
fn normal_body(sqrt: &str, log: &str, cos: &str, u32_ty: &str) -> String {
    debug_assert_eq!(
        NORMAL_TWO_PI_LITERAL.parse::<f32>().ok(),
        Some(NORMAL_TWO_PI)
    );
    let stride = NORMAL_PAIR_STRIDE;
    let two_pi = NORMAL_TWO_PI_LITERAL;
    let mut out = String::new();
    let _ = writeln!(out, "  const {u32_ty} lane = index * {stride}u;");
    let _ = writeln!(
        out,
        "  const float u0 = ptir_rng_hash_uniform(seed_eff, lane);"
    );
    let _ = writeln!(
        out,
        "  const float u1 = ptir_rng_hash_uniform(seed_eff, lane + 1u);"
    );
    let _ = writeln!(out, "  const float radius = {sqrt}(-2.0f * {log}(u0));");
    let _ = writeln!(out, "  return radius * {cos}({two_pi}f * u1);");
    out
}

/// The `__device__` projection, as spliced into emitted CUDA sources.
pub fn cuda_device_functions() -> String {
    let mut out = String::new();
    let (inline, u64_ty, u32_ty, u64_suffix) = (
        "__device__ __forceinline__",
        "unsigned long long",
        "unsigned int",
        "ULL",
    );
    let denominator = 1u64 << RNG_FORMULA.uniform_mantissa_bits;
    let uniform_max = UNIFORM_MAX_LITERAL;
    let _ = writeln!(out, "{inline} {u64_ty} ptir_rng_splitmix64({u64_ty} x) {{");
    for round in RNG_FORMULA.splitmix64_rounds {
        let _ = writeln!(out, "  x ^= x >> {};", round.xor_shift);
        if let Some(multiplier) = round.multiplier {
            let _ = writeln!(out, "  x *= 0x{multiplier:016X}{u64_suffix};");
        }
    }
    out.push_str("  return x;\n}\n");
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff({u32_ty} seed) {{\n  return ({u64_ty})seed ^ 0x{:016X}{u64_suffix};\n}}",
        RNG_FORMULA.ambient_seed_xor
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_stream_salt({u32_ty} stream) {{\n  return ptir_rng_splitmix64(\n      ({u64_ty})stream * 0x{:016X}{u64_suffix});\n}}",
        RNG_FORMULA.lane_stride
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_seed_eff_stream(\n    {u32_ty} seed, {u32_ty} stream) {{\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}}"
    );
    let _ = writeln!(
        out,
        "{inline} {u64_ty} ptir_rng_keyed_seed(\n    {u32_ty} key, {u32_ty} counter) {{\n  return ptir_rng_splitmix64(\n      (({u64_ty})key << {}) | ({u64_ty})counter);\n}}",
        RNG_FORMULA.keyed_word_bits
    );
    let _ = writeln!(
        out,
        "{inline} float ptir_rng_hash_uniform(\n    {u64_ty} seed_eff, {u32_ty} index) {{\n  const {u64_ty} x = seed_eff +\n      0x{:016X}{u64_suffix} * (({u64_ty})index + {}{u64_suffix});\n  const {u32_ty} bits =\n      ({u32_ty})(ptir_rng_splitmix64(x) >> {});\n  const float raw = ((float)bits + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f, which would\n     make gumbel = -log(-log(u)) evaluate to +inf and hijack every argmax */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    );
    let _ = writeln!(
        out,
        "{inline} float ptir_rng_hash_normal(\n    {u64_ty} seed_eff, {u32_ty} index) {{\n{}}}",
        normal_body("sqrtf", "logf", "cosf", u32_ty)
    );
    out
}

/// Renders the `ptir_rng.generated.metal` preamble — the same RNG contract in
/// MSL, wrapped in an include guard.
pub fn generate_msl_preamble() -> String {
    let mut out = String::from(
        "// ptir_rng.generated.metal — GENERATED from crates/eta-ir/src/rng.rs.\n\
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract\n\
#ifndef PIE_PTIR_RNG_GENERATED_METAL\n\
#define PIE_PTIR_RNG_GENERATED_METAL\n\
\n\
inline ulong ptir_rng_splitmix64(ulong x) {\n",
    );
    for round in RNG_FORMULA.splitmix64_rounds {
        let _ = writeln!(out, "  x ^= x >> {};", round.xor_shift);
        if let Some(multiplier) = round.multiplier {
            let _ = writeln!(out, "  x *= 0x{multiplier:016X}ul;");
        }
    }
    out.push_str("  return x;\n}\n");
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_seed_eff(uint seed) {{\n  return ulong(seed) ^ 0x{:016X}ul;\n}}",
        RNG_FORMULA.ambient_seed_xor
    );
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_stream_salt(uint stream) {{\n  return ptir_rng_splitmix64(\n      ulong(stream) * 0x{:016X}ul);\n}}",
        RNG_FORMULA.lane_stride
    );
    out.push_str(
        "inline ulong ptir_rng_seed_eff_stream(uint seed, uint stream) {\n  return ptir_rng_seed_eff(seed) ^ ptir_rng_stream_salt(stream);\n}\n",
    );
    let _ = writeln!(
        out,
        "inline ulong ptir_rng_keyed_seed(uint key, uint counter) {{\n  return ptir_rng_splitmix64(\n      (ulong(key) << {}) | ulong(counter));\n}}",
        RNG_FORMULA.keyed_word_bits
    );
    let denominator = 1u64 << RNG_FORMULA.uniform_mantissa_bits;
    let uniform_max = UNIFORM_MAX_LITERAL;
    let _ = writeln!(
        out,
        "inline float ptir_rng_hash_uniform(ulong seed_eff, uint index) {{\n  const ulong x = seed_eff +\n      0x{:016X}ul * (ulong(index) + {}ul);\n  const uint bits = uint(ptir_rng_splitmix64(x) >> {});\n  const float raw = (float(bits) + {:.1}f) * (1.0f / {denominator}.0f);\n  /* clamp off the one draw in 2^24 that rounds to exactly 1.0f */\n  return raw < {uniform_max}f ? raw : {uniform_max}f;\n}}\n",
        RNG_FORMULA.lane_stride,
        RNG_FORMULA.lane_index_bias,
        RNG_FORMULA.uniform_mantissa_shift,
        RNG_FORMULA.uniform_midpoint
    );
    let _ = writeln!(
        out,
        "inline float ptir_rng_hash_normal(ulong seed_eff, uint index) {{\n{}}}",
        normal_body("sqrt", "precise::log", "precise::cos", "uint")
    );
    out.push_str("#endif\n");
    out
}

