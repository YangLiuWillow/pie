use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Per-token math — tokens are independent. Per-token reductions like
/// rmsnorm's mean-of-squares belong here.
/// The YaRN interpolation a partial rope states beside its theta: the
/// reference's `precompute_freqs(dim, original_seq_len, base, factor,
/// beta_fast, beta_slow)` with `original_seq_len > 0`. The ramp bounds are
/// derived on the device side from these and the rotated width.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

/// The trailing norm of a fused [`RmsnormResidualAdd`](Elementwise::RmsnormResidualAdd):
/// `out = rmsnorm(row) * (weight [+ 1])` over the row the chain produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNorm {
    pub weight: ValueId,
    pub plus_one: bool,
    pub eps: f32,
    pub out: ValueId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Elementwise {
    Rmsnorm {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        y: ValueId,
    },
    RmsnormPerHead {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    /// Scales by `weight + 1` (Gemma-style).
    RmsnormPlusOne {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        y: ValueId,
    },
    RmsnormPerHeadPlusOne {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    RmsnormNoScale {
        x: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    /// The centred norm: `y = (x - mean(x)) / rms(x - mean(x))`, no scale, no
    /// bias. Separate from [`RmsnormNoScale`](Elementwise::RmsnormNoScale)
    /// since the mean subtraction is a second reduction, a different kernel.
    LayernormNoScale {
        x: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// The whole `nn.LayerNorm` in one row: `y = (x - mean(x)) *
    /// rsqrt(var(x) + eps) * w + b`, scale/bias read as `[width]` planes.
    /// Exists beside [`LayernormNoScale`](Elementwise::LayernormNoScale)
    /// because the import fold can't express this pair.
    Layernorm {
        x: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// Hyper-connection norm (qwen4): moments per `group`-wide slice, scaled
    /// by `weight + 1` over the full row width (per-stream weight, unlike
    /// [`RmsnormPerHeadPlusOne`](Elementwise::RmsnormPerHeadPlusOne)).
    RmsnormGroupedPlusOne {
        x: ValueId,
        weight: ValueId,
        group: u32,
        eps: f32,
        y: ValueId,
    },
    /// `x = min(max(x, lo), hi)`, in place, bounds as trace constants
    /// (gemma4's `use_clipped_linears`).
    Clamp {
        x: ValueId,
        lo: f32,
        hi: f32,
        x_out: ValueId,
    },
    /// The same clamp, with `lo`/`hi` as `[1]` device-held planes instead of
    /// trace constants (checkpoints shipping per-linear QAT bounds).
    ClampLearned {
        x: ValueId,
        lo: ValueId,
        hi: ValueId,
        x_out: ValueId,
    },
    /// `x` is f32; the norm is gated by `act(gate)`, per group of
    /// `head_dim`. `act` is the checkpoint's `output_gate_type` (qwen3.5:
    /// silu, qwen4: sigmoid).
    RmsnormGated {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        act: GateActivation,
        y: ValueId,
    },
    /// Like `RmsnormGated`, but grouped by head count instead of head width.
    RmsnormGatedBy {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        heads: u32,
        eps: f32,
        y: ValueId,
    },
    ResidualAdd {
        x: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    /// `y += x`, then `out = rmsnorm(y)`: the residual fold and the norm
    /// that reads it, one launch. Written by [`crate::fuse`], never traced.
    ResidualAddRmsnorm {
        x: ValueId,
        y: ValueId,
        y_out: ValueId,
        weight: ValueId,
        plus_one: bool,
        eps: f32,
        out: ValueId,
    },
    /// `t = rmsnorm(x) * weight`, then `y += t` in place, then — when the
    /// chain carries them — `scaled = y * s` for a device-held scalar `s`
    /// and `out = rmsnorm(scaled or y)` (`post`): the norm-add-scale-norm
    /// run between a block's projection and the next block, one launch
    /// where the trace lands three or four. Every intermediate (`t`, `y_out`,
    /// `scaled`) is still written with the bf16 rounding its own launch
    /// would give it, so every reader of the traced values survives.
    /// Written by [`crate::fuse`], never traced.
    RmsnormResidualAdd {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        t: ValueId,
        y: ValueId,
        y_out: ValueId,
        /// `(s, scaled)`: the scalar plane and the row it scales `y_out` into.
        scale: Option<(ValueId, ValueId)>,
        post: Option<PostNorm>,
    },
    /// `e = table[ids]`, `e_scaled = e * embed_scale`, `y += e_scaled` in
    /// place, `y_scaled = y * out_scale`: a per-layer input embedding folded
    /// into the stream it joins (gemma's per-layer inputs), one launch where
    /// the trace lands four. Every intermediate is written as its own launch
    /// would write it. Written by [`crate::fuse`], never traced.
    EmbedScaleAdd {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        e: ValueId,
        embed_scale: f32,
        e_scaled: ValueId,
        y: ValueId,
        y_out: ValueId,
        out_scale: f32,
        y_scaled: ValueId,
    },
    /// [`Elementwise::EmbedScaleAdd`] whose residual row is layer `layer`'s
    /// `width`-wide slice of the stacked table `stacked`, read in place —
    /// the `select` that copied it out folded away (`fuse::embed_select`).
    /// `y_out` is a fresh row, not an alias.
    EmbedScaleAddSelect {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        e: ValueId,
        embed_scale: f32,
        e_scaled: ValueId,
        stacked: ValueId,
        layer: u32,
        width: u32,
        y_out: ValueId,
        out_scale: f32,
        y_scaled: ValueId,
    },
    AddBias {
        bias: ValueId,
        out: ValueId,
        out_out: ValueId,
    },
    /// Vision tower output standardization (`vision_config.standardize`):
    /// `y = (x - bias) * scale`, per column, both planes `[width]`, in place.
    Standardize {
        x: ValueId,
        bias: ValueId,
        scale: ValueId,
        x_out: ValueId,
    },
    MulScalar {
        s: f32,
        x: ValueId,
        x_out: ValueId,
    },
    /// `silu(s * x)`, in place. The scalar is inside the activation
    /// (`silu(s*x) != s*silu(x)`), so this is one launch where
    /// [`MulScalar`](Elementwise::MulScalar) before a bare silu would be
    /// two.
    SiluScaled {
        s: f32,
        x: ValueId,
        x_out: ValueId,
    },
    Scale {
        s: ValueId,
        x: ValueId,
        x_out: ValueId,
    },
    /// Norms the summed blocks against the prefix, then projects the blend.
    ResBlend {
        prefix: ValueId,
        blocks: Vec<ValueId>,
        weight: ValueId,
        eps: f32,
        proj: ValueId,
        y: ValueId,
    },
    RopeFull {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        q_out: ValueId,
        k_out: ValueId,
    },
    RopePartial {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        q_out: ValueId,
        k_out: ValueId,
    },
    /// [`RopePartial`](Elementwise::RopePartial) over a position triple:
    /// `positions` is `[rows, 3]` `i32` (one `(t, h, w)` per row); `sections`
    /// is the checkpoint's `mrope_section`. `form` says which section layout
    /// applies; see [`MropeForm`].
    RopeMrope {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        sections: [u32; 3],
        form: MropeForm,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        q_out: ValueId,
        k_out: ValueId,
    },
    RopePartialQ {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        q_out: ValueId,
    },
    /// `RmsnormPerHead` then the `RopePartialQ` over its result, one node
    /// (`fuse::q_norm_rope`); `q_out` aliases `y`.
    RmsnormRopePartialQ {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        y: ValueId,
        q_out: ValueId,
    },
    /// Partial rope over the last `rotary_dim` lanes of each head.
    ///
    /// **`inverse` UNROTATES** — the angle is negated — for the one place a
    /// value carries a key's rope: MLA's shared latent is both key and value,
    /// so the attention output's rope lanes come back rotated by the query's
    /// own position and the reference undoes it (`apply_rotary_emb(o[...,
    /// -rd:], freqs, inverse=True)`, the official `Attention.forward`).
    ///
    /// **`yarn` IS THE LAYER'S OWN RULE, NOT THE MODEL'S.** DeepSeek-V4-Flash
    /// ropes its compressor layers at `compress_rope_theta` WITH the YaRN
    /// ramp and its pure sliding-window layers at `rope_theta` without one
    /// (`if self.compress_ratio: original_seq_len, rope_theta =
    /// args.original_seq_len, args.compress_rope_theta else 0, args.rope_theta`),
    /// so the ramp rides the op beside the theta and is `None` where the
    /// layer states none.
    RopePartialLast {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
        q_out: ValueId,
    },
    RopeYarn {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        q_out: ValueId,
        k_out: ValueId,
    },
    GateSigmoidMul {
        x: ValueId,
        gate: ValueId,
        x_out: ValueId,
    },
    /// `x[:, h·head_dim + j] *= scale · sigmoid(gate[:, h])`, in place on
    /// `x` — a per-HEAD gate, one logit per head per row, broadcast across
    /// the head's channels. `x` is `[rows, heads·head_dim]`, `gate` is
    /// `[rows, heads]` at `x`'s dtype. The `scale` is the constant in front
    /// of the sigmoid (LTX-2's gated attention is `out · 2σ(W·x)`; a plain
    /// gate states `1.0`). fp32 sigmoid and product, one rounding at the
    /// store.
    GateSigmoidMulHeads {
        x: ValueId,
        gate: ValueId,
        head_dim: u32,
        scale: f32,
        x_out: ValueId,
    },
    // Hyper-connections: residual streams expanded, mixed by learned gates, and
    // folded back layer by layer.
    /// Tiles `x` across `streams` residual streams.
    HcExpand {
        x: ValueId,
        streams: u32,
        y: ValueId,
    },
    HcRmsnormF32 {
        streams: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// The per-token mix row: `rmsnorm(streams) . hc_fn^T`, the row
    /// [`Self::HcGates`] splits into pre, post and the combiner. Not
    /// [`Linear::Matmul`](crate::ops::Linear): kept f32, too sensitive for bf16.
    /// The row is as wide as the plane says: `2M + M²` for a layer's
    /// `{attn,ffn}_hc.fn`, `M` for the trunk's `hc_head.fn` ([`Self::HcCollapse`]).
    HcProject {
        normed: ValueId,
        weight: ValueId,
        stream_count: u32,
        mixes: ValueId,
    },
    /// Computes the layer input `x` plus the post/comb mixing matrices.
    ///
    /// `normed` is the mix row [`Self::HcProject`] lands — `[N, 2M + M²]`,
    /// which is the stride this op has always read its operand at.
    HcGates {
        normed: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        x: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
    },
    /// Mixes the layer output back into the streams under the gate matrices.
    HcFold {
        x: ValueId,
        streams: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
        y: ValueId,
    },
    /// The trunk collapse (`hc_head`): the `M` streams folded into the row the
    /// final norm reads under `M` sigmoid gates off the `[N, M]` mix row
    /// [`Self::HcProject`] lands through `hc_head.fn` — no post, no combiner,
    /// no Sinkhorn. `y[h] = Σₛ (σ(mixes[s]·scale[0] + base[s]) + hc_eps) · streams[s·H + h]`.
    HcCollapse {
        mixes: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        hc_eps: f32,
        y: ValueId,
    },

    // The gated-residual flavor (qwen4): mixes through per-element sigmoid
    // gates instead of a sinkhorn-normalized matrix. The GEMMs stay
    // `linear.matmul` nodes; these two ops are the arithmetic around them.
    /// `y[h] = mean_s(sigmoid(gates[s*H + h]) * normed[s*H + h])` — one
    /// `hidden`-wide layer input mixed out of `streams` normed residual
    /// streams under per-element sigmoid gates.
    HcMix {
        gates: ValueId,
        normed: ValueId,
        streams: u32,
        y: ValueId,
    },
    /// `hyper[s*H + h] += 2*sigmoid(gates[s] / streams) * o[h]` — the layer
    /// output injected back into every stream under its own scalar gate. In
    /// place on `hyper`.
    HcInject {
        o: ValueId,
        gates: ValueId,
        streams: u32,
        hyper: ValueId,
        hyper_out: ValueId,
    },
    /// PLE gate (qwen4): `y[s*H+h] = sigmoid(sgn(d)*sqrt(|d|)) * value[h]`
    /// where `d = sum_j key[s*H+j] * query[s*H+j] / sqrt(H)`.
    PleGate {
        key: ValueId,
        query: ValueId,
        value: ValueId,
        streams: u32,
        y: ValueId,
    },

    // The generative families' conditioning algebra (D6): adaptive
    // modulation from a per-lane or per-token vector, the gated residual it
    // pairs with, and the bare activations and binary ops a DiT's
    // embedders and heads are written in.
    /// Adaptive modulation: `y = form(x, m)` where `m` is a `[Lanes,
    /// k·width]` vector broadcast over each lane's rows through
    /// `lane_of_row` (`GeomKind::RequestOfToken`, `[Tokens]` i32), or a
    /// `[Tokens, k·width]` per-token vector with `lane_of_row: None`. `k`
    /// is the form's ([`ModulateForm`]); the halves of `m` are laid out
    /// `[s | b]` — the first `width` columns scale, the next shift — and a
    /// family reorders its modulation linear's rows at import to say so.
    /// `x`, `y` share a type; `m` rides the activation dtype. Arithmetic in
    /// fp32, rounded once at the store.
    Modulate {
        x: ValueId,
        m: ValueId,
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        y: ValueId,
    },
    /// The gated residual fold: `r += g · y`, in place on `r`, where `g` is
    /// a `[Lanes, width]` gate broadcast through `lane_of_row`
    /// (`GeomKind::RequestOfToken`) or a `[Tokens, width]` per-token gate
    /// (`lane_of_row: None`). The adaLN-Zero `gate_msa * attn(...)` step.
    /// fp32 product and sum, rounded once.
    GatedResidualAdd {
        r: ValueId,
        g: ValueId,
        y: ValueId,
        lane_of_row: Option<ValueId>,
        r_out: ValueId,
    },
    /// `normed = norm(x)` (a scale-free norm, [`NormKind`]) then `y =
    /// form(normed, m)`: the adaLN pre-norm and its modulation, one launch
    /// where the trace lands two. `normed` is still written as its own
    /// launch would write it. Written by [`crate::fuse`], never traced.
    NormModulate {
        x: ValueId,
        norm: NormKind,
        normed: ValueId,
        m: ValueId,
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        y: ValueId,
    },
    /// `r += g · y` in place, then `normed = norm(r)`, then `out =
    /// form(normed, m)`: the deferred-residual form the FLUX.2 / LTX
    /// references run between a block's attention and its MLP — three
    /// traced nodes, one launch, two outputs a reader wants (`r_out`, the
    /// stream, and `out`, the modulated input of the next sub-block) plus
    /// the `normed` intermediate written as traced. `lane_of_row` serves
    /// both the gate and the modulation: either both are per lane or both
    /// per token. Written by [`crate::fuse`], never traced.
    GatedResidualNormModulate {
        r: ValueId,
        g: ValueId,
        y: ValueId,
        lane_of_row: Option<ValueId>,
        r_out: ValueId,
        norm: NormKind,
        normed: ValueId,
        m: ValueId,
        form: ModulateForm,
        out: ValueId,
    },
    /// The sinusoidal timestep embedding: `t` is `[rows, 1]` f32 (`rows`
    /// being `Lanes` for a per-lane timestep, `Tokens` for a per-token
    /// one), `y` is `[rows, dim]` f32 with `half = dim / 2`,
    /// `freq_i = exp(-ln(max_period) · i / half)` for `i < half`,
    /// `arg_i = scale · t · freq_i`, and `y = [sin(arg) | cos(arg)]`, or
    /// `[cos | sin]` under `flip_sin_cos` — diffusers'
    /// `get_timestep_embedding` at `downscale_freq_shift = 0`, which is what
    /// every target family runs. `dim` is even. All fp32.
    Sinusoid {
        t: ValueId,
        dim: u32,
        max_period: f32,
        flip_sin_cos: bool,
        scale: f32,
        y: ValueId,
    },
    /// The dense relative-position bias table a bidirectional encoder
    /// layer's attention adds to its logits
    /// ([`RaggedMask::RelativeBias`](super::attn::RaggedMask::RelativeBias)):
    /// `y[h][d + max_len − 1] = embedding[bucket(d)][h]` for every signed
    /// distance `d = kj − qi` in `−(max_len − 1) ..= max_len − 1`, where
    /// `bucket` is the T5 relative-position bucket function — Hugging Face's
    /// `_relative_position_bucket(relative_position = memory_position −
    /// context_position, bidirectional, num_buckets, max_distance)`:
    ///
    /// ```text
    /// bucket = 0
    /// if bidirectional: num_buckets /= 2; bucket += (d > 0) · num_buckets; n = |d|
    /// else:             n = −min(d, 0)
    /// max_exact = num_buckets / 2
    /// if n < max_exact: bucket + n
    /// else: bucket + min(num_buckets − 1, max_exact +
    ///           trunc(ln(n / max_exact) / ln(max_distance / max_exact) · (num_buckets − max_exact)))
    /// ```
    ///
    /// (the logarithms' ratio in f32, as torch computes it). `embedding` is
    /// the checkpoint's `[num_buckets, heads]` plane
    /// (`relative_attention_bias.weight`, bf16 or f32) and `y` is
    /// `[heads, 2·max_len − 1]` f32, a constant of the plan: computed from a
    /// weight alone, it depends on no row of any axis. `max_len` is the
    /// longest segment the table answers exactly.
    RelativeBucketBias {
        embedding: ValueId,
        max_len: u32,
        num_buckets: u32,
        max_distance: f32,
        bidirectional: bool,
        y: ValueId,
    },
    /// `x = x · sigmoid(x)`, in place. [`SiluScaled`](Elementwise::SiluScaled)
    /// at `s = 1`, named so a text does not spell a scale it does not have.
    Silu {
        x: ValueId,
        x_out: ValueId,
    },
    /// `x = gelu(x)`, in place: the erf form, or the tanh approximation
    /// when `tanh` is set (`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`).
    Gelu {
        x: ValueId,
        tanh: bool,
        x_out: ValueId,
    },
    /// `x = tanh(x)`, in place.
    Tanh {
        x: ValueId,
        x_out: ValueId,
    },
    /// `z = x · y`, two activations of one type, fresh output. fp32 product,
    /// rounded once.
    Mul {
        x: ValueId,
        y: ValueId,
        z: ValueId,
    },
    /// `z = x + y`, two activations of one type, fresh output — unlike
    /// [`ResidualAdd`](Elementwise::ResidualAdd), which folds in place.
    Add {
        x: ValueId,
        y: ValueId,
        z: ValueId,
    },
    /// Rotary embedding over up to four position axes with a theta per axis
    /// (D7), in place on one `[rows, heads·head_dim]` rectangle — called
    /// once for `q` and once for `k` (LTX's a2v rotates the two by
    /// different positions). `positions` is `[rows, axes]` f32
    /// (`RuntimeInput::AxisPositions`); `dims[a]` is axis `a`'s CHANNEL count
    /// (`0` past the last axis; `Σ dims == rotary_dim <= head_dim`, the
    /// tail `head_dim - rotary_dim` of every head passing through);
    /// `thetas[a]` its base. Pair `i` of axis `a` turns by `positions[a] ·
    /// thetas[a]^(-2i / dims[a])`, angles in fp32. Which channels pair `i`
    /// joins is [`form`](RopeForm).
    RopeAxes {
        x: ValueId,
        positions: ValueId,
        dims: [u32; 4],
        thetas: [f32; 4],
        form: RopeForm,
        rotary_dim: u32,
        head_dim: u32,
        x_out: ValueId,
    },
}

/// Which arithmetic a [`Modulate`](Elementwise::Modulate) applies, and how
/// many `width`-wide slices (`k`) its vector carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulateForm {
    /// `y = x · (1 + s) + b`, `m = [s | b]`, `k = 2` — adaLN's shift/scale.
    ScaleShift,
    /// `y = x · (1 + s)`, `k = 1`.
    Scale,
    /// `y = tanh(g) · x`, `k = 1` — Z-Image's gated form.
    TanhGate,
}

impl ModulateForm {
    /// How many `width`-wide slices the modulation vector carries.
    #[must_use]
    pub fn slices(self) -> u64 {
        match self {
            ModulateForm::ScaleShift => 2,
            ModulateForm::Scale | ModulateForm::TanhGate => 1,
        }
    }
}

/// The scale-free norm a fused [`NormModulate`](Elementwise::NormModulate)
/// runs before its modulation: the same arithmetic as the traced variant it
/// replaces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormKind {
    /// [`LayernormNoScale`](Elementwise::LayernormNoScale): centred, whole row.
    Layernorm { eps: f32 },
    /// [`RmsnormNoScale`](Elementwise::RmsnormNoScale): per `head_dim` group.
    Rmsnorm { head_dim: u32, eps: f32 },
}

/// Which channels pair `i` of an axis block joins in a
/// [`RopeAxes`](Elementwise::RopeAxes). Every form keeps each axis's
/// `dims[a]` channels as one contiguous block `b_a..b_a + dims[a]` of the
/// rotated prefix; they differ in the pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopeForm {
    /// Adjacent pairs within the block: `(b + 2i, b + 2i + 1)` — FLUX's and
    /// Z-Image's complex-pair layout.
    Interleaved,
    /// `rotate_half` over the whole rotated prefix: pair `p` of
    /// `[0, rotary_dim/2)` is `(p, p + rotary_dim/2)`, and the axis owning
    /// pair `p` is the one whose block of `dims[a]/2` pairs contains it —
    /// MiniMax's layout (96 of 128 channels rotated).
    Neox,
    /// `rotate_half` WITHIN each block: pair `i` of axis `a` is `(b + i,
    /// b + dims[a]/2 + i)` — Wan's layout, and [`MropeForm::Split`]'s
    /// pairing.
    Split,
    /// ONE frequency ladder across the whole `[rows, heads·rotary_dim]`
    /// rectangle, the axes handed out round-robin along it — LTX-2's
    /// layout, which no per-head rule states.
    ///
    /// The row's angle slots are numbered `g = head · rotary_dim/2 + i`
    /// across every head. The first `pad` slots turn by nothing
    /// (`cos = 1`, `sin = 0`); slot `g >= pad` belongs to axis
    /// `a = (g − pad) mod axes` at ladder index `f = (g − pad) div axes`
    /// and turns by `positions[a] · thetas[a]^(f / (F_a − 1))` — a
    /// POSITIVE, endpoint-inclusive exponent (`torch.linspace(0, 1, F_a)`),
    /// where `F_a = dims[a]/2` is how many frequencies axis `a` owns over
    /// the WHOLE row. So `dims[a]` here counts the ROW's channels, not a
    /// head's, and `pad = (heads · rotary_dim − Σ dims) / 2`; every live
    /// axis owns the same count, and `rotary_dim == head_dim` because the
    /// pairing is `rotate_half` within each head (`(i, i + rotary_dim/2)`).
    /// The positions are the reference's already-normalised coordinates
    /// (`(2·coord/max − 1) · π/2`), which is why they are fractional and
    /// signed.
    SplitLadder,
}

/// Which activation gates a [`RmsnormGated`](Elementwise::RmsnormGated) —
/// the checkpoint's `output_gate_type`, as a form rather than a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateActivation {
    Silu,
    Sigmoid,
}

/// Which section layout a [`RopeMrope`](Elementwise::RopeMrope) turns by —
/// how `(t, h, w)` frequency pairs are handed out. Both arms pair
/// `(d, d + head_dim/2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MropeForm {
    /// The trunk's (`mrope_interleaved: true`): pairs alternate `t, h, w, ...`;
    /// pair `p` turns at `theta^(-2p/head_dim)` whichever axis it took.
    Interleaved,
    /// The tower's (`apply_rotary_pos_emb_vision`): each section is a
    /// contiguous block of pairs, each restarting the frequency ladder
    /// (`sections[0] == 0`, no time axis).
    Blocked,
    /// Gemma's tower (`apply_multidimensional_rope`, mlx_vlm's spelling of
    /// the JAX original): each section owns a contiguous CHANNEL block of
    /// `2 · s` channels, and pair `i` of the block is `(x[b + i], x[b + s +
    /// i])` — `rotate_half` WITHIN the block, never across axes — turning at
    /// `theta^(-i / s)`. [`Blocked`](MropeForm::Blocked) pairs `x[p]` with
    /// `x[p + head_dim/2]` across the whole head and picks the axis by `p`;
    /// the same sections, a different pairing, and a picture rotated the
    /// other way is a picture whose patches have lost their places.
    Split,
}

impl Operands for Elementwise {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHead { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHeadPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormGroupedPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormNoScale { x, .. } => sink.push(*x),
            Self::LayernormNoScale { x, .. } => sink.push(*x),
            Self::Layernorm { x, weight, bias, .. } => sink.extend([*x, *weight, *bias]),
            Self::Clamp { x, .. } => sink.push(*x),
            Self::ClampLearned { x, lo, hi, .. } => sink.extend([*x, *lo, *hi]),
            Self::RmsnormGated { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::RmsnormGatedBy { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::ResidualAdd { x, y, .. } => sink.extend([*x, *y]),
            Self::ResidualAddRmsnorm { x, y, weight, .. } => sink.extend([*x, *y, *weight]),
            Self::RmsnormResidualAdd {
                x,
                weight,
                y,
                scale,
                post,
                ..
            } => {
                sink.extend([*x, *weight, *y]);
                if let Some((s, _)) = scale {
                    sink.push(*s);
                }
                if let Some(post) = post {
                    sink.push(post.weight);
                }
            }
            Self::EmbedScaleAdd { ids, table, y, .. } => sink.extend([*ids, *table, *y]),
            Self::EmbedScaleAddSelect {
                ids,
                table,
                stacked,
                ..
            } => sink.extend([*ids, *table, *stacked]),
            Self::AddBias { bias, out, .. } => sink.extend([*bias, *out]),
            Self::Standardize { x, bias, scale, .. } => sink.extend([*x, *bias, *scale]),
            Self::MulScalar { x, .. } => sink.push(*x),
            Self::SiluScaled { x, .. } => sink.push(*x),
            Self::Scale { s, x, .. } => sink.extend([*s, *x]),
            Self::ResBlend { prefix, blocks, weight, proj, .. } => {
                sink.push(*prefix);
                sink.extend_from_slice(blocks);
                sink.push(*weight);
                sink.push(*proj);
            }
            Self::RopeFull { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartial { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopeMrope { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartialQ { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RmsnormRopePartialQ {
                x,
                weight,
                positions,
                ..
            } => sink.extend([*x, *weight, *positions]),
            Self::RopePartialLast { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopeYarn { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::GateSigmoidMul { x, gate, .. } => sink.extend([*x, *gate]),
            Self::GateSigmoidMulHeads { x, gate, .. } => sink.extend([*x, *gate]),
            Self::HcExpand { x, .. } => sink.push(*x),
            Self::HcRmsnormF32 { streams, .. } => sink.push(*streams),
            Self::HcProject { normed, weight, .. } => sink.extend([*normed, *weight]),
            Self::HcGates { normed, streams, scale, base, .. } => {
                sink.extend([*normed, *streams, *scale, *base]);
            }
            Self::HcFold { x, streams, post_mix, comb_mix, .. } => {
                sink.extend([*x, *streams, *post_mix, *comb_mix]);
            }
            Self::HcCollapse { mixes, streams, scale, base, .. } => {
                sink.extend([*mixes, *streams, *scale, *base]);
            }
            Self::HcMix { gates, normed, .. } => sink.extend([*gates, *normed]),
            Self::HcInject { o, gates, hyper, .. } => sink.extend([*o, *gates, *hyper]),
            Self::PleGate { key, query, value, .. } => sink.extend([*key, *query, *value]),
            Self::Modulate { x, m, lane_of_row, .. } => {
                sink.extend([*x, *m]);
                sink.extend(*lane_of_row);
            }
            Self::GatedResidualAdd { r, g, y, lane_of_row, .. } => {
                sink.extend([*r, *g, *y]);
                sink.extend(*lane_of_row);
            }
            Self::NormModulate { x, m, lane_of_row, .. } => {
                sink.extend([*x, *m]);
                sink.extend(*lane_of_row);
            }
            Self::GatedResidualNormModulate { r, g, y, m, lane_of_row, .. } => {
                sink.extend([*r, *g, *y, *m]);
                sink.extend(*lane_of_row);
            }
            Self::Sinusoid { t, .. } => sink.push(*t),
            Self::RelativeBucketBias { embedding, .. } => sink.push(*embedding),
            Self::Silu { x, .. } => sink.push(*x),
            Self::Gelu { x, .. } => sink.push(*x),
            Self::Tanh { x, .. } => sink.push(*x),
            Self::Mul { x, y, .. } => sink.extend([*x, *y]),
            Self::Add { x, y, .. } => sink.extend([*x, *y]),
            Self::RopeAxes { x, positions, .. } => sink.extend([*x, *positions]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { y, .. } => sink.push(*y),
            Self::RmsnormPerHead { y, .. } => sink.push(*y),
            Self::RmsnormPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormPerHeadPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormGroupedPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormNoScale { y, .. } => sink.push(*y),
            Self::LayernormNoScale { y, .. } => sink.push(*y),
            Self::Layernorm { y, .. } => sink.push(*y),
            Self::Clamp { x_out, .. } => sink.push(*x_out),
            Self::ClampLearned { x_out, .. } => sink.push(*x_out),
            Self::RmsnormGated { y, .. } => sink.push(*y),
            Self::RmsnormGatedBy { y, .. } => sink.push(*y),
            Self::ResidualAdd { y_out, .. } => sink.push(*y_out),
            Self::ResidualAddRmsnorm { y_out, out, .. } => sink.extend([*y_out, *out]),
            Self::RmsnormResidualAdd {
                t,
                y_out,
                scale,
                post,
                ..
            } => {
                sink.extend([*t, *y_out]);
                if let Some((_, scaled)) = scale {
                    sink.push(*scaled);
                }
                if let Some(post) = post {
                    sink.push(post.out);
                }
            }
            Self::EmbedScaleAdd {
                e,
                e_scaled,
                y_out,
                y_scaled,
                ..
            } => sink.extend([*e, *e_scaled, *y_out, *y_scaled]),
            Self::EmbedScaleAddSelect {
                e,
                e_scaled,
                y_out,
                y_scaled,
                ..
            } => sink.extend([*e, *e_scaled, *y_out, *y_scaled]),
            Self::AddBias { out_out, .. } => sink.push(*out_out),
            Self::Standardize { x_out, .. } => sink.push(*x_out),
            Self::MulScalar { x_out, .. } => sink.push(*x_out),
            Self::SiluScaled { x_out, .. } => sink.push(*x_out),
            Self::Scale { x_out, .. } => sink.push(*x_out),
            Self::ResBlend { y, .. } => sink.push(*y),
            Self::RopeFull { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartial { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopeMrope { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartialQ { q_out, .. } => sink.push(*q_out),
            Self::RmsnormRopePartialQ { y, q_out, .. } => sink.extend([*y, *q_out]),
            Self::RopePartialLast { q_out, .. } => sink.push(*q_out),
            Self::RopeYarn { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::GateSigmoidMul { x_out, .. } => sink.push(*x_out),
            Self::GateSigmoidMulHeads { x_out, .. } => sink.push(*x_out),
            Self::HcExpand { y, .. } => sink.push(*y),
            Self::HcRmsnormF32 { y, .. } => sink.push(*y),
            Self::HcProject { mixes, .. } => sink.push(*mixes),
            Self::HcGates { x, post_mix, comb_mix, .. } => sink.extend([*x, *post_mix, *comb_mix]),
            Self::HcFold { y, .. } => sink.push(*y),
            Self::HcCollapse { y, .. } => sink.push(*y),
            Self::HcMix { y, .. } => sink.push(*y),
            Self::HcInject { hyper_out, .. } => sink.push(*hyper_out),
            Self::PleGate { y, .. } => sink.push(*y),
            Self::Modulate { y, .. } => sink.push(*y),
            Self::GatedResidualAdd { r_out, .. } => sink.push(*r_out),
            Self::NormModulate { normed, y, .. } => sink.extend([*normed, *y]),
            Self::GatedResidualNormModulate { r_out, normed, out, .. } => {
                sink.extend([*r_out, *normed, *out]);
            }
            Self::Sinusoid { y, .. } => sink.push(*y),
            Self::RelativeBucketBias { y, .. } => sink.push(*y),
            Self::Silu { x_out, .. } => sink.push(*x_out),
            Self::Gelu { x_out, .. } => sink.push(*x_out),
            Self::Tanh { x_out, .. } => sink.push(*x_out),
            Self::Mul { z, .. } => sink.push(*z),
            Self::Add { z, .. } => sink.push(*z),
            Self::RopeAxes { x_out, .. } => sink.push(*x_out),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Rmsnorm { .. } => {}
            Self::RmsnormPerHead { .. } => {}
            Self::RmsnormPlusOne { .. } => {}
            Self::RmsnormPerHeadPlusOne { .. } => {}
            Self::RmsnormGroupedPlusOne { .. } => {}
            Self::RmsnormNoScale { .. } => {}
            Self::LayernormNoScale { .. } => {}
            Self::Layernorm { .. } => {}
            Self::Clamp { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ClampLearned { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::RmsnormGated { .. } => {}
            Self::RmsnormGatedBy { .. } => {}
            Self::ResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::ResidualAddRmsnorm { y_out, y, .. } => sink.push((*y_out, *y)),
            // Only the fold is in place: the scaled row and the embed rows
            // are fresh outputs, since an alias may name only an input.
            Self::RmsnormResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAddSelect { .. } => {}
            Self::AddBias { out_out, out, .. } => sink.push((*out_out, *out)),
            Self::Standardize { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::MulScalar { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::SiluScaled { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Scale { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ResBlend { .. } => {}
            Self::RopeFull { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::RopePartial { q_out, q, k_out, k, .. } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopeMrope { q_out, q, k_out, k, .. } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopePartialQ { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RmsnormRopePartialQ { q_out, y, .. } => sink.push((*q_out, *y)),
            Self::RopePartialLast { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopeYarn { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::GateSigmoidMul { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::GateSigmoidMulHeads { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::HcExpand { .. } => {}
            Self::HcRmsnormF32 { .. } => {}
            Self::HcProject { .. } => {}
            Self::HcGates { .. } => {}
            Self::HcFold { .. } => {}
            Self::HcCollapse { .. } => {}
            Self::HcMix { .. } => {}
            Self::HcInject { hyper_out, hyper, .. } => sink.push((*hyper_out, *hyper)),
            Self::PleGate { .. } => {}
            Self::Modulate { .. } => {}
            Self::GatedResidualAdd { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::NormModulate { .. } => {}
            // Only the fold is in place; the normed row and the modulated
            // output are fresh, since an alias may name only an input.
            Self::GatedResidualNormModulate { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::Sinusoid { .. } => {}
            Self::RelativeBucketBias { .. } => {}
            Self::Silu { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Gelu { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Tanh { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Mul { .. } => {}
            Self::Add { .. } => {}
            Self::RopeAxes { x_out, x, .. } => sink.push((*x_out, *x)),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Rmsnorm { .. } => "elementwise.rmsnorm",
            Self::RmsnormPerHead { .. } => "elementwise.rmsnorm_per_head",
            Self::RmsnormPlusOne { .. } => "elementwise.rmsnorm_plus_one",
            Self::RmsnormPerHeadPlusOne { .. } => "elementwise.rmsnorm_per_head_plus_one",
            Self::RmsnormGroupedPlusOne { .. } => "elementwise.rmsnorm_grouped_plus_one",
            Self::RmsnormNoScale { .. } => "elementwise.rmsnorm_no_scale",
            Self::LayernormNoScale { .. } => "elementwise.layernorm_no_scale",
            Self::Layernorm { .. } => "elementwise.layernorm",
            Self::Clamp { .. } => "elementwise.clamp",
            Self::ClampLearned { .. } => "elementwise.clamp_learned",
            Self::RmsnormGated { .. } => "elementwise.rmsnorm_gated",
            Self::RmsnormGatedBy { .. } => "elementwise.rmsnorm_gated_by",
            Self::ResidualAdd { .. } => "elementwise.residual_add",
            Self::ResidualAddRmsnorm { .. } => "elementwise.residual_add_rmsnorm",
            Self::RmsnormResidualAdd { .. } => "elementwise.rmsnorm_residual_add",
            Self::EmbedScaleAdd { .. } => "elementwise.embed_scale_add",
            Self::EmbedScaleAddSelect { .. } => "elementwise.embed_scale_add_select",
            Self::AddBias { .. } => "elementwise.add_bias",
            Self::Standardize { .. } => "elementwise.standardize",
            Self::MulScalar { .. } => "elementwise.mul_scalar",
            Self::SiluScaled { .. } => "elementwise.silu_scaled",
            Self::Scale { .. } => "elementwise.scale",
            Self::ResBlend { .. } => "elementwise.res_blend",
            Self::RopeFull { .. } => "elementwise.rope_full",
            Self::RopePartial { .. } => "elementwise.rope_partial",
            Self::RopeMrope { .. } => "elementwise.rope_mrope",
            Self::RopePartialQ { .. } => "elementwise.rope_partial_q",
            Self::RmsnormRopePartialQ { .. } => "elementwise.rmsnorm_rope_partial_q",
            Self::RopePartialLast { .. } => "elementwise.rope_partial_last",
            Self::RopeYarn { .. } => "elementwise.rope_yarn",
            Self::GateSigmoidMul { .. } => "elementwise.gate_sigmoid_mul",
            Self::GateSigmoidMulHeads { .. } => "elementwise.gate_sigmoid_mul_heads",
            Self::HcExpand { .. } => "elementwise.hc_expand",
            Self::HcRmsnormF32 { .. } => "elementwise.hc_rmsnorm_f32",
            Self::HcProject { .. } => "elementwise.hc_project",
            Self::HcGates { .. } => "elementwise.hc_gates",
            Self::HcFold { .. } => "elementwise.hc_fold",
            Self::HcCollapse { .. } => "elementwise.hc_collapse",
            Self::HcMix { .. } => "elementwise.hc_mix",
            Self::HcInject { .. } => "elementwise.hc_inject",
            Self::PleGate { .. } => "elementwise.ple_gate",
            Self::Modulate { .. } => "elementwise.modulate",
            Self::GatedResidualAdd { .. } => "elementwise.gated_residual_add",
            Self::NormModulate { .. } => "elementwise.norm_modulate",
            Self::GatedResidualNormModulate { .. } => "elementwise.gated_residual_norm_modulate",
            Self::Sinusoid { .. } => "elementwise.sinusoid",
            Self::RelativeBucketBias { .. } => "elementwise.relative_bucket_bias",
            Self::Silu { .. } => "elementwise.silu",
            Self::Gelu { .. } => "elementwise.gelu",
            Self::Tanh { .. } => "elementwise.tanh",
            Self::Mul { .. } => "elementwise.mul",
            Self::Add { .. } => "elementwise.add",
            Self::RopeAxes { .. } => "elementwise.rope_axes",
        }
    }
}
