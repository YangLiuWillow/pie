//! Reading `tencent/HunyuanImage-3.0` — one flat namespace of 5 161
//! tensors across 32 shards (`model.safetensors.index.json`) — into this
//! family's own scheme.
//!
//! Four places the checkpoint's spelling and the plan's part company:
//!
//! 1. **`qkv_proj` is head-interleaved.** The reference reshapes it
//!    `[kv_heads, groups + 2, head_dim, hidden]` and splits `[groups, 1,
//!    1]` on axis 1, so per KV group the four query heads come first, then
//!    K, then V. This text wants `[q | k | v]`. One `Expr::gather` over
//!    axis 0 states the whole rearrangement — no slicing of a strided view,
//!    no concatenation.
//! 2. **The same gather carries the 2-D rope's channel permutation** (see
//!    [`super::model::rope_x_scale`]): each head's `y` pairs move to the
//!    low half and its `x` pairs to the high half, in `RopeForm::Split`'s
//!    own pairing. The two QK-norm gains are permuted with them — the norm
//!    runs AFTER the rotation, so its gain is indexed by rotated channel.
//! 3. **`gate_and_up_proj`'s halves are swapped.** The reference computes
//!    `down(x1 · silu(x2))` with `x1` the FIRST half; `linear.mlp_swiglu`
//!    computes `silu(first) · second`. Swapping at import keeps the fused
//!    kernel and costs nothing at serving time.
//! 4. **`timestep_emb`'s second linear is stacked twice.** The plan lands
//!    the `<timestep>` row with one `ScaleShift` over a zero row, which
//!    reads `[t_emb | t_emb]`; the checkpoint stores `t_emb`'s projection
//!    once. (wan_2's `head_proj` is the same device.)
//!
//! Everything else is a rename. The 64 routed experts of a layer are 128
//! separate tensors, stacked into the two `[64, ·, ·]` banks the routed
//! matmul reads; on the flagship rows those banks are declared `U8g64`, so
//! the ladder encodes them on the way in (design D10) while every dense
//! plane stays bf16.
//!
//! Not read here: `vae.*` (1.26 B fp32) and `vision_model.*` /
//! `vision_aligner.*` (0.45 B) — this text traces neither, and a plane no
//! trace names is a plane no artifact needs.

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, encoding, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{Conv, Embedder, GroupNorm, Linear, Model, ResBlock};

impl Model {
    /// The whole trunk and image head, from the HF spelling.
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        let d = &self.dims;

        b.read(&self.embed, "model.wte.weight")?;
        b.read(&self.head, "lm_head.weight")?;
        b.read(&self.final_norm, "model.ln_f.weight")?;
        // The one plane no checkpoint carries: `[+1 | -1]`, the two signs
        // of the canvas lane's `special` flag broadcast to a row.
        signs(&mut b, src, &self.ones, "model.ln_f.weight")?;

        let head_dim = d.head_dim;
        let groups = d.q_heads / d.kv_heads;
        let q_perm = head_permutation(d.q_heads, head_dim);
        let k_perm = head_permutation(d.kv_heads, head_dim);
        let rope_perm = rope_channels(head_dim);

        for (l, w) in self.layers.iter().enumerate() {
            let at = |tail: &str| format!("model.layers.{l}.{tail}");
            b.read(&w.attn_norm, at("input_layernorm.weight"))?;
            b.read(&w.mlp_norm, at("post_attention_layernorm.weight"))?;
            b.read_expr(
                &w.qkv,
                Expr::src(at("self_attn.qkv_proj.weight"))
                    .gather(0, qkv_rows(d.kv_heads, groups, head_dim, &q_perm, &k_perm)),
            )?;
            b.read_expr(
                &w.q_norm,
                Expr::src(at("self_attn.query_layernorm.weight")).gather(0, rope_perm.clone()),
            )?;
            b.read_expr(
                &w.k_norm,
                Expr::src(at("self_attn.key_layernorm.weight")).gather(0, rope_perm.clone()),
            )?;
            b.read(&w.o_proj, at("self_attn.o_proj.weight"))?;
            b.read(&w.router, at("mlp.gate.wg.weight"))?;

            let inter = i64::from(self.moe_inter);
            let hidden = i64::from(d.hidden);
            let shared = i64::from(self.shared_inter);
            b.read_expr(
                &w.shared_gate_up,
                swap_halves(
                    Expr::src(at("mlp.shared_mlp.gate_and_up_proj.weight")),
                    shared,
                ),
            )?;
            b.read(&w.shared_down, at("mlp.shared_mlp.down_proj.weight"))?;

            let stored = stored_encoding(src, &at("mlp.experts.0.down_proj.weight"))?;
            b.read_expr(
                &w.experts_gate_up,
                Expr::concat(
                    0,
                    (0..d.experts)
                        .map(|e| {
                            slab(
                                swap_halves(
                                    Expr::src(at(&format!(
                                        "mlp.experts.{e}.gate_and_up_proj.weight"
                                    ))),
                                    inter,
                                ),
                                vec![1, 2 * inter, hidden],
                                stored.clone(),
                            )
                        })
                        .collect(),
                ),
            )?;
            b.read_expr(
                &w.experts_down,
                Expr::concat(
                    0,
                    (0..d.experts)
                        .map(|e| {
                            slab(
                                Expr::src(at(&format!("mlp.experts.{e}.down_proj.weight"))),
                                vec![1, hidden, inter],
                                stored.clone(),
                            )
                        })
                        .collect(),
                ),
            )?;
        }

        // ---- the three timestep embedders ---------------------------------
        // `timestep_emb`'s second linear is doubled; the other two are read
        // as they are stored.
        b.read(&self.timestep_emb.mlp_in.w, "timestep_emb.mlp.0.weight")?;
        b.read(&self.timestep_emb.mlp_in.bias, "timestep_emb.mlp.0.bias")?;
        b.read_expr(
            &self.timestep_emb.mlp_out.w,
            doubled(Expr::src("timestep_emb.mlp.2.weight")),
        )?;
        b.read_expr(
            &self.timestep_emb.mlp_out.bias,
            doubled(Expr::src("timestep_emb.mlp.2.bias")),
        )?;
        embedder(&mut b, &self.time_embed, "time_embed")?;
        embedder(&mut b, &self.time_embed_2, "time_embed_2")?;

        // ---- the conv image head ------------------------------------------
        conv(
            &mut b,
            src,
            &self.patch_embed.conv_in,
            "patch_embed.model.0",
        )?;
        resblock(&mut b, src, &self.patch_embed.res, "patch_embed.model.1")?;
        resblock(&mut b, src, &self.final_layer.res, "final_layer.model.0")?;
        group_norm(&mut b, &self.final_layer.norm_out, "final_layer.model.1.0")?;
        conv(
            &mut b,
            src,
            &self.final_layer.conv_out,
            "final_layer.model.1.2",
        )?;

        Ok(b.build())
    }
}

/// `Linear(256 → hidden)` then `Linear(hidden → out)`, both as stored.
fn embedder(b: &mut Builder, e: &Embedder, prefix: &str) -> Result<(), Error> {
    b.read(&e.mlp_in.w, format!("{prefix}.mlp.0.weight"))?;
    b.read(&e.mlp_in.bias, format!("{prefix}.mlp.0.bias"))?;
    b.read(&e.mlp_out.w, format!("{prefix}.mlp.2.weight"))?;
    b.read(&e.mlp_out.bias, format!("{prefix}.mlp.2.bias"))
}

/// A `nn.Conv2d`'s `[C_out, C_in, kh, kw]` flattened to the `[C_out,
/// C_in·taps]` tap-major plane `spatial.conv3d` reads, and its f32 bias.
fn conv(b: &mut Builder, src: &ztensor::Source, c: &Conv, name: &str) -> Result<(), Error> {
    let shape = extents(&c.w);
    let stored = stored_encoding(src, &format!("{name}.weight"))?;
    b.read_expr(
        &c.w,
        Expr::src(format!("{name}.weight")).transmute(TensorType::new(shape, stored)),
    )?;
    b.read(&c.bias, format!("{name}.bias"))
}

fn group_norm(b: &mut Builder, g: &GroupNorm, name: &str) -> Result<(), Error> {
    b.read(&g.weight, format!("{name}.weight"))?;
    b.read(&g.bias, format!("{name}.bias"))
}

/// One `ResBlock`: `in_layers.{0,2}`, `emb_layers.1`, `out_layers.{0,3}`
/// and the 1×1 `skip_connection`.
fn resblock(
    b: &mut Builder,
    src: &ztensor::Source,
    r: &ResBlock,
    prefix: &str,
) -> Result<(), Error> {
    group_norm(b, &r.norm_in, &format!("{prefix}.in_layers.0"))?;
    conv(b, src, &r.conv_in, &format!("{prefix}.in_layers.2"))?;
    linear(b, &r.emb, &format!("{prefix}.emb_layers.1"))?;
    group_norm(b, &r.norm_out, &format!("{prefix}.out_layers.0"))?;
    conv(b, src, &r.conv_out, &format!("{prefix}.out_layers.3"))?;
    match &r.skip {
        Some(c) => conv(b, src, c, &format!("{prefix}.skip_connection")),
        None => Ok(()),
    }
}

fn linear(b: &mut Builder, w: &Linear, name: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{name}.weight"))?;
    b.read(&w.bias, format!("{name}.bias"))
}

/// `[up | gate]` as stored becomes `[gate | up]` as `linear.mlp_swiglu`
/// reads it.
fn swap_halves(src: Expr, inter: i64) -> Expr {
    Expr::concat(
        0,
        vec![src.clone().slice(0, inter, inter), src.slice(0, 0, inter)],
    )
}

/// One plane stacked on top of itself: `[x | x]`.
fn doubled(src: Expr) -> Expr {
    Expr::concat(0, vec![src.clone(), src])
}

fn slab(expr: Expr, shape: Vec<i64>, stored: Encoding) -> Expr {
    expr.transmute(TensorType::new(shape, stored))
}

/// **THE ROTARY CHANNEL PERMUTATION OF ONE HEAD**, `perm[new] = old`.
///
/// The reference's angle vector is `[y·θ0, x·θ1, y·θ2, x·θ3, …]` of length
/// `d/2`, repeated twice and applied with `rotate_half` — so pair `k` is
/// channels `(k, k + d/2)` and its axis alternates. `RopeForm::Split` over
/// two `d/2`-wide blocks pairs `(b + i, b + d/4 + i)` and gives each block
/// one axis. Sending the `d/4` even (`y`) pairs to block 0 and the `d/4`
/// odd (`x`) pairs to block 1, in order, makes the two forms the same
/// rotation.
fn rope_channels(head_dim: u32) -> Vec<i64> {
    let d = i64::from(head_dim);
    let quarter = d / 4;
    let mut perm = Vec::with_capacity(d as usize);
    for j in 0..quarter {
        perm.push(2 * j);
    }
    for j in 0..quarter {
        perm.push(2 * j + d / 2);
    }
    for j in 0..quarter {
        perm.push(2 * j + 1);
    }
    for j in 0..quarter {
        perm.push(2 * j + 1 + d / 2);
    }
    perm
}

/// [`rope_channels`] laid out over `heads` heads: `perm[new] = old` on a
/// `[heads·head_dim]` axis.
fn head_permutation(heads: u32, head_dim: u32) -> Vec<i64> {
    let channels = rope_channels(head_dim);
    let d = i64::from(head_dim);
    (0..i64::from(heads))
        .flat_map(|h| channels.iter().map(move |c| h * d + c))
        .collect()
}

/// The rows of `[q | k | v]` in the checkpoint's own `[kv_heads, groups +
/// 2, head_dim, hidden]` row space. Query head `groups·g + j` is block
/// `(g, j)`; K is block `(g, groups)` and V block `(g, groups + 1)`. Q and
/// K carry the rotary permutation; V does not turn and keeps its order.
fn qkv_rows(kv_heads: u32, groups: u32, head_dim: u32, q_perm: &[i64], k_perm: &[i64]) -> Vec<i64> {
    let d = i64::from(head_dim);
    let stride = i64::from(groups + 2) * d;
    let mut rows = Vec::new();
    // q, in head order `groups·g + j` — which is what the reference's
    // `reshape(bsz, q_len, num_heads, head_dim)` numbers them.
    for q in q_perm {
        let head = q / d;
        let c = q % d;
        let (g, j) = (head / i64::from(groups), head % i64::from(groups));
        rows.push(g * stride + j * d + c);
    }
    // k: block `groups` of each kv head, rotary-permuted.
    for k in k_perm {
        let head = k / d;
        let c = k % d;
        rows.push(head * stride + i64::from(groups) * d + c);
    }
    // v: block `groups + 1`, in order — V does not turn.
    for g in 0..i64::from(kv_heads) {
        for c in 0..d {
            rows.push(g * stride + (i64::from(groups) + 1) * d + c);
        }
    }
    rows
}

/// `[+1 | -1]` down a `[2·h, 1]` column: two constant fills joined, stated
/// in the dtype `seed` is stored in and cast where the row wants another.
fn signs(b: &mut Builder, src: &ztensor::Source, w: &Weight, seed: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, seed)?;
    let Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{seed}` is stored {stored:?}; a constant is stated in a raw dtype"),
        });
    };
    let shape = extents(w);
    let half = shape[0] / 2;
    let mut parts = Vec::new();
    for (tail, value) in [("pos", 1.0f32), ("neg", -1.0f32)] {
        let name = format!("{}.{tail}", w.name);
        b.push(
            TensorContract::new(
                name.clone(),
                Expr::fill(0.0, TensorType::raw(vec![half, 1], dtype)).bias(value),
                vec![half, 1],
                stored.clone(),
            )
            .internal(),
        );
        parts.push(Expr::out(name));
    }
    let want = encoding(w.dtype);
    let joined = Expr::concat(0, parts);
    let expr = if want == stored {
        joined
    } else {
        joined.cast(want.clone())
    };
    b.push(TensorContract::new(w.name.clone(), expr, shape, want));
    Ok(())
}
