//! FLUX.2's reading of a checkpoint: the only place the checkpoint's own
//! tensor spellings appear.
//!
//! The shipped checkpoint is a diffusers pipeline folder
//! (`black-forest-labs/FLUX.2-klein-4B`), which `checkpoint::file::diffusers`
//! opens as ONE name space with a component prefix per folder:
//! `transformer/` → `dit.`, `text_encoder/` → `te.`, `vae/` → `vae.`. Below
//! the prefix the names are the components' own `state_dict` names, read
//! off the snapshot's safetensors headers (169 `dit.` tensors, 398 `te.`,
//! 251 `vae.`). The miniature `flux2_golden.py --mini` writes the
//! transformer's `state_dict` alone, unprefixed, in one fp32 file:
//! [`Layout::Bare`].
//!
//! What is not a plain read:
//!
//! * `to_q`/`to_k`/`to_v` (and `add_{q,k,v}_proj`) fuse into one packed
//!   `qkv` bank (the `read_concat` every family uses);
//! * the three shared modulation linears are stored in the diffusers order
//!   `[shift | scale | gate]` per set and read in the plan's `[scale | shift
//!   | gate]`, because `elementwise.modulate` takes its pair as `[s | b]`
//!   and this IR cannot permute columns at run time; the swap is a
//!   slice-and-concatenate of the stored plane, done once here.
//!   `norm_out.linear` is `[scale | shift]` already and is read whole;
//! * the single block's `to_out` `[dim, dim + inter]` is cut into its two
//!   column blocks (`model::SingleBlock`), and on the flagship
//!   `context_embedder` `[dim, 3·2560]` into its three
//!   (`model::TextEncoder::context_embed`);
//! * every VAE conv kernel `[C_out, C_in, k, k]` is read as the `[C_out,
//!   C_in·k²]` rectangle `Weight::conv_taps_major` declares (a transmute:
//!   same bytes, the shell relabels at load); conv biases and group-norm
//!   affines are cast to f32;
//! * the frozen BatchNorm becomes three `[128]` planes: the deviation
//!   `√(var + eps)` (`UnaryOp::Sqrt` over `bn.running_var + eps`) as
//!   `standardize`'s scale, a zero fill as its bias, and the stored mean
//!   as the `add_bias` after it;
//! * the miniature ships fp32 and every bank is declared bf16, so each of
//!   its reads is a cast;
//! * the encoder's last nine layers and final norm are not read
//!   (`model::TE_LAYERS`), nor is `lm_head` (tied, and not shipped).
//!
//! Not read at all: `vae.encoder.*` and `vae.quant_conv.*` (no encode arm
//! yet — `forward.rs`), and `vae.bn.num_batches_tracked`.

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType, UnaryOp};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, encoding, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{
    Attn, Conv, DOUBLE_MOD_SLICES, Dit, GroupNorm, MidAttention, Model, Resnet, SINGLE_MOD_SLICES,
    Swiglu, TE_HIDDEN, TextEncoder, VAE_BN_EPS, Vae,
};

/// Where a checkpoint puts the components.
#[derive(Clone, Copy)]
enum Layout {
    /// A diffusers pipeline folder read through `checkpoint::file::diffusers`:
    /// `dit.<transformer name>`, `te.model.<Qwen3Model name>`, `vae.<name>`.
    Diffusers,
    /// The transformer's `state_dict` alone, at the root — the miniature's
    /// one file. No encoder or VAE can be read under it.
    Bare,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Diffusers => "a diffusers pipeline (`dit.`/`te.`/`vae.` prefixes)",
            Self::Bare => "a bare transformer state_dict",
        }
    }

    fn dit(self, tail: &str) -> String {
        match self {
            Self::Diffusers => format!("dit.{tail}"),
            Self::Bare => tail.to_string(),
        }
    }

    fn component(self, prefix: &str, tail: &str) -> Result<String, Error> {
        match self {
            Self::Diffusers => Ok(format!("{prefix}{tail}")),
            Self::Bare => Err(Error::Illegible {
                name: format!("{prefix}{tail}"),
                detail: format!(
                    "{} carries no `{prefix}` component, and this row declares one",
                    self.spelling()
                ),
            }),
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut refusals: Vec<String> = Vec::new();
        let layouts: &[Layout] = match (&self.te, &self.vae) {
            (None, None) => &[Layout::Bare, Layout::Diffusers],
            _ => &[Layout::Diffusers],
        };
        for layout in layouts {
            match self.import_from(src, platform, *layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {}, {why}", layout.spelling())),
            }
        }
        Err(Error::Illegible {
            name: "flux_2".to_string(),
            detail: format!(
                "no reading of this checkpoint lands every plane this family declares — {}",
                refusals.join("; ")
            ),
        })
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        dit(&mut b, src, &self.dit, self.dims.dim, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, src, te, layout)?;
        }
        if let Some(vae) = &self.vae {
            self::vae(&mut b, src, vae, layout)?;
        }
        Ok(b.build())
    }
}

fn dit(
    b: &mut Builder,
    src: &ztensor::Source,
    m: &Dit,
    dim: u32,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);
    let w = |tail: &str| at(&format!("{tail}.weight"));

    b.read(&m.x_embed, w("x_embedder"))?;
    if let Some(ctx) = &m.context_embed {
        b.read(ctx, w("context_embedder"))?;
    }
    b.read(
        &m.t_embed.linear_1,
        w("time_guidance_embed.timestep_embedder.linear_1"),
    )?;
    b.read(
        &m.t_embed.linear_2,
        w("time_guidance_embed.timestep_embedder.linear_2"),
    )?;
    if let Some(g) = &m.g_embed {
        b.read(
            &g.linear_1,
            w("time_guidance_embed.guidance_embedder.linear_1"),
        )?;
        b.read(
            &g.linear_2,
            w("time_guidance_embed.guidance_embedder.linear_2"),
        )?;
    }

    // The three shared modulation linears, `(shift, scale)` swapped per set.
    reordered(
        b,
        &m.mod_img,
        &w("double_stream_modulation_img.linear"),
        DOUBLE_MOD_SLICES,
        dim,
    )?;
    reordered(
        b,
        &m.mod_txt,
        &w("double_stream_modulation_txt.linear"),
        DOUBLE_MOD_SLICES,
        dim,
    )?;
    reordered(
        b,
        &m.mod_single,
        &w("single_stream_modulation.linear"),
        SINGLE_MOD_SLICES,
        dim,
    )?;

    for (i, block) in m.double.iter().enumerate() {
        let stem = at(&format!("transformer_blocks.{i}"));
        attn(
            b,
            &block.img.attn,
            &stem,
            ["to_q", "to_k", "to_v"],
            ["norm_q", "norm_k"],
            "to_out.0",
        )?;
        swiglu(b, &block.img.ff, &format!("{stem}.ff"))?;
        attn(
            b,
            &block.txt.attn,
            &stem,
            ["add_q_proj", "add_k_proj", "add_v_proj"],
            ["norm_added_q", "norm_added_k"],
            "to_add_out",
        )?;
        swiglu(b, &block.txt.ff, &format!("{stem}.ff_context"))?;
    }

    for (i, block) in m.single.iter().enumerate() {
        let stem = at(&format!("single_transformer_blocks.{i}.attn"));
        b.read(&block.in_proj, format!("{stem}.to_qkv_mlp_proj.weight"))?;
        b.read(&block.q_norm, format!("{stem}.norm_q.weight"))?;
        b.read(&block.k_norm, format!("{stem}.norm_k.weight"))?;
        // `to_out` `[dim, dim + inter]`: the attention columns, then the
        // MLP's.
        let out = format!("{stem}.to_out.weight");
        let inter = block.out_mlp.shape[1] as i64;
        column_block(b, src, &block.out_attn, &out, 0, i64::from(dim))?;
        column_block(b, src, &block.out_mlp, &out, i64::from(dim), inter)?;
    }

    b.read(&m.norm_out, w("norm_out.linear"))?;
    b.read(&m.proj_out, w("proj_out"))
}

/// One attention's planes under `stem.attn`: the three projections fused
/// into the packed `qkv`, the two QK gains, the output.
fn attn(
    b: &mut Builder,
    a: &Attn,
    stem: &str,
    qkv: [&str; 3],
    norms: [&str; 2],
    out: &str,
) -> Result<(), Error> {
    let n = |s: &str| format!("{stem}.attn.{s}.weight");
    b.read_concat(&a.qkv, qkv.map(n))?;
    b.read(&a.q_norm, n(norms[0]))?;
    b.read(&a.k_norm, n(norms[1]))?;
    b.read(&a.out, n(out))
}

/// `Flux2FeedForward`: `linear_in` is stored `[gate | up]` already.
fn swiglu(b: &mut Builder, ff: &Swiglu, stem: &str) -> Result<(), Error> {
    b.read(&ff.linear_in, format!("{stem}.linear_in.weight"))?;
    b.read(&ff.linear_out, format!("{stem}.linear_out.weight"))
}

/// A modulation projection with every `(shift, scale)` pair swapped into
/// the plan's `(scale, shift)` order and the gate left where it is:
/// `slices` consecutive `dim`-row blocks of axis 0, concatenated in the
/// plan's order.
fn reordered(b: &mut Builder, w: &Weight, from: &str, slices: u32, dim: u32) -> Result<(), Error> {
    let take = |i: i64| Expr::src(from.to_string()).slice(0, i * i64::from(dim), i64::from(dim));
    let order: Vec<i64> = match slices {
        3 => vec![1, 0, 2],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("FLUX.2 states modulation vectors of 3 or 6 slices, not {other}"),
    };
    b.read_expr(w, Expr::concat(0, order.into_iter().map(take).collect()))
}

/// The encoder: `Qwen3ForCausalLM`'s `model.*` names under `te.`, the
/// first `TE_LAYERS` layers only, no final norm, no head; plus
/// `context_embedder` cut into its three tap blocks.
fn text_encoder(
    b: &mut Builder,
    src: &ztensor::Source,
    te: &TextEncoder,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.component("te.model.", tail);
    b.read(&te.embed, at("embed_tokens.weight")?)?;
    for (l, w) in te.layers.iter().enumerate() {
        let n = |s: &str| at(&format!("layers.{l}.{s}"));
        b.read(&w.attn_norm, n("input_layernorm.weight")?)?;
        b.read(&w.q, n("self_attn.q_proj.weight")?)?;
        b.read(&w.k, n("self_attn.k_proj.weight")?)?;
        b.read(&w.v, n("self_attn.v_proj.weight")?)?;
        b.read(&w.o, n("self_attn.o_proj.weight")?)?;
        b.read(&w.q_norm, n("self_attn.q_norm.weight")?)?;
        b.read(&w.k_norm, n("self_attn.k_norm.weight")?)?;
        b.read(&w.mlp_norm, n("post_attention_layernorm.weight")?)?;
        b.read_concat(
            &w.gate_up,
            [n("mlp.gate_proj.weight")?, n("mlp.up_proj.weight")?],
        )?;
        b.read(&w.down, n("mlp.down_proj.weight")?)?;
    }
    let embedder = layout.dit("context_embedder.weight");
    for (i, w) in te.context_embed.iter().enumerate() {
        let start = i64::from(TE_HIDDEN) * i as i64;
        column_block(b, src, w, &embedder, start, i64::from(TE_HIDDEN))?;
    }
    Ok(())
}

/// Columns `[start, start + len)` of a stored `[rows, cols]` plane as `w`.
/// A column slice is a strided walk of the checkpoint, and the executor
/// casts only a compact block, so where the row's dtype is not the
/// checkpoint's the slice lands first as an internal plane in the stored
/// dtype and the cast is a second step over it (the `read_over` shape).
fn column_block(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    from: &str,
    start: i64,
    len: i64,
) -> Result<(), Error> {
    let stored = stored_encoding(src, from)?;
    let want = encoding(w.dtype);
    let sliced = Expr::src(from.to_string()).slice(1, start, len);
    if stored == want {
        return b.read_expr(w, sliced);
    }
    let staged = format!("{}.read", w.name);
    b.push(TensorContract::new(staged.clone(), sliced, extents(w), stored).internal());
    b.push(TensorContract::new(
        w.name.clone(),
        Expr::out(staged).cast(want.clone()),
        extents(w),
        want,
    ));
    Ok(())
}

/// The VAE decoder side, `AutoencoderKLFlux2`'s own names under `vae.`.
fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    batch_norm(b, src, v, &at("bn.running_mean")?, &at("bn.running_var")?)?;
    conv(b, src, &v.post_quant_conv, &at("post_quant_conv")?)?;
    conv(b, src, &v.conv_in, &at("decoder.conv_in")?)?;

    resnet(b, src, &v.mid.resnet_0, &at("decoder.mid_block.resnets.0")?)?;
    mid_attention(b, &v.mid.attention, &at("decoder.mid_block.attentions.0")?)?;
    resnet(b, src, &v.mid.resnet_1, &at("decoder.mid_block.resnets.1")?)?;

    for (i, up) in v.up_blocks.iter().enumerate() {
        for (r, block) in up.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                block,
                &at(&format!("decoder.up_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(c) = &up.upsample {
            conv(
                b,
                src,
                c,
                &at(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"))?,
            )?;
        }
    }

    group_norm(b, &v.norm_out, &at("decoder.conv_norm_out")?)?;
    conv(b, src, &v.conv_out, &at("decoder.conv_out")?)
}

/// A `nn.Conv2d`: the kernel `[C_out, C_in, k, k]` transmuted to the
/// declared `[C_out, C_in·k²]` (the same bytes), and its bias.
fn conv(b: &mut Builder, src: &ztensor::Source, c: &Conv, stem: &str) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    b.read_expr(
        &c.w,
        Expr::src(name).transmute(TensorType::new(extents(&c.w), stored)),
    )?;
    b.read(&c.bias, format!("{stem}.bias"))
}

fn group_norm(b: &mut Builder, n: &GroupNorm, stem: &str) -> Result<(), Error> {
    b.read(&n.weight, format!("{stem}.weight"))?;
    b.read(&n.bias, format!("{stem}.bias"))
}

fn resnet(b: &mut Builder, src: &ztensor::Source, r: &Resnet, stem: &str) -> Result<(), Error> {
    group_norm(b, &r.norm1, &format!("{stem}.norm1"))?;
    conv(b, src, &r.conv1, &format!("{stem}.conv1"))?;
    group_norm(b, &r.norm2, &format!("{stem}.norm2"))?;
    conv(b, src, &r.conv2, &format!("{stem}.conv2"))?;
    if let Some(s) = &r.shortcut {
        conv(b, src, s, &format!("{stem}.conv_shortcut"))?;
    }
    Ok(())
}

fn mid_attention(b: &mut Builder, a: &MidAttention, stem: &str) -> Result<(), Error> {
    group_norm(b, &a.norm, &format!("{stem}.group_norm"))?;
    for (w, bias, name) in [
        (&a.q, &a.q_bias, "to_q"),
        (&a.k, &a.k_bias, "to_k"),
        (&a.v, &a.v_bias, "to_v"),
        (&a.out, &a.out_bias, "to_out.0"),
    ] {
        b.read(w, format!("{stem}.{name}.weight"))?;
        b.read(bias, format!("{stem}.{name}.bias"))?;
    }
    Ok(())
}

/// The frozen BatchNorm's three planes: `scale = √(var + eps)`, the zero
/// bias `standardize` reads beside it (a fill in the declared dtype), and
/// the mean.
fn batch_norm(
    b: &mut Builder,
    src: &ztensor::Source,
    v: &Vae,
    mean: &str,
    var: &str,
) -> Result<(), Error> {
    // `√(var + eps)` is two kernels, and the load plan lowers a kernel only
    // at the root of a step: the sum is one internal plane, the root another
    // over it, and the published plane is that root, cast where the row's
    // dtype is not the checkpoint's.
    let stored = stored_encoding(src, var)?;
    let shape = extents(&v.bn_scale);
    let var_eps = format!("{}.var_eps", v.bn_scale.name);
    let std = format!("{}.std", v.bn_scale.name);
    b.push(
        TensorContract::new(
            var_eps.clone(),
            Expr::src(var.to_string()).bias(VAE_BN_EPS),
            shape.clone(),
            stored.clone(),
        )
        .internal(),
    );
    b.push(
        TensorContract::new(
            std.clone(),
            Expr::out(var_eps).unary(UnaryOp::Sqrt),
            shape.clone(),
            stored.clone(),
        )
        .internal(),
    );
    let want = encoding(v.bn_scale.dtype);
    let published = if want == stored {
        Expr::out(std)
    } else {
        Expr::out(std).cast(want.clone())
    };
    b.push(TensorContract::new(
        v.bn_scale.name.clone(),
        published,
        shape,
        want,
    ));
    b.read(&v.bn_mean, mean.to_string())?;
    let want = encoding(v.bn_zero.dtype);
    let Encoding::Raw(dtype) = want.clone() else {
        return Err(Error::Illegible {
            name: v.bn_zero.name.clone(),
            detail: format!("a zero plane is stated raw, not {want:?}"),
        });
    };
    let shape = extents(&v.bn_zero);
    b.push(TensorContract::new(
        v.bn_zero.name.clone(),
        Expr::fill(0.0, TensorType::raw(shape.clone(), dtype)),
        shape,
        want,
    ));
    Ok(())
}
