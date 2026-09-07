//! Wan 2.2's reading of a checkpoint: the only place the checkpoint's own
//! tensor spellings appear.
//!
//! The shipped checkpoint is a diffusers pipeline folder
//! (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`), which `checkpoint::file::diffusers`
//! opens as ONE name space with a component prefix per folder:
//! `transformer/` → `dit.` (825 fp32 tensors over five shards),
//! `text_encoder/` → `te.` (242 bf16 tensors over three shards), `vae/` →
//! `vae.` (196 fp32 tensors). Below the prefix the names are the
//! components' own `state_dict` names, read off the snapshot's safetensors
//! headers. The miniatures `wan22_golden.py --mini` write the
//! transformer's `state_dict` alone, unprefixed, in one fp32 file each:
//! [`Layout::Bare`].
//!
//! What is not a plain read:
//!
//! * `attn1.to_{q,k,v}` fuse into one packed `qkv` (weights and biases),
//!   `attn2.to_{k,v}` into one packed `kv`;
//! * `patch_embedding.weight` `[dim, C, 1, 2, 2]` is read as the
//!   `[dim, C·4]` linear it is (a transmute: the `(c, ph, pw)` row order
//!   is the conv's own input order);
//! * `proj_out`'s rows are permuted from the checkpoint's `(ph, pw, c)`
//!   feature order to `(c, ph, pw)`, so the velocity row is laid out like
//!   the latent row in (`forward.rs`);
//! * the modulation planes are stored `[shift | scale | gate]` per set
//!   (`time_proj`, `scale_shift_table` `[1, 6, dim]`, the head's
//!   `[1, 2, dim]` as `[shift | scale]`) and read in the plan's
//!   `[scale | shift | gate]`, because `elementwise.modulate` takes its
//!   pair as `[s | b]` and this IR cannot permute columns at run time; the
//!   swap is a slice-and-concatenate of the stored plane, done once here.
//!   The two tables are declared f32 and read without a cast;
//! * `time_embedder.linear_2` is stacked twice into `head_proj`
//!   (`model::Dit::head_proj`);
//! * every VAE conv kernel `[C_out, C_in, kt, kh, kw]` is read as the
//!   `[C_out, C_in·taps]` rectangle `Weight::conv_taps_major` declares (a
//!   transmute; the shell relabels at load); the `time_conv` rows are
//!   permuted from `(r1, c)` to `(c, r1)` and `conv_out`'s from
//!   `(c, pw, ph)` to `(c, ph, pw)` for the two depth-to-space ops that
//!   read them; conv biases are cast to f32; the RMS gains `[C, 1, 1, 1]`
//!   are read as `[C]`;
//! * the transformer and VAE ship fp32 and every bank is declared bf16, so
//!   those reads are casts;
//! * the encoder's `conv_in` has its INPUT channels permuted from the
//!   checkpoint's `(c, pw, ph)` patchify order into the `(c, ph, pw)` a
//!   `spatial.pixel_unshuffle` lands — the mirror of `conv_out`'s row
//!   gather on the decode side, on axis 1 because a conv's input channels
//!   are its second axis; and `quant_conv` is read as its FIRST 48 output
//!   rows alone, the posterior mean's, the logvar's never being computed.
//!
//! Every plane of the checkpoint is read: the decoder's, the encoder's,
//! `post_quant_conv` and `quant_conv`.

use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{
    Block, Conv, Dims, Dit, Downsampler, HEAD_SLICES, Linear, MOD_SLICES, Model, Resnet,
    TextEncoder, VAE_LATENTS_MEAN, VAE_LATENTS_STD, VAE_PATCH, VAE_RGB, VAE_Z, Vae, VaeEncoder,
};

/// Where a checkpoint puts the components.
#[derive(Clone, Copy)]
enum Layout {
    /// A diffusers pipeline folder read through `checkpoint::file::diffusers`:
    /// `dit.<transformer name>`, `te.<UMT5EncoderModel name>`, `vae.<name>`.
    Diffusers,
    /// The transformer's `state_dict` alone, at the root — a miniature's
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
            name: "wan_2".to_string(),
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
        dit(&mut b, src, &self.dit, &self.dims, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, te, layout)?;
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
    d: &Dims,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);
    let dim = d.dim;

    // `Conv3d(C, dim, (1, 2, 2), stride (1, 2, 2))` as the linear it is.
    transmuted(b, src, &m.patch_embed.w, &at("patch_embedding.weight"))?;
    b.read(&m.patch_embed.bias, at("patch_embedding.bias"))?;

    let cond = |s: &str| at(&format!("condition_embedder.{s}"));
    biased(b, &m.text_embed.linear_1, &cond("text_embedder.linear_1"))?;
    biased(b, &m.text_embed.linear_2, &cond("text_embedder.linear_2"))?;
    biased(b, &m.time_embed.linear_1, &cond("time_embedder.linear_1"))?;
    biased(b, &m.time_embed.linear_2, &cond("time_embedder.linear_2"))?;
    // `linear_2` twice over, for the head's `[temb | temb]`.
    let l2 = cond("time_embedder.linear_2");
    let doubled = |e: Expr| Expr::concat(0, vec![e.clone(), e]);
    b.read_over(&m.head_proj.w, format!("{l2}.weight"), doubled)?;
    b.read_over(&m.head_proj.bias, format!("{l2}.bias"), doubled)?;
    // `time_proj`: six `dim`-row sets, `(shift, scale)` swapped in each.
    let proj = cond("time_proj");
    let swap = |axis: u8, width: i64| move |e: Expr| slices_reordered(&e, MOD_SLICES, width, axis);
    b.read_over(
        &m.time_proj.w,
        format!("{proj}.weight"),
        swap(0, i64::from(dim)),
    )?;
    b.read_over(
        &m.time_proj.bias,
        format!("{proj}.bias"),
        swap(0, i64::from(dim)),
    )?;

    for (i, block) in m.blocks.iter().enumerate() {
        transformer_block(b, src, block, &at(&format!("blocks.{i}")))?;
    }

    // The head's `[1, 2, dim]` table as a `[2·dim]` f32 bias, `[scale |
    // shift]`, and `proj_out` with its rows brought to `(c, ph, pw)`.
    table(b, src, &m.head_table, &at("scale_shift_table"), HEAD_SLICES)?;
    let rows = head_rows(d.out_channels);
    b.read_over(&m.proj_out.w, at("proj_out.weight"), |e| {
        e.gather(0, rows.clone())
    })?;
    b.read_over(&m.proj_out.bias, at("proj_out.bias"), |e| e.gather(0, rows))
}

/// One `WanTransformerBlock`'s planes under `stem`.
fn transformer_block(
    b: &mut Builder,
    src: &ztensor::Source,
    block: &Block,
    stem: &str,
) -> Result<(), Error> {
    let n = |s: &str| format!("{stem}.{s}");
    table(b, src, &block.table, &n("scale_shift_table"), MOD_SLICES)?;

    let a = &block.self_attn;
    packed(
        b,
        &a.qkv,
        &[n("attn1.to_q"), n("attn1.to_k"), n("attn1.to_v")],
    )?;
    b.read(&a.norm_q, n("attn1.norm_q.weight"))?;
    b.read(&a.norm_k, n("attn1.norm_k.weight"))?;
    biased(b, &a.out, &n("attn1.to_out.0"))?;

    b.read(&block.norm2, n("norm2.weight"))?;
    b.read(&block.norm2_bias, n("norm2.bias"))?;

    let c = &block.cross;
    biased(b, &c.q, &n("attn2.to_q"))?;
    packed(b, &c.kv, &[n("attn2.to_k"), n("attn2.to_v")])?;
    b.read(&c.norm_q, n("attn2.norm_q.weight"))?;
    b.read(&c.norm_k, n("attn2.norm_k.weight"))?;
    biased(b, &c.out, &n("attn2.to_out.0"))?;

    biased(b, &block.ffn.up, &n("ffn.net.0.proj"))?;
    biased(b, &block.ffn.down, &n("ffn.net.2"))
}

/// The encoder: `UMT5EncoderModel`'s names under `te.` — `shared`, the 24
/// blocks (`layer.0` attention with its bucket embedding, `layer.1` the
/// gated MLP), `final_layer_norm`.
fn text_encoder(b: &mut Builder, te: &TextEncoder, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("te.", tail);
    b.read(&te.embed, at("shared.weight")?)?;
    for (l, w) in te.layers.iter().enumerate() {
        let attn = |s: &str| at(&format!("encoder.block.{l}.layer.0.{s}"));
        let ffn = |s: &str| at(&format!("encoder.block.{l}.layer.1.{s}"));
        b.read(&w.attn_norm, attn("layer_norm.weight")?)?;
        b.read(&w.q, attn("SelfAttention.q.weight")?)?;
        b.read(&w.k, attn("SelfAttention.k.weight")?)?;
        b.read(&w.v, attn("SelfAttention.v.weight")?)?;
        b.read(&w.o, attn("SelfAttention.o.weight")?)?;
        b.read(
            &w.rel_bias,
            attn("SelfAttention.relative_attention_bias.weight")?,
        )?;
        b.read(&w.ffn_norm, ffn("layer_norm.weight")?)?;
        b.read(&w.wi_0, ffn("DenseReluDense.wi_0.weight")?)?;
        b.read(&w.wi_1, ffn("DenseReluDense.wi_1.weight")?)?;
        b.read(&w.wo, ffn("DenseReluDense.wo.weight")?)?;
    }
    b.read(&te.final_norm, at("encoder.final_layer_norm.weight")?)
}

/// The VAE decoder side, `AutoencoderKLWan`'s own names under `vae.`.
fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    // The two config rows the decoder arm denormalises with. They are in
    // `vae/config.json` and in no tensor, so the contract STATES them:
    // `Fill` is realized by zeroing and `Bias` adds the number, one
    // element at a time, concatenated into the row (the same algebra
    // `z_image`'s pad tables use).
    row_of(b, src, &v.denorm_scale, &at("post_quant_conv.bias")?, |i| {
        VAE_LATENTS_STD[i]
    })?;
    row_of(b, src, &v.denorm_bias, &at("post_quant_conv.bias")?, |i| {
        -VAE_LATENTS_MEAN[i] / VAE_LATENTS_STD[i]
    })?;

    conv(b, src, &v.post_quant, &at("post_quant_conv")?, None)?;
    conv(b, src, &v.conv_in, &at("decoder.conv_in")?, None)?;

    resnet(b, src, &v.mid_res0, &at("decoder.mid_block.resnets.0")?)?;
    let attn = at("decoder.mid_block.attentions.0")?;
    gamma(b, src, &v.mid_attn.norm, &format!("{attn}.norm.gamma"))?;
    transmuted(b, src, &v.mid_attn.qkv.w, &format!("{attn}.to_qkv.weight"))?;
    b.read(&v.mid_attn.qkv.bias, format!("{attn}.to_qkv.bias"))?;
    transmuted(b, src, &v.mid_attn.proj.w, &format!("{attn}.proj.weight"))?;
    b.read(&v.mid_attn.proj.bias, format!("{attn}.proj.bias"))?;
    resnet(b, src, &v.mid_res1, &at("decoder.mid_block.resnets.1")?)?;

    for (i, up) in v.up.iter().enumerate() {
        for (r, block) in up.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                block,
                &at(&format!("decoder.up_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(u) = &up.upsampler {
            let stem = at(&format!("decoder.up_blocks.{i}.upsampler"))?;
            if let Some(tc) = &u.time_conv {
                conv(
                    b,
                    src,
                    tc,
                    &format!("{stem}.time_conv"),
                    Some(time_conv_rows(tc.c_in)),
                )?;
            }
            conv(b, src, &u.resample, &format!("{stem}.resample.1"), None)?;
        }
    }

    gamma(b, src, &v.norm_out, &at("decoder.norm_out.gamma")?)?;
    conv(
        b,
        src,
        &v.conv_out,
        &at("decoder.conv_out")?,
        Some(conv_out_rows()),
    )?;
    encoder(b, src, &v.enc, layout)
}

/// The VAE encoder side, `WanEncoder3d`'s own names under `vae.encoder.`,
/// plus `vae.quant_conv`.
fn encoder(
    b: &mut Builder,
    src: &ztensor::Source,
    e: &VaeEncoder,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    // The mirror of the decoder's stated rows: `mean` and `1/std`, so the
    // arm lands `(z − mean)/std` — the denoise reading's own space.
    row_of(b, src, &e.norm_bias, &at("quant_conv.bias")?, |i| {
        VAE_LATENTS_MEAN[i]
    })?;
    row_of(b, src, &e.norm_scale, &at("quant_conv.bias")?, |i| {
        1.0 / VAE_LATENTS_STD[i]
    })?;

    // `patchify`'s `(c, pw, ph)` feature order into the `(c, ph, pw)` a
    // `pixel_unshuffle` lands, on the kernel's INPUT-channel axis.
    conv_over_channels(b, src, &e.conv_in, &at("encoder.conv_in")?, conv_out_rows())?;

    for (i, block) in e.down.iter().enumerate() {
        for (r, res) in block.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                res,
                &at(&format!("encoder.down_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(Downsampler {
            resample,
            time_conv,
        }) = &block.downsampler
        {
            let stem = at(&format!("encoder.down_blocks.{i}.downsampler"))?;
            conv(b, src, resample, &format!("{stem}.resample.1"), None)?;
            if let Some(tc) = time_conv {
                conv(b, src, tc, &format!("{stem}.time_conv"), None)?;
            }
        }
    }

    resnet(b, src, &e.mid_res0, &at("encoder.mid_block.resnets.0")?)?;
    let attn = at("encoder.mid_block.attentions.0")?;
    gamma(b, src, &e.mid_attn.norm, &format!("{attn}.norm.gamma"))?;
    transmuted(b, src, &e.mid_attn.qkv.w, &format!("{attn}.to_qkv.weight"))?;
    b.read(&e.mid_attn.qkv.bias, format!("{attn}.to_qkv.bias"))?;
    transmuted(b, src, &e.mid_attn.proj.w, &format!("{attn}.proj.weight"))?;
    b.read(&e.mid_attn.proj.bias, format!("{attn}.proj.bias"))?;
    resnet(b, src, &e.mid_res1, &at("encoder.mid_block.resnets.1")?)?;

    gamma(b, src, &e.norm_out, &at("encoder.norm_out.gamma")?)?;
    conv(b, src, &e.conv_out, &at("encoder.conv_out")?, None)?;
    // `quant_conv`'s first `z_dim` output rows: the posterior MEAN's. A
    // LEADING run, so it is a slice and not a gather (the contract checker
    // refuses the gather that spells the same thing).
    let quant = at("quant_conv")?;
    let name = format!("{quant}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&e.quant.w), stored);
    let rows = i64::from(VAE_Z);
    b.read_over(&e.quant.w, name, move |x| {
        x.slice(0, 0, rows).transmute(shape)
    })?;
    b.read_over(&e.quant.bias, format!("{quant}.bias"), move |x| {
        x.slice(0, 0, rows)
    })
}

/// A conv whose INPUT channels are permuted: the stored `[C_out, C_in, kt,
/// kh, kw]` kernel gathered along AXIS 1 (a conv's input-channel axis) and
/// then transmuted to the declared `[C_out, C_in·taps]`. The bias is read
/// straight through — it is per output channel and nothing moved there.
fn conv_over_channels(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &Conv,
    stem: &str,
    cols: Vec<i64>,
) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&c.w), stored);
    b.read_over(&c.w, name, move |e| e.gather(1, cols).transmute(shape))?;
    b.read(&c.bias, format!("{stem}.bias"))
}

/// A `nn.Linear`: `<stem>.weight` and `<stem>.bias`.
fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

/// Several `nn.Linear`s fused into one packed bank, weights and biases.
fn packed(b: &mut Builder, w: &Linear, stems: &[String]) -> Result<(), Error> {
    b.read_concat(&w.w, stems.iter().map(|s| format!("{s}.weight")))?;
    b.read_concat(&w.bias, stems.iter().map(|s| format!("{s}.bias")))
}

/// A stored tensor read as the declared shape over the same bytes (a conv
/// kernel as `[C_out, C_in·taps]`, a `[C, 1, 1, 1]` gain as `[C]`).
fn transmuted(b: &mut Builder, src: &ztensor::Source, w: &Weight, name: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    let shape = TensorType::new(extents(w), stored);
    b.read_over(w, name.to_string(), |e| e.transmute(shape))
}

/// A `scale_shift_table` `[1, k, dim]` as the `[k·dim]` bias the plan
/// reads, its `(shift, scale)` pairs swapped.
fn table(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    slices: u32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    let shape = TensorType::new(extents(w), stored);
    b.read_over(w, name.to_string(), |e| {
        slices_reordered(&e, slices, 1, 1).transmute(shape)
    })
}

/// The permutation itself: `slices` consecutive `width`-wide blocks of
/// `axis`, concatenated in the plan's order — every `(shift, scale)` pair
/// exchanged, every gate left in place.
fn slices_reordered(from: &Expr, slices: u32, width: i64, axis: u8) -> Expr {
    let take = |i: i64| from.clone().slice(axis, i * width, width);
    let order: Vec<i64> = match slices {
        2 => vec![1, 0],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("Wan states modulation vectors of 2 or 6 slices, not {other}"),
    };
    Expr::concat(axis, order.into_iter().map(take).collect())
}

/// `proj_out`'s rows in the plan's `(c, ph, pw)` order: row `(c·2 + ph)·2
/// + pw` of the plan is row `(ph·2 + pw)·C + c` of the checkpoint (the
/// reference's `reshape(.., p_t, p_h, p_w, -1)` puts the channel last).
fn head_rows(c_out: u32) -> Vec<i64> {
    let c_out = i64::from(c_out);
    let mut rows = Vec::with_capacity((4 * c_out) as usize);
    for c in 0..c_out {
        for ph in 0..2 {
            for pw in 0..2 {
                rows.push((ph * 2 + pw) * c_out + c);
            }
        }
    }
    rows
}

/// A `time_conv`'s `2C` output rows in `(c, r1)` order: the checkpoint's
/// `reshape(b, 2, c, ..)` puts the frame parity first.
fn time_conv_rows(c: u32) -> Vec<i64> {
    let c = i64::from(c);
    (0..c).flat_map(|ch| [ch, c + ch]).collect()
}

/// The 12 patchify channels in `(c, ph, pw)` order: the checkpoint's
/// `patchify`/`unpatchify` pair reads channel `c·4 + pw·2 + ph`. The
/// decoder gathers `conv_out`'s ROWS by it; the encoder gathers
/// `conv_in`'s input CHANNELS by it.
fn conv_out_rows() -> Vec<i64> {
    let p = i64::from(VAE_PATCH);
    let mut rows = Vec::new();
    for c in 0..i64::from(VAE_RGB) {
        for ph in 0..p {
            for pw in 0..p {
                rows.push(c * p * p + pw * p + ph);
            }
        }
    }
    rows
}

/// A conv: the kernel `[C_out, C_in, kt, kh, kw]` transmuted to the
/// declared `[C_out, C_in·taps]` (rows gathered first where `rows` says
/// so), and its bias (gathered the same way).
fn conv(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &Conv,
    stem: &str,
    rows: Option<Vec<i64>>,
) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&c.w), stored);
    let gathered = rows.clone();
    b.read_over(&c.w, name, move |e| {
        let kernel = match gathered {
            Some(rows) => e.gather(0, rows),
            None => e,
        };
        kernel.transmute(shape)
    })?;
    let bias = format!("{stem}.bias");
    match rows {
        Some(rows) => b.read_over(&c.bias, bias, move |e| e.gather(0, rows)),
        None => b.read(&c.bias, bias),
    }
}

/// A `[n]` row of STATED numbers — a config vector the checkpoint holds no
/// tensor for (`latents_mean`, `latents_std`).
///
/// The affine fragment of the contract algebra composes byte spans, and a
/// `Bias` is a kernel, so a filled-and-biased element cannot sit inside the
/// `Concat` directly: each element is its own internal one-element tensor
/// and the row is their concatenation, exactly as `z_image`'s pad tables
/// are built. `seed` names a stored plane whose raw dtype the constants are
/// stated in, so every checkpoint of this family derives them the same way;
/// the row is cast where it declares another dtype.
fn row_of(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    seed: &str,
    value: impl Fn(usize) -> f32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, seed)?;
    let checkpoint::types::Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{seed}` is stored {stored:?}; a stated row wants a raw dtype"),
        });
    };
    let shape = extents(w);
    let n = shape.iter().product::<i64>();
    let mut parts = Vec::with_capacity(n as usize);
    for i in 0..n {
        let cell = format!("{}.{i}", w.name);
        b.push(
            checkpoint::contract::TensorContract::new(
                cell.clone(),
                Expr::fill(0.0, TensorType::raw(vec![1], dtype)).bias(value(i as usize)),
                vec![1],
                stored.clone(),
            )
            .internal(),
        );
        parts.push(Expr::out(cell));
    }
    // Pushed, not `read_expr`d: a row built from `Out` alone names no
    // checkpoint tensor, and `read_expr` reads the stored encoding off one.
    let want = checkpoint_dsl::encoding(w.dtype);
    let row = Expr::concat(0, parts);
    let row = if want == stored {
        row
    } else {
        row.cast(want.clone())
    };
    b.push(checkpoint::contract::TensorContract::new(
        w.name.clone(),
        row,
        shape,
        want,
    ));
    Ok(())
}

/// A `WanRMS_norm` gain `[C, 1, 1, 1]` (or `[C, 1, 1]`) as `[C]`.
fn gamma(b: &mut Builder, src: &ztensor::Source, w: &Weight, name: &str) -> Result<(), Error> {
    transmuted(b, src, w, name)
}

fn resnet(b: &mut Builder, src: &ztensor::Source, r: &Resnet, stem: &str) -> Result<(), Error> {
    gamma(b, src, &r.norm1, &format!("{stem}.norm1.gamma"))?;
    conv(b, src, &r.conv1, &format!("{stem}.conv1"), None)?;
    gamma(b, src, &r.norm2, &format!("{stem}.norm2.gamma"))?;
    conv(b, src, &r.conv2, &format!("{stem}.conv2"), None)?;
    if let Some(s) = &r.shortcut {
        conv(b, src, s, &format!("{stem}.conv_shortcut"), None)?;
    }
    Ok(())
}
