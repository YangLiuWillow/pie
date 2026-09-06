//! `mini-dit`'s reading of a checkpoint: the only place the reference's own
//! tensor spellings appear.
//!
//! The reference (`scripts/imagegen/mini_dit_ref.py`) writes one
//! `mini_dit.safetensors` of `nn.Module` state-dict names beside a
//! `config.json`. Two things are not plain reads:
//!
//! * the SwiGLU halves, stored as separate `gate_proj`/`up_proj` and read as
//!   one packed bank (the `read_concat` every family uses);
//! * the adaLN vectors, stored in the diffusers order
//!   `[shift | scale | gate]` per sublayer and read in the plan's
//!   `[scale | shift | gate]` order, because `elementwise.modulate` takes its
//!   pair as `[s | b]` and this IR has no way to permute a rectangle's
//!   columns at run time. The permutation is free here — it is a slice-and-
//!   concatenate of the stored plane, done once at import.

use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint_dsl::{Builder, Error, extents};
use model_dsl::Platform;

use super::model::{HIDDEN, MOD_SLICES, Model};

/// Where the reference's tensors sit. One layout today; a second checkpoint
/// spelling would be a second variant, never a second module.
#[derive(Clone, Copy)]
enum Layout {
    /// `mini_dit_ref.py`'s own `state_dict()` names, at the root.
    Reference,
}

impl Layout {
    fn at(self, tail: &str) -> String {
        match self {
            Layout::Reference => tail.to_string(),
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from(src, platform, Layout::Reference)
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        let at = |tail: &str| layout.at(tail);

        // The patch embedder and the head.
        biased(&mut b, &self.x_embed, &at("x_embedder"))?;
        biased(&mut b, &self.final_proj, &at("final_proj"))?;
        // Two slices, `[shift | scale]` stored, `[scale | shift]` read.
        reordered(&mut b, &self.final_ada, &at("final_adaLN"), 2)?;

        // Block 0 — the single-stream joint block.
        reordered(&mut b, &self.single.ada, &at("blocks.0.adaLN"), MOD_SLICES)?;
        self_attn(&mut b, &self.single.attn, &at("blocks.0.attn"))?;
        swiglu(&mut b, &self.single.mlp, &at("blocks.0.mlp"))?;

        // Block 1 — the MM-DiT double-stream block, one side at a time.
        for (side, stem) in [
            (&self.double.img, "blocks.1.img"),
            (&self.double.txt, "blocks.1.txt"),
        ] {
            reordered(&mut b, &side.ada, &at(&format!("{stem}_adaLN")), MOD_SLICES)?;
            self_attn(&mut b, &side.attn, &at(&format!("{stem}_attn")))?;
            swiglu(&mut b, &side.mlp, &at(&format!("{stem}_mlp")))?;
        }

        // Block 2 — the Wan-style cross-attention block.
        let cross = &self.cross;
        // Wan's `scale_shift_table`: stored `[6, HIDDEN]`, read as one
        // `[6 * HIDDEN]` bias in the plan's slice order.
        b.read_expr(
            &cross.mod_table,
            // Stored `[6, HIDDEN]`: the slices are single ROWS of axis 0,
            // and the reordered stack is transmuted flat to `[6 * HIDDEN]`.
            slices_reordered(at("blocks.2.mod_table"), MOD_SLICES, 1, 0).transmute(
                TensorType::new(
                    extents(&cross.mod_table),
                    encoding_of(src, &at("blocks.2.mod_table"))?,
                ),
            ),
        )?;
        reordered(&mut b, &cross.ada, &at("blocks.2.adaLN"), MOD_SLICES)?;
        self_attn(&mut b, &cross.self_attn, &at("blocks.2.self_attn"))?;
        b.read(&cross.norm, at("blocks.2.norm_cross.weight"))?;
        b.read(&cross.norm_bias, at("blocks.2.norm_cross.bias"))?;
        biased(&mut b, &cross.cross.q, &at("blocks.2.cross_attn.q"))?;
        biased(&mut b, &cross.cross.kv, &at("blocks.2.cross_attn.kv"))?;
        b.read(&cross.cross.q_norm, at("blocks.2.cross_attn.norm_q"))?;
        b.read(&cross.cross.k_norm, at("blocks.2.cross_attn.norm_k"))?;
        biased(&mut b, &cross.cross.out, &at("blocks.2.cross_attn.out"))?;
        swiglu(&mut b, &cross.mlp, &at("blocks.2.mlp"))?;

        Ok(b.build())
    }
}

/// A `nn.Linear`: `<stem>.weight` and `<stem>.bias`.
fn biased(b: &mut Builder, w: &super::model::Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

/// A self-attention sublayer: the packed qkv, the two QK gains, the output.
fn self_attn(b: &mut Builder, a: &super::model::SelfAttn, stem: &str) -> Result<(), Error> {
    biased(b, &a.qkv, &format!("{stem}.qkv"))?;
    b.read(&a.q_norm, format!("{stem}.norm_q"))?;
    b.read(&a.k_norm, format!("{stem}.norm_k"))?;
    biased(b, &a.out, &format!("{stem}.out"))
}

/// The SwiGLU: `gate_proj` and `up_proj` fused into one bank, then `down`.
fn swiglu(b: &mut Builder, m: &super::model::Swiglu, stem: &str) -> Result<(), Error> {
    b.read_concat(
        &m.gate_up.w,
        [
            format!("{stem}.gate_proj.weight"),
            format!("{stem}.up_proj.weight"),
        ],
    )?;
    b.read_concat(
        &m.gate_up.bias,
        [
            format!("{stem}.gate_proj.bias"),
            format!("{stem}.up_proj.bias"),
        ],
    )?;
    biased(b, &m.down, &format!("{stem}.down_proj"))
}

/// An adaLN projection, both planes, with the per-sublayer `(shift, scale)`
/// pair swapped into the plan's `(scale, shift)` order and the gate left
/// where it is.
fn reordered(
    b: &mut Builder,
    w: &super::model::Linear,
    stem: &str,
    slices: u32,
) -> Result<(), Error> {
    b.read_expr(
        &w.w,
        slices_reordered(format!("{stem}.weight"), slices, i64::from(HIDDEN), 0),
    )?;
    b.read_expr(
        &w.bias,
        slices_reordered(format!("{stem}.bias"), slices, i64::from(HIDDEN), 0),
    )
}

/// The permutation itself: `slices` consecutive `width`-row blocks of `axis`,
/// concatenated in the plan's order. `slices` is 2 (`shift, scale`) or 6
/// (`shift, scale, gate` twice); in both cases the swap is "exchange the
/// first two of every three, or of every two".
fn slices_reordered(from: String, slices: u32, width: i64, axis: u8) -> Expr {
    let take = |i: i64| Expr::src(from.clone()).slice(axis, i * width, width);
    let order: Vec<i64> = match slices {
        2 => vec![1, 0],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("mini-dit states adaLN vectors of 2 or 6 slices, not {other}"),
    };
    Expr::concat(axis, order.into_iter().map(take).collect())
}

/// The stored encoding of `name`, for a transmute that must not change it.
fn encoding_of(src: &ztensor::Source, name: &str) -> Result<checkpoint::types::Encoding, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    checkpoint::file::encoding_of(&tensor).map_err(|why| Error::Illegible {
        name: name.to_string(),
        detail: why.to_string(),
    })
}
