//! `ltx_2` — Lightricks' LTX-2.5, the fourth real generative family in the
//! catalog (design §0, milestone M4).
//!
//! An ASYMMETRIC DUAL-STREAM video-and-audio DiT: 48 blocks, each carrying a
//! 4096-wide video side (32 heads × 128) and a 2048-wide audio side (32 × 64)
//! with separate weights for everything, coupled only by two
//! cross-attentions whose queries and keys turn by DIFFERENT rotary tables on
//! one shared absolute-time axis. Six attentions and two feed-forwards a
//! block, every one of them gated (`out · 2σ(W·x)`), every norm modulated
//! from 9 + 9 stream rows, 4 + 4 + 1 + 1 cross-modal rows and 2 + 2
//! text-context rows. In front of it, the two connector transformers that
//! turn a Gemma-4 trunk's 49 stacked hidden states into the two text
//! contexts ([`forward`] states the table a guest programs against, and what
//! is not yet a reading a guest can name).
//!
//! Two rows: the flagship `Lightricks/LTX-2.5-Diffusers` (bf16 throughout,
//! eight pinned distilled sigmas, no CFG) and the miniature
//! `scripts/imagegen/ltx2_golden.py --mini` writes — two blocks, two heads a
//! side at the REAL head widths, which is the parity fixture.
//!
//! # THREE FAULTS THIS TEXT FOUND, AND WHERE THEY WERE FIXED
//!
//! The first family with a modulation chain PER STREAM is the first to put
//! two lane vectors of one shape in two classes, and the first to read one
//! checkpoint tensor twice in one plane. Both broke silently:
//!
//! * **The arena aliased two live lane rectangles.** `RowExpr::cut_per_class`
//!   said a `Dim::Lanes` value's rows are cut per class, so two of one shape
//!   in two classes were placed at one offset as "two row windows of one
//!   column" — but a lane-shaped chain is launched WITHOUT the staged seat
//!   (`IMAGEGEN_CONTRACT.md` §7) and writes EVERY lane, so the video's
//!   modulation vector and the audio's clobbered each other. Fixed in
//!   `model_compiler::arena` (`Lanes` no longer cuts). Any two-stream text
//!   with a per-stream conditioning vector was exposed to this.
//! * **A cast from a run-collapsed source was refused.** `[x | x]` — this
//!   family's head reads `[embedded_timestep | embedded_timestep]` — lowers
//!   to ONE run repeated, whose extent carries its elements in
//!   `element_bytes` rather than in `dims`, and the cast's element-count
//!   check compared runs against elements. Fixed in
//!   `checkpoint::executor::walk` (the check is now on the bytes the cast
//!   PRODUCED). `wan_2`'s `head_proj` states the same doubling and would
//!   have hit it the moment its miniature was imported.
//! * **`elementwise.add_bias` folds IN PLACE.** One adaLN head serves all 48
//!   blocks, so a block that added its `scale_shift_table` straight onto that
//!   shared vector handed the next block the sum of every table before it.
//!   This text copies first (`forward::table_add`), and
//!   `the_ltx_2_rows_bake`'s claim (j) is the guard. **`wan_2::forward` does
//!   the same thing to its `time_proj` vector and is still to be fixed.**

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today:
/// this family's pixel side is a VAE, not a vision tower.
pub const ARCH: &str = "ltx_2";

/// The flagship first, the miniature last (identification is catalog order,
/// and a miniature reads a checkpoint no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "ltx25",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ltx_2_5(Dtype::Bf16, tp),
        ),
        (
            "ltx25-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp),
        ),
    ];
    // The generative facts a guest sizes a job from (design D12), stated
    // beside the row rather than by the macro. Built from the same
    // constructor the row traces with, so the two cannot disagree.
    for row in &mut rows {
        let model = match row.recipe.text {
            "ltx25" => Model::ltx_2_5(Dtype::Bf16, 1),
            "ltx25-mini" => Model::mini(Dtype::Bf16, 1),
            other => unreachable!("no ltx_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
