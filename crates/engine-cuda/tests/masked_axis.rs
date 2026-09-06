//! The `masked` axis, end to end — and the gate that says the C1 axis runs.
//!
//! **WHAT THIS FILE IS FOR.** `masked` is design §0's second supergraph axis
//! and the first one beyond decode/prefill: a per-lane fact the model
//! declares, a run-length mask on the submission, an `attention.masked` arm
//! over its own window. C1 wrote the bits path and then found the catalog
//! could not run it, for three reasons that were each somebody's to fix and
//! none of them the mask path's (build log 20). This file used to PIN those
//! three refusals so that the day one was fixed the test asserting it would
//! fail and say so. C1b fixed all three, so every one of them has flipped:
//! what is asserted now is the fix, in the same place and against the same
//! catalog.
//!
//! ```text
//! blocker            what it was                        what it is now
//! kv::probe          one geometry per kv SPACE, and     facts are keyed by
//!                    gemma states two                   ROW and by PLAN
//! the schedule       one plan_prefill read by two       gemma mints six, one
//!                    classes -> Fault::Straddled        per (reading x class)
//! the windowed arm   "fa2 has no custom+sliding"        it always had one:
//!                                                       VariantCustom IS it
//! ```
//!
//! **THE AXIS IS NO LONGER GEMMA'S ALONE, AND THIS FILE IS STILL GEMMA'S.**
//! `8cb1b6ce6` seated `attention.masked` in the qwen text as well, so the
//! catalog now has two declaring families and four documented-maskless ones.
//! The device gates below stay gemma's — it is the family that also states a
//! sliding window on most of its masked arms, so its arm exercises both terms
//! at once — but two things moved with the catalog: the sentinel names both
//! families and pins the four gaps, and the maskless-refusal gate boots
//! gpt-oss, because the qwen it used to call maskless is not one any more.
//!
//! **AND THEN A FOURTH PIN FLIPPED, THE WAY THE THREE ABOVE DID — EXCEPT
//! THIS ONE WAS THE PIN'S FAULT AND NOT THE GAP'S.** `a6c211c20` hung z-lab's
//! DFlash block drafter off the gpt-oss 20B. A DFlash block is bidirectional
//! over its draft rows, so its full-attention layers each state
//! `attention.masked` — `GPTOSS_20B_DFLASH` carries `windows: &[None; 8]`, and
//! `gptoss-20b-dflash-u4g64-mxfp4-kv-bf16` therefore bakes eight arms. The
//! sentinel below read that as gpt-oss growing the arm, because it was matching
//! a SKU PREFIX and calling the answer a family — which was the same thing
//! right up until a row started reading two attention texts.
//!
//! The written gap is not the stale side and is not deleted here. gpt-oss folds
//! a learned sink through the LSE and `Attention::Masked` exports no LSE, so the
//! gpt-oss TRUNK still cannot carry the arm, and every plain `gptoss-*` row
//! still bakes zero. What the overlay adds is a SECOND attention text, the
//! drafter's own, which has no sinks in it — and CUDA serves it:
//! `attention.masked` is absent from `every_catalog_sku_dispatches`' `REFUSED`
//! (so no arm and no kernel entry refuses it), and the head's 64-wide
//! attention is the width the gpt-oss trunk itself already fires fa2 at, well
//! inside the stamped lattice of 64/128/256/512. So the gate narrows to the
//! question it always meant to ask — does the FAMILY's own text bake an arm —
//! and a row that merely borrows one from an overlaid head is named as such
//! through `models::published::PUBLISHED`, which is where the catalog already
//! records who wears a head.
//!
//! ```text
//! cargo test -p engine-cuda --test masked_axis
//! cargo test -p engine-cuda --features cuda --test masked_axis -- --nocapture
//! ```

use model_dsl::Platform;
use model_ir::{Attention, Operation, Trace};

/// Whether `sku` is a catalog row that reads an OVERLAID DRAFTER HEAD on top
/// of its trunk — the pairing table is the catalog's own record of which rows
/// do (`models::published`), so this asks it rather than reading a `-dflash-`
/// out of the name.
///
/// A head is its own attention text: it states its own layers, its own widths
/// and its own windows, and `dflash::Head { windows: &[None; _] }` is a block
/// that attends bidirectionally over its draft rows. So a masked arm on such a
/// row may belong to the head and say nothing at all about the trunk family
/// underneath it, which is what the gap notes below are about.
fn carries_a_head(sku: &str) -> bool {
    models::published::PUBLISHED.iter().any(|p| p.sku == sku)
}

/// How many `attention.masked` arms a SKU's trace carries.
fn masked_arms(trace: &Trace) -> usize {
    trace.nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Attention(Attention::Masked { .. })))
        .count()
}

/// **WHO DECLARES THE AXIS, AND WHO IS DOCUMENTED AS NOT.**
///
/// Stated rather than assumed, because the device gates below are gemma's and
/// a reader is entitled to know that this is a choice about where the hardware
/// coverage is rather than a fact about the catalog. `masked` is a
/// model-declared fact (design §8), the bits are a runtime input, and a plan
/// with no `attention.masked` node has nowhere for them to go.
///
/// **TWO FAMILIES DECLARE IT.** Gemma always did. Qwen joined at `8cb1b6ce6`,
/// which added `Facts::masked` as a fifth fact FIRST in the priority split and
/// `ops::attn::masked` as a fourth arm of the attention merge. So "a qwen lane
/// carrying a mask is a mask nothing reads", which this gate used to say, is
/// no longer true — and the maskless-refusal gate below had to move off qwen
/// to keep meaning anything.
///
/// **THE OTHER FOUR ARE MASKLESS ON PURPOSE**, each for a written reason
/// (`8cb1b6ce6`'s report): gpt-oss and deepseek-v4 fold learned sinks — and
/// deepseek's pooled long-range merge — through the LSE, which
/// `Attention::Masked` does not export; glm-5 and kimi-k3 attend through MLA
/// latent caches with absorbed queries, and no masked MLA variant exists in
/// the vocabulary. This gate is the sentinel on that pair of claims: the day
/// one of those four grows the arm, or one of the two loses it, is the day
/// this fails and says which.
///
/// **FOUR FAMILIES, NOT FOUR PREFIXES.** Each of those reasons is a statement
/// about a family's OWN attention text, and a catalog row is no longer the same
/// thing as one text: a row that reads an overlaid drafter head bakes that
/// head's attention too. `gptoss-20b-dflash-u4g64-mxfp4-kv-bf16` is the row
/// where the two came apart (`a6c211c20`) — its eight arms are the DFlash
/// block's bidirectional layers, its trunk still bakes none — so the gap is
/// asserted over the rows that carry the family text alone, and the rows
/// wearing a head are recognised by [`carries_a_head`] rather than excused by
/// name. A trunk that grew the arm would still be caught: it would show up on
/// the plain rows, which is exactly where this looks.
///
/// PURE CPU, and not `#[ignore]`d for that reason: it reads the catalog's
/// traces and loads no checkpoint and no device.
#[test]
fn the_masked_axis_is_declared_by_gemma_and_qwen_and_by_nobody_else() {
    // The families whose texts state `attention.masked`, by SKU prefix. The
    // qwen38 row joined with the qwen4 campaign: its hybrid keeps full
    // attention rows, and those rows carry the mask predicate qwen35 does.
    // DiffusionGemma is the gemma4 26B-A4B trunk under a second reading;
    // its attention rows carry the same mask predicate.
    // HunyuanImage 3 is the image half of the same story: its denoise reading
    // states one mask per layer — the guest's slab, a causal text prefix
    // joined to a canvas whose rows all see each other — so the arm it takes
    // is `attention.masked{causal: false}`, the reading the diffusion axis
    // was built for.
    // Muse Glimmer states the mask predicate its qwen siblings do — it is a
    // text decoder with the same rel-bias attention, and it arrived on `dev`
    // while the image families were landing.
    const DECLARE: [&str; 7] = [
        "gemma4-",
        "diffusiongemma-",
        "hunyuanimage3-",
        "muse-glimmer-",
        "qwen35-",
        "qwen36-",
        "qwen38-",
    ];
    // And the four whose OWN attention text has a written reason it cannot
    // state the arm. Their trunk rows must stay maskless; a row of theirs that
    // wears a drafter head is reading a second text and is judged as such.
    const GAPPED: [&str; 4] = ["dsv4-", "glm5-", "gptoss-", "kimik3-"];
    // The artifact the maskless rig below boots, named as a SKU and not as a
    // prefix: `gptoss-` stopped being a maskless prefix at `a6c211c20`, and a
    // rig that resolved a prefix would have picked up the drafting row and
    // silently started testing a masked model against a maskless refusal.
    const MASKLESS_RIG: &str = "gptoss-20b-u4g64-mxfp4-kv-bf16";

    let mut declaring: Vec<(String, usize)> = Vec::new();
    let mut maskless: Vec<String> = Vec::new();
    for row in models::skus() {
        let (sku, trace) = (row.name.as_str(), row.trace);
        let arms = masked_arms(&trace(Platform::Cuda));
        if arms > 0 {
            declaring.push((sku.to_string(), arms));
        } else {
            maskless.push(sku.to_string());
        }
    }

    assert!(
        declaring.iter().all(|(sku, _)| {
            DECLARE.iter().any(|family| sku.starts_with(family)) || carries_a_head(sku)
        }),
        "a family beyond gemma and qwen declares `attention.masked` from its \
         own text — not from an overlaid drafter head — and the device gates \
         in this file were written against gemma: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no SKU declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );

    // BOTH of them, and not just one. A text that DROPPED the arm would pass
    // the prefix check above by simply not appearing in the list.
    for family in DECLARE {
        assert!(
            declaring.iter().any(|(sku, _)| sku.starts_with(family)),
            "no `{family}*` SKU declares `attention.masked` any more, so the \
             axis lost a family: {declaring:?}"
        );
    }

    // And the documented gaps stay gaps — in the family's own text, which is
    // what was written down. A row wearing a drafter head is skipped here and
    // covered by the DECLARE check above instead: its masked arms are the
    // head's, and the head is not the family this note is about.
    for family in GAPPED {
        let grew: Vec<&(String, usize)> = declaring
            .iter()
            .filter(|(sku, _)| sku.starts_with(family) && !carries_a_head(sku))
            .collect();
        assert!(
            grew.is_empty(),
            "`{family}*` grew an `attention.masked` arm in its own text, and a \
             kernel gap was written down as the reason it could not have one — \
             the note and the text now disagree: {grew:?}"
        );
        assert!(
            maskless.iter().any(|sku| sku.starts_with(family)),
            "no `{family}*` SKU is in the catalog at all, so this gate asserts \
             nothing about it"
        );
    }

    // The half that gives the maskless rig below an artifact to stand on. It
    // is a SKU and not a prefix because gpt-oss no longer answers the question
    // as a family: this row must be maskless, and it must still exist.
    assert!(
        maskless.iter().any(|sku| sku == MASKLESS_RIG),
        "`{MASKLESS_RIG}` is either gone from the catalog or bakes an \
         `attention.masked` arm, and it is the artifact the maskless rig boots \
         to watch a maskless model refuse a mask — pick another row that is \
         genuinely maskless and name it here: {maskless:?}"
    );
}

// ── THE GATE: gemma, on a device, with all three classes co-firing ─────────

/// The load, shared with `serve_smoke` in shape and stated here rather than
/// imported because a test binary is its own crate.

/// The maskless rig: the family with a WRITTEN reason it cannot carry the arm.
///
/// **IT REPLACED A QWEN RIG THAT HAD EXACTLY ONE USER.** The refusal gate below
/// used to boot qwen through a `common` module, because qwen baked no masked
/// arm; `8cb1b6ce6` gave it one, and the module went from "the maskless rig" to
/// "a second masked rig" without a line of it changing. Nothing else in this
/// file read it — the device gates all stand on `gemma` — so it is gone rather
/// than kept beside this one, and the name here states the property the gate
/// actually depends on instead of the model that happened to have it.
///
/// gpt-oss folds a learned sink through the LSE, which `Attention::Masked` does
/// not export. That is a written reason rather than an accident, and the
/// sentinel at the top of this file fails the day it stops being true.
///
/// **THE RIG NAMES A ROW, NOT A FAMILY, AND THAT IS THE SECOND TIME THIS HAS
/// MOVED.** It moved off qwen once because qwen grew the arm; it did not have
/// to move again when `a6c211c20` overlaid a DFlash head on the gpt-oss 20B,
/// because what grew an arm was one ROW —
/// `gptoss-20b-dflash-u4g64-mxfp4-kv-bf16`, whose eight arms are the block
/// drafter's bidirectional layers — and not gpt-oss's own text. But a rig that
/// resolved `gptoss-20b-*` would have found the drafting row first and quietly
/// booted a masked model to watch it refuse a mask, so the SKU is spelled out:
/// `MASKLESS_RIG` in the gate above, pinned there against the same catalog the
/// rest of this file reads.
mod maskless {
    

    
    
    

}

// ─────────────────────────────────────────────────────────────────────────────
// THE DEVICE-GEOMETRY CLASS, AGAINST THE HOST-GEOMETRY FIRE IT MUST EQUAL
// ─────────────────────────────────────────────────────────────────────────────

/// The device-geometry gate's fixture: one guest program that is nothing but
/// descriptor ports, and the two fires it is compared through.
///
/// **THE PROGRAM HAS NO BODY ON PURPOSE.** What is under test is the
/// descriptor-port plane — `program::ports` reading committed cells and
/// `serve::prepare` using them — and a stage that computed anything would put
/// its own arithmetic between the seeds this module writes and the geometry
/// the fire resolves. The channels are seeded and the epilogue does nothing,
/// so the cell the port reads is the cell this file wrote, and a wrong logit
/// is the shell's reading of it.
mod devgeo {
    
    
    
    
    
    
    
    

}

/// The gemma load, and the greedy loop the gates above share.
mod gemma {
    
    

    
    
    
    

}
