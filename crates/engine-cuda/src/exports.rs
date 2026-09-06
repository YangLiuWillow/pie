//! **THE EXPORT SEAM, AND THE TWO OP-VOCABULARY SCANS BESIDE IT** — pure IR
//! analysis, and none of it is call order.
//!
//! It lived in [`serve`](crate::serve) because [`Shell::load`] is what asks
//! the questions, and a file whose header says it has no logic was carrying
//! four passes over the trace and the template. Nothing here touches a
//! device, a stream or a fire: given a [`Trace`] and the [`CompiledModel`]
//! that baked it, each function answers one question about WHICH CLASSES RUN
//! WHICH NODE, which is a fact about the artifact and is true before a device
//! is bound.
//!
//! * [`Exports::of`] — the declared seams (`out`, `mtp`, `attn.scores`,
//!   `mtp.drafts`)
//!   resolved to values, layers and the class sets that fill them.
//! * [`masked_classes`] / [`corrected_classes`] — the same reading taken from
//!   the OP VOCABULARY rather than from a seam: which classes' windows run an
//!   `attention.masked` arm, and which run a `linear.lora_correct` one.
//! * [`regions_shifting`] — the op vocabulary read a third time and answered
//!   PER REGION instead of per class: which regions hold nothing but ops that
//!   address off the staged seat's start ([`crate::shifted`]), and can
//!   therefore be replayed somewhere other than the fire's row zero.
//! * [`regions_launching_schedules`] — which region LAUNCHES each attention
//!   schedule, so a schedule is carved at the ceilings of the region that
//!   reads it rather than the prepare region that built it.
//! * [`regions_lane_shifting`] — the same reading one AXIS over
//!   ([`crate::lane_shifted`]): which regions hold nothing but ops that find
//!   their own LANE inside the fire, and can therefore be replayed somewhere
//!   other than the fire's lane zero.
//!
//! [`Shell::load`]: crate::serve::Shell::load

use model_compiler::CompiledModel;
use model_ir::{Operands, Trace, ValueId};

use crate::error::{Fault, Result};

/// The names `model_dsl::seam` states for the values a reader touches after
/// the graph has run — `out`, `mtp`, `attn.scores`, in that order.
///
/// **READ FROM THE COMPILER, NOT SPELLED AGAIN** (palo C3b). This crate does
/// not depend on the authoring surface, and until this wave it kept its own
/// copy of the literal `"out"` with a comment in each place saying the other
/// one existed. `model_compiler::arena` is what gives these values their
/// delivery tail, so it is the honest place for the list to live: a shell
/// reading a name the carve does not pin would be reading bytes the carve was
/// free to give away.
pub(crate) const OUT_SEAM: &str = model_compiler::EXPORT_SEAMS[0];
pub(crate) const MTP_SEAM: &str = model_compiler::EXPORT_SEAMS[1];
pub(crate) const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];
/// The float readouts a plan may carry instead of `out` (`velocity`,
/// `hidden`): a plan with one of these and no `out` still computes
/// something a reader takes.
pub(crate) const FLOAT_READOUT_SEAMS: [&str; 3] = model_compiler::FLOAT_READOUT_SEAMS;
/// The VAE decode readout (D8): the pixel plane and its grid.
pub(crate) const PIXELS_SEAM: &str = model_compiler::EXPORT_SEAMS[6];
pub(crate) const DRAFTS_SEAM: &str = model_compiler::EXPORT_SEAMS[3];

/// One declared export, resolved against this load's plan and bake.
///
/// **A VALUE AND THE CLASSES THAT FILL IT, AND BOTH HALVES ARE USED.** The
/// value is what the fire's carve turns into a rectangle; the class set is
/// what a lane's word is checked against, because an export is written by an
/// ARM and an arm runs over a window. `Shell::masked` and `Shell::corrected`
/// are the same reading taken from the op vocabulary; this one is taken from
/// the seam, because a draft head's attention and a trunk layer's attention
/// are the same `Attention::Prefill` variant and only the export tells them
/// apart.
#[derive(Debug, Clone)]
pub struct Export {
    /// The exported value, as the plan's `Seam` row names it.
    pub value: ValueId,
    /// Which transformer layer it came from, for a per-layer export.
    pub layer: u32,
    /// The classes whose window runs the node that writes it.
    pub classes: model_ir::ClassSet,
}

/// This load's declared exports (design §9), resolved once at boot.
#[derive(Debug, Clone)]
pub(crate) struct Exports {
    /// The classes whose window writes the `out` seam (empty without one).
    pub(crate) out_classes: model_ir::ClassSet,
    /// The trunk's logits. `None` for a plan whose readout is a float seam
    /// (`velocity`, `hidden`) — a denoiser has no logits — and a plan with
    /// neither is refused at boot, since a fire would compute nothing a
    /// reader can take. M0: reading the float seams back is the runtime
    /// agent's; this shell reads logits only.
    pub(crate) out: Option<ValueId>,
    /// The draft head's logits over the draft window, for a SKU whose model
    /// text declares one (palo C3).
    pub(crate) mtp: Option<Export>,
    /// The draft head's token plane — the block drafter's picks, `[rows,
    /// depth]` i32 — for a SKU whose readout plants one (`mtp.drafts`).
    pub(crate) drafts: Option<Export>,
    /// The token plane's declared width: the depth a guest sizes its
    /// `mtp_drafts` read by, fixed by the model text. Zero without a plane.
    pub(crate) drafts_depth: u32,
    /// The attention's per-query mass, one entry per attention layer that
    /// exports it, in the plan's own order (palo C4).
    pub(crate) scores: Vec<Export>,
    /// The union of every capture column's classes — the set a capturing
    /// lane's word must land in, and empty for an artifact with no capture
    /// arm at all.
    pub(crate) capturing: model_ir::ClassSet,
    /// The denoiser's prediction (`seam::VELOCITY`), for a plan that plants
    /// one (design D3): the eta `velocity()` intrinsic and the
    /// `ReadoutSeam::Velocity` readback point at it.
    pub(crate) velocity: Option<Export>,
    /// The hidden-state exports (`seam::HIDDEN`), one per layer they were
    /// planted in, in plan order. The LAST one is what a `hidden()`
    /// intrinsic and a `ReadoutSeam::Hidden` readback read.
    pub(crate) hidden: Vec<Export>,
    /// The pixel planes and their `[Clips, 4]` grids (D8), one per
    /// planting of `seam::PIXELS` in plan order — a VAE plants one on its
    /// decode arm and one on its encode arm — each with the classes whose
    /// arm writes the plane; empty for a plan that plants none.
    pub(crate) pixels: Vec<(Export, ValueId)>,
}

/// Which seam a fire's host readback mirrors, and the value it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadoutSeam {
    pub(crate) seam: engine::fire::ReadoutSeam,
    pub(crate) value: ValueId,
}

impl Exports {
    /// The pixel plane and grid a lane of `class` reads back from: the
    /// planting whose arm the class runs, else — for a plan with one
    /// planting — that one.
    #[must_use]
    pub(crate) fn pixels_for(&self, class: Option<usize>) -> Option<(ValueId, ValueId)> {
        class
            .and_then(|class| {
                self.pixels
                    .iter()
                    .find(|(export, _)| export.classes.contains(class))
            })
            .or_else(|| (self.pixels.len() == 1).then(|| &self.pixels[0]))
            .map(|(export, grid)| (export.value, *grid))
    }

    /// Every pixels planting's row width, in plan order — one per
    /// `seam::PIXELS` the text plants. A planting whose width is symbolic
    /// is skipped rather than guessed, so a plan of only symbolic pixel
    /// widths reads as planting none.
    pub(crate) fn pixels_widths<'a>(&'a self, trace: &'a Trace) -> impl Iterator<Item = u32> + 'a {
        self.pixels.iter().filter_map(move |(export, _)| {
            crate::store::kv::width_of(trace, export.value)
                .ok()
                .and_then(|width| u32::try_from(width).ok())
        })
    }

    /// The seam a lane of `class` reads back from — the export its OWN arm
    /// writes (a multi-reading plan plants `hidden` on its encoder arm and
    /// `velocity` on its denoise arm, design D1/D5): `out` when the class
    /// writes it, else its `velocity`, else the last `hidden` its class
    /// writes; a class that writes none falls back to the plan-wide
    /// [`readout`](Exports::readout).
    #[must_use]
    pub(crate) fn readout_for(&self, class: usize) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out
            && self.out_classes.contains(class)
        {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = self.velocity_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        if let Some(hidden) = self.hidden_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Hidden,
                value: hidden.value,
            });
        }
        self.readout()
    }

    /// The velocity export a lane of `class` writes, else the plan's (for
    /// a class writing none — the plan-wide answer keeps a text SKU's one
    /// seam bound as before).
    #[must_use]
    pub(crate) fn velocity_for(&self, class: usize) -> Option<&Export> {
        self.velocity
            .as_ref()
            .filter(|export| export.classes.contains(class))
    }

    /// The last hidden export a lane of `class` writes.
    #[must_use]
    pub(crate) fn hidden_for(&self, class: usize) -> Option<&Export> {
        self.hidden
            .iter()
            .rev()
            .find(|export| export.classes.contains(class))
    }

    /// The seam a lane's rows are read back from: `out` when the plan has
    /// one, else `velocity`, else the last `hidden` — a plan with none was
    /// refused at [`Exports::of`].
    #[must_use]
    pub(crate) fn readout(&self) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = &self.velocity {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        self.hidden.last().map(|hidden| ReadoutSeam {
            seam: Seam::Hidden,
            value: hidden.value,
        })
    }
}

impl Exports {
    /// Resolve the export seams against a plan and the bake that placed them.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a plan with no export at all: neither an
    /// `out` seam nor a float readout.
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let float_readout = trace.seams.iter().any(|seam| {
            FLOAT_READOUT_SEAMS.contains(&seam.seam.as_str()) && !seam.values.is_empty()
        });
        if out.is_none() && !float_readout {
            return Err(Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam and no float readout ({}), so a fire would compute \
                     nothing a reader can take",
                    FLOAT_READOUT_SEAMS.join(", ")
                ),
            });
        }
        let named = |name: &str| -> Vec<Export> {
            trace
                .seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| {
                    let layer = seam.layer.unwrap_or(0);
                    seam.values.iter().map(move |value| (layer, *value))
                })
                .map(|(layer, value)| Export {
                    value,
                    layer,
                    classes: writer_classes(trace, compiled, value),
                })
                .collect()
        };
        let scores = named(SCORES_SEAM);
        let drafts = named(DRAFTS_SEAM).into_iter().next();
        let drafts_depth = match &drafts {
            Some(export) => {
                let width = model_exec::store::kv::width_of(trace, export.value).map_err(|why| {
                    Fault::Unbound {
                        what: format!("the `{DRAFTS_SEAM}` export's width: {why}"),
                    }
                })?;
                match u32::try_from(width) {
                    Ok(depth) if depth > 0 => depth,
                    _ => {
                        return Err(Fault::Unbound {
                            what: format!("a `{DRAFTS_SEAM}` export {width} wide drafts nothing"),
                        });
                    }
                }
            }
            None => 0,
        };
        let mut capturing = model_ir::ClassSet::default();
        for export in &scores {
            for class in export.classes.iter() {
                capturing.insert(class);
            }
        }
        let pixels = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| match seam.values.as_slice() {
                [plane, grid, ..] => Some((
                    Export {
                        value: *plane,
                        layer: seam.layer.unwrap_or(0),
                        classes: writer_classes(trace, compiled, *plane),
                    },
                    *grid,
                )),
                _ => None,
            })
            .collect();
        Ok(Exports {
            out_classes: out.map_or_else(model_ir::ClassSet::default, |out| {
                writer_classes(trace, compiled, out)
            }),
            out,
            mtp: named(MTP_SEAM).into_iter().next(),
            drafts,
            drafts_depth,
            scores,
            capturing,
            velocity: named(FLOAT_READOUT_SEAMS[0]).into_iter().next(),
            hidden: named(FLOAT_READOUT_SEAMS[1]),
            pixels,
        })
    }
}

/// The classes whose window runs the node that writes `value`.
///
/// **THE NODE, NOT THE OP NAME.** An export is told apart from the trunk by
/// WHAT IT IS, not by which kernel wrote it: the draft head's readout and the
/// trunk's are both `linear.lm_head`, and the capture arm's output and a
/// pooled attention's are both `[rows, heads]` F32. Asking which regions hold
/// the writing node is the one reading that cannot be fooled by a model text
/// reusing an op.
fn writer_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    // A merged export is written by its arms: every class that writes any
    // arm reads the seam back from the merged column.
    if let Some(model_ir::Def::Merge(arms)) =
        trace.values.get(value.0 as usize).map(|decl| &decl.def)
    {
        let mut classes = model_ir::ClassSet::default();
        for (arm, _) in arms {
            for class in writer_classes(trace, compiled, *arm).iter() {
                classes.insert(class);
            }
        }
        return classes;
    }
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}

/// The classes whose window runs an `attention.masked` arm.
///
/// **WHETHER THIS ARTIFACT HAS ANYWHERE FOR A MASK TO GO.** `masked` is a
/// fact the model declares (design §8), so a plan with no `attention.masked`
/// arm cannot serve one, and accepting the bits anyway would answer with the
/// unmasked continuation.
///
/// A CLASS SET rather than a boolean, because the question a fire asks is per
/// lane: does the class this lane's word resolved to run the masked arm? The
/// word and the mask are stamped at two instants by two parties — the runtime
/// computes the word from the model's `Classify::of`, the caller states the
/// mask — and this set is what lets the shell check that they agree
/// (`Fault::{Maskless, MaskWord}`).
#[must_use]
pub(crate) fn masked_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Attention(model_ir::Attention::Masked { .. })
        )
    })
}

/// The classes whose window runs a `linear.lora_correct` arm.
///
/// [`masked_classes`]'s adapter-axis twin, read off the bake for the same
/// reason and checked against a submission the same way, with the same three
/// consequences: an artifact with no correction op has nowhere for an adapter
/// id to go (`Fault::Adapterless`), a lane whose word puts it outside the
/// correction's window may not carry one and a lane whose word puts it inside
/// must (`Fault::AdapterWord`), and a fire in whose composition NO class of
/// this set has rows never stages the routes vector, never binds the seat,
/// and never launches the arm.
#[must_use]
pub(crate) fn corrected_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Linear(model_ir::Linear::LoraCorrect { .. })
        )
    })
}

/// The classes whose window SCATTERS TOWER OUTPUT INTO TOKEN ROWS — the
/// MEDIA classes, in the only vocabulary this shell has for the word (the
/// multi-unit bodies wave).
///
/// [`masked_classes`]'s fourth twin, and it exists for
/// [`decoding_classes`]'s reason exactly: the bodies path's load-time arming
/// (`Shell::arm_bodies`) has to synthesize a fire that carries an IMAGE
/// before any caller has shown it one, and a shell cannot compute a lane's
/// fact word — the word is the model's `Classify::of`, runtime-side, and
/// which bit is `media` stays the model's business (multimodal §15). So the
/// question is asked about OPS: a class whose window runs the embed merge is
/// a class an image lane lands in, and `Class::word` then names a word that
/// resolves back to it.
///
/// **THE MERGE AND NOT THE TOWER.** The tower's own regions are on the PATCH
/// axis and their rectangles are `Dim::Patches`; a class does not "run" them
/// in the sense this predicate means, because an axis-empty fire simply does
/// not launch that unit. What a media lane's class does run is the scatter
/// that puts the tower's soft tokens onto that lane's placeholder rows —
/// `layout.scatter_rows` or its dropping form — which is a TRUNK-unit node
/// with a class mask, guarded on the media fact. That guard is the thing this
/// reads.
///
/// Empty for every text-only artifact, and then the arming pass's tower arm
/// enumerates nothing at all — which is the same nothing a plan with no
/// decode arm gives the decode arm, and is why neither needs a second clause
/// anywhere.
#[must_use]
pub(crate) fn media_classes(trace: &Trace, compiled: &CompiledModel) -> model_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            model_ir::Operation::Layout(
                model_ir::Layout::ScatterRows { .. } | model_ir::Layout::ScatterLiveRows { .. }
            )
        )
    })
}

/// How many declared readings (`Request::in_reading`) the enumeration
/// tries per stream. The IR declares no reading count, so every reading a
/// family could index below this is classified; one past the family's last
/// lands where the family's classifier puts it (the default arm, or a word
/// no class has, which is dropped).
const READINGS: u8 = 8;

/// Every request shape, classified by the model: per class, the requests
/// that land in it, fewest flags first. A class no request reaches is one
/// no caller can bring, and the arming pass does not synthesize it. The
/// seven boolean facts, every [`model_ir::Stream`] and [`READINGS`] readings
/// are enumerated, so a class only a stream or a reading selects (design
/// D2: the image lanes' arm of an MM-DiT) is reachable by the arming pass.
#[must_use]
pub(crate) fn landing_requests(
    classify: model_ir::ClassifyFn,
    classes: &model_ir::ClassTable,
) -> Vec<Vec<model_ir::Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    for reading in 0..READINGS {
        for stream in model_ir::Stream::ALL {
            for bits in 0..128u32 {
                let request =
                    model_ir::Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
                        .adapted(bits & 4 != 0)
                        .drafting(bits & 8 != 0)
                        .capturing_scores(bits & 16 != 0)
                        .with_media(bits & 32 != 0)
                        .denoising(bits & 64 != 0)
                        .on_stream(stream)
                        .in_reading(reading);
                let word = classify(&request) & classes.mask;
                if let Some(class) = classes.class_of(word) {
                    // A stream or reading the family packs no bit for lands
                    // on its default's word; the extra entries sort after
                    // it (`request_flags`) and cost a text family nothing.
                    landing[class].push(request);
                }
            }
        }
    }
    for requests in &mut landing {
        requests.sort_by_key(request_flags);
    }
    landing
}

fn request_flags(request: &model_ir::Request) -> u32 {
    u32::from(request.query_len() != 1)
        + u32::from(request.has_custom_mask())
        + u32::from(request.has_adapter())
        + u32::from(request.drafts())
        + u32::from(request.captures_scores())
        + u32::from(request.has_media())
        + u32::from(request.denoise())
        + u32::from(request.stream() != model_ir::Stream::Text)
        + u32::from(request.reading() != 0)
}

/// The DECODE classes: every request that lands in one carries a single
/// row, so a lane of it is one row and its rung is the lane ceiling.
#[must_use]
pub(crate) fn decoding_of(landing: &[Vec<model_ir::Request>]) -> model_ir::ClassSet {
    model_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| {
                !requests.is_empty() && requests.iter().all(|request| request.query_len() == 1)
            })
            .map(|(class, _)| class),
    )
}

/// **WHICH TEMPLATE REGIONS CAN MOVE THEIR OWN BASE** — one `bool` per
/// region of [`CompiledModel::template`], in region order, `true` when EVERY
/// op in it is named by [`crate::shifted`].
///
/// [`masked_classes`]'s and [`corrected_classes`]'s structural twin — the
/// same walk of the same template testing the same node ops — and it differs
/// in exactly two ways, both forced by what the answer is for. It asks ALL
/// rather than ANY, because one guard-only op in a region addresses the wrong
/// row for the whole region's launch; and it answers PER REGION rather than
/// per class, because the thing that gets a seat is a region's launch and the
/// thing that reads it is a region's kernel. A class set could not say it: two
/// classes share a region, and it is the region that either moves or does not.
///
/// **A REGION WITH NO NODES IS `true`, AND IT IS NOT MERELY THE VACUOUS
/// ANSWER.** `all` over an empty range is `true` for free, and here that is
/// also what the question means: a region carries a window because of its
/// MASK — `Windows::of` cuts one per template region off the class table and
/// never looks at `Region::nodes` — so an empty region can hold a windowed
/// rectangle and still launch nothing over it. Nothing in it can address a
/// row, so nothing in it can address the wrong one, and refusing it would
/// refuse a body over a region that computes no bytes. The compiler ships no
/// such region; this says what would be true of one.
///
/// A node index the trace does not hold reads as NOT shifting, which refuses
/// the region. The two tables are baked together and that cannot happen; if it
/// ever does, the narrow reading is the one that stays sound.
#[must_use]
pub(crate) fn regions_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace.nodes.get(node as usize).is_some_and(|node| {
                    let name = model_ir::Operands::name(&node.op);
                    crate::shifted(name) || crate::PLANNED.contains(&name)
                })
            })
        })
        .collect()
}

/// **WHICH TEMPLATE REGIONS FIND THEIR OWN LANE** — one `bool` per region of
/// [`CompiledModel::template`], in region order, `true` when every op in it
/// either is named by [`crate::lane_shifted`] / [`crate::PLANNED`] or NAMES
/// NOTHING THAT IS LANE-INDEXED.
///
/// [`regions_shifting`]'s twin one axis over, and everything that function's
/// note argues about ALL-rather-than-ANY, about answering per REGION rather
/// than per class, and about an empty region reading `true` holds here word
/// for word.
///
/// **AND IT IS AN OPERAND WALK WHERE ITS TWIN IS A NAME LOOKUP, WHICH IS THE
/// ONE REAL DIFFERENCE AND IS FORCED BY WHERE THE HAZARD LIVES.** The row
/// axis's hazard is what a KERNEL does with the pointer it is handed, so only
/// a name can answer it. The lane axis's hazard is what THIS SHELL hands over:
/// `Run::pool` advances the page bounds and last-page fills by `lane_offset`,
/// `Run::recurrent` advances the slot map, the fold predicate and the commit
/// length, and `Run::cut`'s lane column advances every operand whose leading
/// `Dim` counts lanes or images. Those are the three doors, and an op reaches
/// them through its OPERANDS — a `Def::Cache` space for the first two, a lane
/// -shaped rectangle for the third. So an op that names neither cannot be
/// handed a `lane_offset`-baked pointer at all, whatever it is called, and
/// refusing it would cost a body for no hazard.
///
/// **WHICH LEAVES EXACTLY TWO WAYS TO PASS**, and they are the two
/// [`crate::lane_shifted`] enumerates and the one this walk adds:
///
/// * the op is on that list, so the tables it names are handed over WHOLE and
///   it finds its lane in a staged datum or off the seat's `win[3]`;
/// * or the op is a planner ([`crate::PLANNED`]), which puts no node in the
///   captured graph and rebuilds its schedule every fire against that fire's
///   own staged geometry, lane offset included;
/// * or the op names nothing lane-indexed, and the question does not arise.
///
/// A node index the trace does not hold, an operand the trace does not
/// declare, and an op family this walk cannot collect all read as NOT
/// lane-shifting — which refuses the region. That is the safe direction on the
/// axis where being wrong reads another lane's state, and it is
/// [`regions_shifting`]'s own tie-break.
#[must_use]
pub(crate) fn regions_lane_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .all(|node| lane_shifting_node(trace, node))
        })
        .collect()
}

/// One node's answer for [`regions_lane_shifting`] — the name lookup first,
/// because it is the cheap one and because a name on the list has already
/// argued its operands.
fn lane_shifting_node(trace: &Trace, node: u32) -> bool {
    let Some(node) = trace.nodes.get(node as usize) else {
        return false;
    };
    let name = Operands::name(&node.op);
    if crate::lane_shifted(name) || crate::PLANNED.contains(&name) {
        return true;
    }
    // **INPUTS AND OUTPUTS BOTH**, `window::copyable`'s reason exactly: a
    // rectangle this op WRITES is resolved through the same `Run::cut` the
    // ones it reads are, and a lane-shaped output would be advanced by the
    // same number.
    let mut operands: Vec<ValueId> = Vec::new();
    node.op.inputs(&mut operands);
    node.op.outputs(&mut operands);
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        // **A CACHE SPACE IS THE FIRST DOOR, PAGED AND RECURRENT ALIKE.** Both
        // `Run::pool` and `Run::recurrent` slice their per-lane tables at
        // `lane_offset`, and only the absolute doors beside them
        // (`pool_absolute`, `recurrent_absolute`) do not — which is what the
        // names on the list took and what nothing off it did.
        if matches!(&decl.def, model_ir::Def::Cache(_)) {
            return false;
        }
        // **AND A LANE-SHAPED RECTANGLE IS THE SECOND**, whatever declared it:
        // `Run::cut`'s lane column is `(span.lane_offset, span.lanes + k)` for
        // every one of these, which is a pointer advanced by a number the key
        // does not fix. `GeomKind::Indices` is spelled `Dim::Lanes` and cut is
        // excluded from slicing it — but its BOUNDS are not, and an op naming
        // one names the other, so nothing is bought by carving an exception
        // here.
        let model_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            // A plan payload is host state resolved through `Run::slot`, and
            // its own window is the region that BUILT it — which is this one,
            // because a schedule may only be read where it was built
            // (`model`'s `no_schedule_straddles_its_readers`).
            return true;
        };
        !matches!(
            shape.first(),
            Some(
                model_ir::Dim::Lanes
                    | model_ir::Dim::LanesPlus(_)
                    | model_ir::Dim::Images
                    | model_ir::Dim::ImagesPlus(_)
            )
        )
    })
}

/// The union of the region masks whose regions run a node `wanted` accepts.
///
/// The one shape both readings above are: a region is a window over classes,
/// a node is inside a region or it is not, and the answer is which classes'
/// windows carry at least one node of the family asked about.
fn classes_running(
    trace: &Trace,
    compiled: &CompiledModel,
    wanted: impl Fn(&model_ir::Operation) -> bool,
) -> model_ir::ClassSet {
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        let runs = region.nodes.clone().any(|node| {
            trace
                .nodes
                .get(node as usize)
                .is_some_and(|node| wanted(&node.op))
        });
        if runs {
            for class in region.mask.iter() {
                classes.insert(class);
            }
        }
    }
    classes
}

/// **THE FLOAT PORTS AND THE PACKING SELECTIONS A PLAN READS** (design
/// D2/D3), read once off the plan at load: what the inputs store carves, and
/// what `prepare` builds per fire.
#[derive(Debug, Clone, Default)]
pub(crate) struct Feeds {
    /// Every float port the plan declares, with the classes whose window
    /// runs a node reading it — the classes a lane must feed it in.
    pub(crate) ports: Vec<(crate::inputs::PortSeat, model_ir::ClassSet)>,
    /// Every row selection a packing table is keyed by, in first-seen order
    /// (the order the inputs store carves them in).
    pub(crate) selections: Vec<model_ir::Selection>,
    /// Every float port merged STRAIGHT into a stream (`Value::merge` with
    /// the port as an arm): the merged column has no node writing that arm's
    /// rows, so the fire lands the port's rows in it before the walk — the
    /// lanes the arm's guard selects, from the port rectangle the feed
    /// filled (zeros for a lane nothing fed). The compiler keeps such a
    /// column live from the fire's first instant.
    pub(crate) merged: Vec<MergedPort>,
    /// Merges with an input arm this shell cannot land: a non-port input, or
    /// an arm guard that is not a conjunction of facts. Refused at load.
    pub(crate) unlanded: Vec<ValueId>,
}

/// One float port that is a merge's arm.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergedPort {
    /// The merged value whose column the rows land in.
    pub(crate) merge: ValueId,
    /// Which port.
    pub(crate) seat: crate::inputs::PortSeat,
    /// The arm's lanes.
    pub(crate) select: model_ir::Selection,
}

impl Feeds {
    #[must_use]
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Feeds {
        use model_ir::{Def, GeomKind, RuntimeInput, Ty};
        let mut feeds = Feeds::default();
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            let dtype = match &decl.ty {
                Ty::Tensor { dtype, .. } => *dtype,
                Ty::Struct(_) => continue,
            };
            let seat = match *input {
                RuntimeInput::Latents { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Latents,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::LaneVector { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::LaneVector,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::Context { port, width } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Context,
                    port,
                    width,
                    dtype,
                }),
                RuntimeInput::AxisPositions { port, axes } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::AxisPositions,
                    port,
                    width: u32::from(axes),
                    dtype,
                }),
                // The voxel port (D8). It is a seat like any other for the
                // purpose of "which class must feed this", but its rectangle
                // is NOT in the inputs store: the payload lives in
                // `voxels::Store`, below the fire's other inputs, so
                // `seats()` keeps it out of the token-axis carve.
                RuntimeInput::Voxels { port, channels } => Some(crate::inputs::PortSeat {
                    kind: engine::fire::PortKind::Voxels,
                    port,
                    width: channels,
                    dtype,
                }),
                RuntimeInput::RowPermutation { select }
                | RuntimeInput::Geometry {
                    kind:
                        GeomKind::GroupIndptr { select }
                        | GeomKind::LaneIndptr { select }
                        | GeomKind::ReferenceTag { select },
                    ..
                } => {
                    if !feeds.selections.contains(&select) {
                        feeds.selections.push(select);
                    }
                    None
                }
                _ => None,
            };
            if let Some(seat) = seat {
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                match feeds
                    .ports
                    .iter_mut()
                    .find(|(have, _)| have.kind == seat.kind && have.port == seat.port)
                {
                    Some((_, classes)) => {
                        for class in readers.iter() {
                            classes.insert(class);
                        }
                    }
                    None => feeds.ports.push((seat, readers)),
                }
            }
        }
        // The ports merged straight into a stream.
        for (at, decl) in trace.values.iter().enumerate() {
            let Def::Merge(arms) = &decl.def else {
                continue;
            };
            for (arm, guard) in arms {
                let Def::Input(input) = &trace.values[arm.0 as usize].def else {
                    continue;
                };
                let seat = feeds
                    .ports
                    .iter()
                    .find(|(seat, _)| {
                        let (kind, port) = match *input {
                            RuntimeInput::Latents { port, .. } => {
                                (engine::fire::PortKind::Latents, port)
                            }
                            RuntimeInput::LaneVector { port, .. } => {
                                (engine::fire::PortKind::LaneVector, port)
                            }
                            RuntimeInput::Context { port, .. } => {
                                (engine::fire::PortKind::Context, port)
                            }
                            RuntimeInput::AxisPositions { port, .. } => {
                                (engine::fire::PortKind::AxisPositions, port)
                            }
                            _ => return false,
                        };
                        seat.kind == kind && seat.port == port
                    })
                    .map(|(seat, _)| *seat);
                let (Some(seat), Some(select)) = (seat, model_ir::Selection::of(guard)) else {
                    // A non-port input merged, or an arm whose guard is no
                    // conjunction: nothing lands it. Left to the load's
                    // refusal (`Shell::load`), which names the value.
                    feeds.unlanded.push(ValueId(at as u32));
                    continue;
                };
                // The port is read through the merge: the classes whose
                // window reads the MERGED column, among the arm's own lanes,
                // are the classes that must feed it.
                let readers = reader_classes(trace, compiled, ValueId(at as u32));
                if let Some((_, classes)) = feeds
                    .ports
                    .iter_mut()
                    .find(|(have, _)| have.kind == seat.kind && have.port == seat.port)
                {
                    for class in readers.iter() {
                        let word = compiled.classes.classes[class].word();
                        if select.holds(word) {
                            classes.insert(class);
                        }
                    }
                }
                feeds.merged.push(MergedPort {
                    merge: ValueId(at as u32),
                    seat,
                    select,
                });
            }
        }
        feeds
    }

    /// The port seats alone, in the store's order.
    #[must_use]
    /// The port rectangles the INPUTS store carves — every seat but the
    /// voxel one, whose payload the voxel store reserves at the ladder's
    /// ceilings instead (design D8).
    pub(crate) fn seats(&self) -> Vec<crate::inputs::PortSeat> {
        self.ports
            .iter()
            .map(|(seat, _)| *seat)
            .filter(|seat| seat.kind != engine::fire::PortKind::Voxels)
            .collect()
    }
}

/// The classes whose window runs a node that READS `value` —
/// [`writer_classes`]'s mirror, for the ports: a lane of one of these
/// classes must feed the port, a lane of any other need not.
fn reader_classes(trace: &Trace, compiled: &CompiledModel, value: ValueId) -> model_ir::ClassSet {
    let mut inputs: Vec<ValueId> = Vec::new();
    let mut readers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        if inputs.contains(&value) {
            readers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = model_ir::ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| readers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}

/// **WHICH TEMPLATE REGION LAUNCHES EACH ATTENTION SCHEDULE** — one entry per
/// `Trace::values` id, holding the region of the node that READS that
/// schedule. `None` for a value no launch reads, and `None` again when two
/// regions read one: nothing can then speak for both, and the caller carves
/// nothing rather than carve at the wrong region's ceilings.
///
/// **THIS IS [`regions_shifting`]'S CONSUMER SIDE, AND IT EXISTS BECAUSE THE
/// TWO REGIONS ARE NEVER THE SAME ONE.** A schedule is BUILT in a
/// `Phase::Prepare` region — a region holds one phase, so a planner op never
/// shares a region with the launch that reads it — and such a region holds
/// nothing but [`crate::PLANNED`] ops, so [`regions_shifting`] reads it as
/// shifting for free: it names no kernel that could address the wrong row.
/// The region that LAUNCHES the schedule answers for itself, and a trunk
/// region carrying one `linear.matmul` (`Reads::Nothing`) does not shift.
///
/// So the ceilings a schedule is carved at — how many requests it names, and
/// which lane it counts them from — must be the LAUNCHER's and not the
/// builder's. It is the launch that is handed a boundary vector, and
/// `Run::ragged_q` picks that vector off the launcher's own standing:
/// carving at the builder's ceiling hands a 32-request schedule a one-lane
/// vector, which `kernels_cuda::attn`'s `lanes_carry` refuses by name.
///
/// `model_exec::store::check::no_schedule_straddles_its_readers` pins the two
/// regions to one MASK, so they see one window and one span; it does not pin
/// them to one region, and the `shifted`/`lane_shifted` bits are per region.
#[must_use]
pub(crate) fn regions_launching_schedules(
    trace: &Trace,
    compiled: &CompiledModel,
) -> Vec<Option<u32>> {
    let mut out: Vec<Option<u32>> = vec![None; trace.values.len()];
    let mut claimed: Vec<bool> = vec![false; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        let here = u32::try_from(at).unwrap_or(u32::MAX);
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                // A host struct is the only thing a plan op defines and the
                // only thing a launch reads it as; a rectangle operand says
                // nothing about schedules.
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, model_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let Some(slot) = out.get_mut(at) else {
                    continue;
                };
                if claimed[at] {
                    // A second region reading one schedule: neither can speak
                    // for the other, so nobody does.
                    if *slot != Some(here) {
                        *slot = None;
                    }
                } else {
                    claimed[at] = true;
                    *slot = Some(here);
                }
            }
        }
    }
    out
}
