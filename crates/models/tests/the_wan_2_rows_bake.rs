//! **THE WAN 2.2 ROWS TRACE, CLASSIFY AND BAKE: FOUR ARMS UNDER ONE PLAN,
//! A CACHELESS ENCODER WITH A RELATIVE BIAS, A VIDEO DENOISER WHOSE
//! CROSS-ATTENTION SPANS TWO STREAMS, AND A CAUSAL-CONV DECODER WITH A
//! FRAME CACHE PER CONV.**
//!
//! ```text
//! cargo test -p models --test the_wan_2_rows_bake
//! ```
//!
//! `wan22-ti2v-5b` is the first catalog row whose plan carries a text
//! encoder with NO kv space (umT5 is bidirectional), a denoiser reading
//! its context from a second stream, and a voxel arm holding
//! `CacheRow::State` slabs (design D5, D8). `wan22-mini-d128` /
//! `wan22-mini-nano` are the parity fixtures `scripts/imagegen/wan22_golden.py
//! --mini` writes. What is asserted:
//!
//! ```text
//! (a) every row traces on every platform; the flagship declares one
//!     state slab per causal conv of the decoder (32) and no kv row, the
//!     miniatures nothing
//! (b) the flagship's seams are `hidden`, `velocity` and `pixels` (twice:
//!     the first-frame arm and the later-frames arm) and never `out`; the
//!     miniatures' is `velocity` alone
//! (c) the ports the trace reads are the ports the facts declare, at the
//!     facts' widths and kind-relative indices, and the named ports
//!     resolve to `model::port`
//! (d) the (reading, stream) lanes the facts list classify into distinct
//!     classes where every merge resolves
//! (e) two ragged reads per block — the self-attention self-paired over
//!     the video selection's group CSR, the cross-attention over the
//!     video CSR against the context CSR under an `Or` guard — and one
//!     per encoder layer, self-paired over the lane CSR at `sm_scale = 1`
//!     under its own `RelativeBias` table
//! (f) every rope turns the row's `[d − 4⌊d/6⌋, 2⌊d/6⌋, 2⌊d/6⌋]` split of
//!     the whole head, interleaved, at θ 10 000 — `[44, 42, 42]` at 128,
//!     `[8, 8, 8]` at 24 — and only the self-attentions turn
//! (g) every row bakes on every platform under a voxel ladder; the
//!     flagship refuses a budget with none
//! (h) the generative facts: readings dense from 0, `text` embedding
//!     tokens with no kv, the latent space `(48, 1×2×2, /16, /4)` and the
//!     static shift 5.0
//! (i) the modulation is a per-lane f32 vector over a bf16 trunk — two
//!     scale-shift pairs per block and one at the head — and every gated
//!     fold aliases its residual
//! ```

use std::collections::{BTreeMap, BTreeSet};

use model_dsl::{
    Attention, CacheRow, Classify, Def, Dim, Dtype, Elementwise, GeomKind, Guard, Operands,
    Operation, Platform, RaggedMask, Request, RopeForm, RuntimeInput, Selection, Stream, Trace, Ty,
    ValueId, seam,
};
use models::wan_2::forward::Facts;
use models::wan_2::model::{self, Dims};
use models::{PortKind, ReadoutKind, ScheduleKind};

const TI2V: &str = "wan22-ti2v-5b-bf16-kv-bf16";
const D128: &str = "wan22-mini-d128-bf16-kv-bf16";
const NANO: &str = "wan22-mini-nano-bf16-kv-bf16";
const ROWS: [&str; 3] = [TI2V, D128, NANO];

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn row(sku: &str) -> &'static models::Sku {
    models::sku(sku).unwrap_or_else(|| {
        let names: Vec<&str> = models::skus().map(|row| row.name.as_str()).collect();
        panic!("this build ships no `{sku}`; rows are {names:#?}")
    })
}

fn trace(sku: &str, platform: Platform) -> Trace {
    (row(sku).trace)(platform)
}

fn dims(sku: &str) -> Dims {
    match sku {
        TI2V => Dims::ti2v_5b(),
        D128 => Dims::mini_d128(),
        NANO => Dims::mini_nano(),
        other => panic!("no dims for `{other}`"),
    }
}

fn is_flagship(sku: &str) -> bool {
    sku == TI2V
}

/// The reading codes `(text, denoise)` of a row.
fn codes(sku: &str) -> (Option<u8>, u8) {
    if is_flagship(sku) {
        (Some(0), 1)
    } else {
        (None, 0)
    }
}

fn word(reading: u8, stream: Stream) -> u64 {
    let request = Request::new(4, false).on_stream(stream).in_reading(reading);
    Facts::of(&request).word()
}

/// (a), (b)
#[test]
fn every_row_traces_on_every_platform_with_the_caches_and_seams_it_states() {
    for sku in ROWS {
        for platform in PLATFORMS {
            let plan = trace(sku, platform);
            assert!(!plan.nodes.is_empty(), "{sku} {platform:?}: an empty plan");
            let states = plan
                .caches
                .iter()
                .filter(|row| matches!(row, CacheRow::State { .. }))
                .count();
            let kvs = plan.caches.len() - states;
            assert_eq!(kvs, 0, "{sku} {platform:?}: no kv space anywhere");
            // 32 causal convs in the decoder, 24 in the encoder.
            let want = if is_flagship(sku) { 32 + 24 } else { 0 };
            assert_eq!(
                states, want,
                "{sku} {platform:?}: one frame cache per causal conv of the VAE"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::VELOCITY.name),
                Some(&1),
                "{sku}: one velocity"
            );
            assert!(
                !seams.contains_key(seam::OUT.name),
                "{sku}: a denoiser has no logits, and `out` was planted anyway"
            );
            if is_flagship(sku) {
                assert_eq!(seams.get(seam::HIDDEN.name), Some(&1), "{sku}: one hidden");
                assert_eq!(
                    seams.get(seam::PIXELS.name),
                    Some(&4),
                    "{sku}: the voxel-axis readout on all four VAE arms"
                );
            } else {
                // Of the float readouts, the velocity alone (the `in` seam
                // `trace_hybrid` plants on the embed input is not one).
                for other in [seam::HIDDEN.name, seam::PIXELS.name] {
                    assert!(
                        !seams.contains_key(other),
                        "{sku}: a miniature reads out its velocity alone, got {seams:?}"
                    );
                }
            }
        }
    }
}

fn traced_ports(plan: &Trace) -> BTreeSet<(String, u8, u32)> {
    let mut traced = BTreeSet::new();
    for decl in &plan.values {
        let (kind, port, width) = match &decl.def {
            Def::Input(RuntimeInput::Latents { port, width }) => ("Latents", *port, *width),
            Def::Input(RuntimeInput::Context { port, width }) => ("Context", *port, *width),
            Def::Input(RuntimeInput::LaneVector { port, width }) => ("LaneVector", *port, *width),
            Def::Input(RuntimeInput::AxisPositions { port, axes }) => {
                ("AxisPositions", *port, u32::from(*axes))
            }
            Def::Input(RuntimeInput::Voxels { port, channels }) => ("Voxels", *port, *channels),
            _ => continue,
        };
        traced.insert((kind.to_string(), port, width));
    }
    traced
}

/// (c)
#[test]
fn the_ports_the_trace_reads_are_the_ports_the_facts_declare() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let facts = row(sku).generative.as_ref().expect("facts");
        let mut declared: BTreeSet<(String, u8, u32)> = BTreeSet::new();
        for reading in &facts.readings {
            for (index, port) in reading.ports_indexed() {
                declared.insert((format!("{:?}", port.kind), index, port.width));
            }
        }
        assert_eq!(
            traced_ports(&plan),
            declared,
            "{sku}: the facts and the trace bind one list of ports"
        );

        let d = dims(sku);
        let denoise = facts
            .readings
            .iter()
            .find(|r| r.name == "denoise")
            .unwrap_or_else(|| panic!("{sku} declares no `denoise`"));
        let at = |name: &str| {
            let (index, port) = denoise
                .port(name)
                .unwrap_or_else(|| panic!("{sku}: `denoise` declares no port `{name}`"));
            (index, port.kind, port.width, port.streams.clone())
        };
        assert_eq!(
            at("latents"),
            (
                model::port::LATENTS,
                PortKind::Latents,
                d.patch_in(),
                vec![Stream::Video]
            )
        );
        assert_eq!(
            at("context"),
            (
                model::port::CONTEXT,
                PortKind::Context,
                d.text_dim,
                vec![Stream::Context]
            )
        );
        assert_eq!(
            at("timestep"),
            (
                model::port::TIMESTEP,
                PortKind::LaneVector,
                1,
                vec![Stream::Video]
            )
        );
        assert_eq!(
            at("positions"),
            (
                model::port::POSITIONS,
                PortKind::AxisPositions,
                u32::from(model::ROPE_AXES),
                vec![Stream::Video]
            )
        );
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, d.patch_out());
        // ONE voxel declaration, read under BOTH decoder arms: the two
        // readings state the same `(kind, index)` at the same width, which
        // is what lets the engine seat one rectangle for the whole plan.
        for name in ["vae.decode.head", "vae.decode"] {
            let Some(arm) = facts.readings.iter().find(|r| r.name == name) else {
                assert!(!is_flagship(sku), "{sku}: a VAE row declares `{name}`");
                continue;
            };
            assert_eq!(arm.streams, vec![Stream::Video]);
            assert!(!arm.has_kv && !arm.takes_tokens);
            assert_eq!(arm.readout, ReadoutKind::Pixels);
            assert_eq!(arm.readout_width, model::VAE_RGB);
            assert_eq!(arm.positions, None, "a VAE tile sits in no rotary space");
            let (index, port) = arm.port("latent").expect("the voxel port");
            assert_eq!(
                (index, port.kind, port.width, port.streams.clone()),
                (
                    model::port::VOXELS,
                    PortKind::Voxels,
                    model::VAE_Z,
                    vec![Stream::Video]
                )
            );
        }
        let voxels = plan
            .values
            .iter()
            .filter(|decl| matches!(decl.def, Def::Input(RuntimeInput::Voxels { .. })))
            .count();
        // TWO voxel ports on the flagship: the decode arms' 48-wide latent
        // clip at index 0 and the encode arms' 3-wide pixel clip at index 1
        // (the engine seats one rectangle per `(kind, index)`).
        assert_eq!(
            voxels,
            if is_flagship(sku) { 2 } else { 0 },
            "{sku}: a voxel port per VAE clip width, and none without a VAE"
        );
    }
}

/// (d)
#[test]
fn each_lane_the_facts_list_classifies_into_its_own_class() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let classes = model_dsl::resolve_classes(&plan)
            .unwrap_or_else(|why| panic!("{sku}: a merge does not resolve: {why:?}"));
        let facts = row(sku).generative.as_ref().expect("facts");
        let catalog = row(sku);
        let mut seen: Vec<((&str, Stream), usize)> = Vec::new();
        for reading in &facts.readings {
            for &stream in &reading.streams {
                let request = Request::new(4, false)
                    .on_stream(stream)
                    .in_reading(reading.index);
                let w = (catalog.classify)(&request);
                assert_eq!(w, Facts::of(&request).word(), "{sku} {stream:?}");
                let class = classes
                    .class_of(w & classes.mask)
                    .unwrap_or_else(|| panic!("{sku}: `{}`/{stream:?} has no class", reading.name));
                seen.push(((reading.name, stream), class));
            }
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{sku}: two lanes share a class: {seen:?}"
        );
        // text/Text, denoise/Video, denoise/Context, and one Video lane
        // for each of the two decoder and two encoder arms.
        let want = if is_flagship(sku) { 7 } else { 2 };
        assert_eq!(seen.len(), want, "{sku}: the lanes the facts list");
    }
}

/// (e)
#[test]
fn the_attentions_pair_as_the_architecture_says() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let (text, denoise) = codes(sku);
        let selection_of = |id: ValueId| -> (Selection, &'static str) {
            match &plan.values[id.0 as usize].def {
                Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::GroupIndptr { select },
                    ..
                }) => (*select, "group"),
                Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::LaneIndptr { select },
                    ..
                }) => (*select, "lane"),
                other => panic!("{sku}: a ragged CSR that is not an indptr: {other:?}"),
            }
        };
        let video = word(denoise, Stream::Video);
        let context = word(denoise, Stream::Context);
        let (mut self_paired, mut crossed, mut encoder) = (0usize, 0usize, 0usize);
        for node in &plan.nodes {
            let Operation::Attention(Attention::Ragged {
                q_indptr,
                kv_indptr,
                head_dim,
                sm_scale,
                mask,
                ..
            }) = &node.op
            else {
                continue;
            };
            match mask {
                RaggedMask::RelativeBias { max_len, .. } => {
                    encoder += 1;
                    assert_eq!(*head_dim, model::TE_HEAD_DIM);
                    assert_eq!(*sm_scale, 1.0, "T5 does not scale its logits");
                    assert_eq!(*max_len, model::TE_MAX_TOKENS);
                    let (select, kind) = selection_of(*q_indptr);
                    assert_eq!(kind, "lane", "the encoder attends its own lane");
                    assert_eq!(q_indptr, kv_indptr);
                    let text = text.expect("only the flagship has an encoder");
                    assert!(select.holds(word(text, Stream::Text)));
                    assert!(!select.holds(video));
                }
                RaggedMask::GroupBlockDiagonal => {
                    assert_eq!(*head_dim, d.head_dim);
                    assert!((sm_scale - d.sm_scale()).abs() < 1e-7);
                    let (q_sel, q_kind) = selection_of(*q_indptr);
                    let (kv_sel, kv_kind) = selection_of(*kv_indptr);
                    assert_eq!((q_kind, kv_kind), ("group", "group"));
                    assert!(q_sel.holds(video) && !q_sel.holds(context));
                    if q_indptr == kv_indptr {
                        self_paired += 1;
                    } else {
                        crossed += 1;
                        assert!(kv_sel.holds(context) && !kv_sel.holds(video));
                        // The node spans both arms: video and context are
                        // the two halves of the reading, so the `Or` of
                        // their guards folds to the reading's own arm,
                        // which holds for both lanes and for no other
                        // reading's.
                        let spans = Selection::of(&node.guard).unwrap_or_else(|| {
                            panic!(
                                "{sku}: a cross attention under a guard that is no selection: {:?}",
                                node.guard
                            )
                        });
                        assert!(spans.holds(video) && spans.holds(context));
                        if let Some(text) = text {
                            assert!(!spans.holds(word(text, Stream::Text)));
                        }
                    }
                }
                other => panic!("{sku}: an unexpected mask {other:?}"),
            }
        }
        assert_eq!(
            self_paired, d.layers as usize,
            "{sku}: one self-attention per block"
        );
        assert_eq!(
            crossed, d.layers as usize,
            "{sku}: one cross-attention per block"
        );
        assert_eq!(
            encoder,
            if is_flagship(sku) {
                model::TE_LAYERS as usize
            } else {
                0
            },
            "{sku}: one relative-bias attention per encoder layer"
        );
    }
}

/// (f)
#[test]
fn every_rope_turns_the_rows_three_axis_split_of_the_whole_head_interleaved() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let ropes: Vec<([u32; 4], [f32; 4], RopeForm, u32, u32)> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Elementwise(Elementwise::RopeAxes {
                    dims,
                    thetas,
                    form,
                    rotary_dim,
                    head_dim,
                    ..
                }) => Some((*dims, *thetas, *form, *rotary_dim, *head_dim)),
                _ => None,
            })
            .collect();
        // q and k of every self-attention; the cross-attentions and the
        // encoder turn nothing.
        assert_eq!(ropes.len(), 2 * d.layers as usize, "{sku}");
        let want_dims = match sku {
            NANO => [8, 8, 8, 0],
            _ => [44, 42, 42, 0],
        };
        assert_eq!(d.rope_dims(), want_dims, "{sku}");
        for rope in &ropes {
            assert_eq!(
                *rope,
                (
                    want_dims,
                    [model::ROPE_THETA; 4],
                    RopeForm::Interleaved,
                    d.head_dim,
                    d.head_dim
                ),
                "{sku}"
            );
        }
        let neox = plan.nodes.iter().any(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeFull { .. })
                    | Operation::Elementwise(Elementwise::RopePartial { .. })
            )
        });
        assert!(!neox, "{sku}: umT5 has no rotary embedding");
    }
}

fn budget() -> model_compiler::Budget {
    model_compiler::Budget {
        max_lanes: 64,
        max_tokens: 4096,
        buckets: vec![64, 256, 1024, 4096],
        max_adapters: 0,
    }
}

/// (g)
#[test]
fn every_row_bakes_on_every_platform_under_a_voxel_ladder() {
    for platform in PLATFORMS {
        for sku in ROWS {
            let plan = trace(sku, platform);
            let budgets = model_compiler::Budgets::of(budget())
                .with_voxels(model_compiler::VoxelLadder::new(4096, 4));
            let compiled = model_compiler::compile_axes(
                &plan,
                &budgets,
                &model_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{platform:?}: `{sku}` does not bake: {why}"));
            let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{platform:?} `{sku}`: the regions tile the node list once"
            );
            assert_eq!(
                compiled.voxels.is_some(),
                is_flagship(sku),
                "{platform:?} `{sku}`: a voxel plan iff the row carries a VAE"
            );
        }
    }
    let refused = model_compiler::compile(
        &trace(TI2V, Platform::Cuda),
        &budget(),
        &model_compiler::DeviceProfile::default(),
    );
    assert!(
        matches!(refused, Err(model_compiler::Error::Unsized { .. })),
        "the flagship bakes against no voxel ladder: {refused:?}"
    );
}

/// (h)
#[test]
fn the_generative_facts_state_the_readings_the_latent_and_the_schedule() {
    for sku in ROWS {
        let facts = row(sku).generative.as_ref().expect("facts");
        let d = dims(sku);
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(usize::from(reading.index), at, "{sku}: dense from 0");
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
        if is_flagship(sku) {
            assert_eq!(
                names,
                vec![
                    "text",
                    "denoise",
                    "vae.decode.head",
                    "vae.decode",
                    "vae.encode.head",
                    "vae.encode"
                ]
            );
            let text = &facts.readings[0];
            assert!(
                text.takes_tokens && !text.has_kv,
                "umT5 embeds ids and holds no kv"
            );
            assert_eq!(text.readout, ReadoutKind::Hidden);
            assert_eq!(text.readout_width, model::TE_HIDDEN);
            assert_eq!(text.streams, vec![Stream::Text]);
            assert!(text.ports.is_empty());
        } else {
            assert_eq!(names, vec!["denoise"]);
        }
        let denoise = facts.readings.iter().find(|r| r.name == "denoise").unwrap();
        assert!(!denoise.has_kv && !denoise.takes_tokens);
        assert_eq!(denoise.streams, vec![Stream::Video, Stream::Context]);
        let latent = facts.latent.expect("a latent space");
        assert_eq!(
            (
                latent.channels,
                latent.patch_t,
                latent.patch_h,
                latent.patch_w
            ),
            (d.in_channels, 1, 2, 2)
        );
        assert_eq!(
            (latent.spatial_compression, latent.temporal_compression),
            (16, 4)
        );
        assert_eq!(
            latent.channels * latent.patch_t * latent.patch_h * latent.patch_w,
            d.patch_in()
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.shift, model::SHIFT_TI2V);
        assert_eq!(schedule.train_steps, model::TRAIN_STEPS);
        assert!(schedule.pinned_sigmas.is_empty(), "nothing is pinned");
        assert!(facts.max_rows >= 4096);
        crate::validate(facts);
    }
}

/// What the runtime's `validate_generative` demands, restated here so the
/// facts are checked where they are written.
fn validate(facts: &models::Generative) {
    for reading in &facts.readings {
        assert!(reading.readout_width > 0);
        if !reading.takes_tokens {
            assert!(
                reading.ports.iter().any(|port| matches!(
                    port.kind,
                    PortKind::Latents | PortKind::Voxels | PortKind::Context
                )),
                "a token-less reading states its rows through a row port"
            );
        }
    }
}

/// (i)
#[test]
fn the_modulation_is_a_per_lane_f32_pair_over_a_bf16_trunk() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
        let mut seen = 0usize;
        for node in &plan.nodes {
            let Operation::Elementwise(Elementwise::Modulate {
                x, m, lane_of_row, ..
            }) = &node.op
            else {
                continue;
            };
            seen += 1;
            assert_eq!(
                ty(*x),
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(d.dim))],
                    dtype: Dtype::Bf16,
                },
                "{sku}"
            );
            assert_eq!(
                ty(*m),
                Ty::Tensor {
                    shape: vec![Dim::Lanes, Dim::Const(u64::from(2 * d.dim))],
                    dtype: Dtype::F32,
                },
                "{sku}"
            );
            let lanes = lane_of_row.expect("every modulation here is per lane");
            assert_eq!(
                plan.values[lanes.0 as usize].def,
                Def::Input(RuntimeInput::Geometry {
                    space: 0,
                    kind: GeomKind::RequestOfToken
                }),
                "{sku}: the broadcast is the fire's token→lane table"
            );
        }
        assert_eq!(seen, 2 * d.layers as usize + 1, "{sku}");

        for node in &plan.nodes {
            if !matches!(
                &node.op,
                Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
            ) {
                continue;
            }
            let mut pairs = Vec::new();
            node.op.aliases(&mut pairs);
            assert_eq!(
                pairs.len(),
                1,
                "{sku}: a gated fold is in place on its residual"
            );
        }
        // The two tables are f32 planes added to f32 lane vectors.
        let f32_tables = plan
            .values
            .iter()
            .filter(|decl| {
                matches!(decl.def, Def::Weight(_))
                    && matches!(&decl.ty, Ty::Tensor { dtype: Dtype::F32, shape } if shape.len() == 1)
            })
            .count();
        // The decoder's 38 convs: `post_quant`, `conv_in`, four in the mid
        // block, twenty-four in the twelve resnets plus two shortcuts,
        // two time convs, three resamples, `conv_out`. And the encoder's
        // 30: `conv_in`, sixteen in the eight resnets plus two shortcuts,
        // three resamples, two time convs, four in the mid block,
        // `conv_out`, `quant`.
        const VAE_CONV_BIASES: usize = 38 + 30;
        assert_eq!(
            f32_tables,
            d.layers as usize + 1 + if is_flagship(sku) { VAE_CONV_BIASES } else { 0 },
            "{sku}: one f32 table per block, the head's, and the decoder's f32 conv biases"
        );
    }
}
