//! **THE LTX-2.5 ROWS TRACE, CLASSIFY AND BAKE: THREE ARMS UNDER ONE PLAN,
//! FOUR STREAMS IN ONE FIRE, SIX ATTENTIONS A BLOCK, AND A ROTARY LADDER
//! THAT RUNS ACROSS THE ROW RATHER THAN DOWN A HEAD.**
//!
//! ```text
//! cargo test -p models --test the_ltx_2_rows_bake
//! ```
//!
//! `ltx25` is the first catalog row whose denoise reading carries FOUR
//! streams of two different widths in one attention group — a video lane at
//! 4096, an audio lane at 2048, and the two text contexts each keyed to its
//! own modality — and the first whose rope is
//! [`RopeForm::SplitLadder`](model_dsl::RopeForm::SplitLadder), one
//! frequency ladder across the whole row with the axes handed out
//! round-robin along it. `ltx25-mini` is the parity fixture
//! `scripts/imagegen/ltx2_golden.py --mini` writes. What is asserted:
//!
//! ```text
//! (a) every row traces on every platform, holding no kv space and no state
//! (b) the seams are one `velocity` (merged over the two modalities) and
//!     two `hidden` (one per connector), and never `out`
//! (c) the ports the trace reads are the ports the facts declare, at the
//!     facts' widths and kind-relative indices, and the named ports resolve
//!     to `model::port`
//! (d) the (reading, stream) lanes the facts list classify into distinct
//!     classes where every merge resolves
//! (e) six ragged reads per block — the two self-attentions over their own
//!     stream's group CSR, the two prompt cross-attentions into the two
//!     context selections, and the cross-modal pair, each reading the OTHER
//!     stream's CSR — plus one per connector layer over its lane CSR
//! (f) every rope is the one-ladder form at theta 1e4: the video rows' three
//!     axes over `dim/6` frequencies each with a two-slot identity pad, the
//!     audio rows' and the cross-modal pair's one axis with none; the text
//!     cross-attentions turn nothing
//! (g) every row bakes on every platform
//! (h) the generative facts: three readings dense from 0, the latent space
//!     (128, 1x1x1, /32, /8), and the eight pinned distilled sigmas
//! (i) the modulation is a per-lane f32 vector over a bf16 trunk — twelve
//!     scale-shift sites a block and one per head — every gated fold aliases
//!     its residual, and every attention ends in a per-head sigmoid gate
//! (j) every per-block table is folded into a COPY of the vector its stream's
//!     adaLN head hands the whole stack, never in place on the vector itself
//! ```

use std::collections::{BTreeMap, BTreeSet};

use model_dsl::{
    Attention, Classify, Def, Dim, Dtype, Elementwise, GeomKind, Operands, Operation, Platform,
    RaggedMask, Request, RopeForm, RuntimeInput, Selection, Stream, Trace, Ty, ValueId, seam,
};
use models::ltx_2::forward::{DENOISE, Facts, REFINE_AUDIO, REFINE_VIDEO};
use models::ltx_2::model::{self, Dims};
use models::{PortKind, ReadoutKind, ScheduleKind};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const MINI: &str = "ltx25-mini-bf16-kv-bf16";
const ROWS: [&str; 2] = [FLAGSHIP, MINI];

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
        FLAGSHIP => Dims::ltx_2_5(),
        MINI => Dims::mini(),
        other => panic!("no dims for `{other}`"),
    }
}

fn word(reading: u8, stream: Stream) -> u64 {
    let request = Request::new(4, false).on_stream(stream).in_reading(reading);
    Facts::of(&request).word()
}

/// (a), (b)
#[test]
fn every_row_traces_on_every_platform_holding_nothing_between_fires() {
    for sku in ROWS {
        for platform in PLATFORMS {
            let plan = trace(sku, platform);
            assert!(!plan.nodes.is_empty(), "{sku} {platform:?}: an empty plan");
            assert!(
                plan.caches.is_empty(),
                "{sku} {platform:?}: a denoiser and two connectors hold nothing between fires"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::VELOCITY.name),
                Some(&1),
                "{sku}: one velocity, planted on the merge of the two modalities"
            );
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&2),
                "{sku}: one hidden per connector"
            );
            assert!(
                !seams.contains_key(seam::OUT.name),
                "{sku}: a denoiser has no logits, and `out` was planted anyway"
            );
            assert!(
                !seams.contains_key(seam::PIXELS.name),
                "{sku}: this text declares no VAE arm yet"
            );
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
                d.channels,
                vec![Stream::Video, Stream::Audio]
            ),
            "{sku}: one latent rectangle for the two modalities"
        );
        assert_eq!(
            at("context"),
            (
                model::port::CONTEXT,
                PortKind::Context,
                d.cross_dim,
                vec![Stream::Context]
            )
        );
        assert_eq!(
            at("audio_context"),
            (
                model::port::AUDIO_CONTEXT,
                PortKind::Context,
                d.audio_cross_dim,
                vec![Stream::Reference]
            )
        );
        // EVERY lane binds a timestep: the two context lanes modulate their
        // own rows from the prompt timestep.
        assert_eq!(
            at("timestep"),
            (model::port::TIMESTEP, PortKind::LaneVector, 1, vec![])
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
        assert_eq!(
            at("audio_positions"),
            (
                model::port::TIME_POSITIONS,
                PortKind::AxisPositions,
                1,
                vec![Stream::Audio]
            )
        );
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, d.channels);
        // The two connectors read ONE packed-text rectangle at one index.
        for name in ["refine.video", "refine.audio"] {
            let refine = facts.readings.iter().find(|r| r.name == name).unwrap();
            let (index, port) = refine.port("text").expect("the packed trunk rows");
            assert_eq!(
                (index, port.kind, port.width),
                (model::port::TEXT, PortKind::Latents, d.text_in()),
                "{sku} {name}: a token-less reading states its rows through a latents port"
            );
            assert_eq!(refine.readout, ReadoutKind::Hidden);
        }
        assert_eq!(
            facts
                .readings
                .iter()
                .find(|r| r.name == "refine.video")
                .unwrap()
                .readout_width,
            d.cross_dim
        );
        assert_eq!(
            facts
                .readings
                .iter()
                .find(|r| r.name == "refine.audio")
                .unwrap()
                .readout_width,
            d.audio_cross_dim
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
        // Four denoise lanes and one text lane per connector.
        assert_eq!(seen.len(), 6, "{sku}: the lanes the facts list");
    }
}

/// (e)
#[test]
fn the_attentions_pair_as_the_architecture_says() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
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
        let video = word(DENOISE, Stream::Video);
        let audio = word(DENOISE, Stream::Audio);
        let ctx = word(DENOISE, Stream::Context);
        let actx = word(DENOISE, Stream::Reference);
        let refine_v = word(REFINE_VIDEO, Stream::Text);
        let refine_a = word(REFINE_AUDIO, Stream::Text);

        // (query stream, key stream) -> how many reads pair them.
        let mut pairs: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut connector_reads = 0usize;
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
            assert_eq!(*mask, RaggedMask::GroupBlockDiagonal, "{sku}");
            let (q_sel, q_kind) = selection_of(*q_indptr);
            let (kv_sel, _) = selection_of(*kv_indptr);
            if q_kind == "lane" {
                // A connector lane attends its own rows.
                connector_reads += 1;
                assert_eq!(q_indptr, kv_indptr, "{sku}: a connector reads itself");
                assert!(q_sel.holds(refine_v) || q_sel.holds(refine_a));
                assert!(!q_sel.holds(video) && !q_sel.holds(audio));
                continue;
            }
            let name = |sel: &Selection| -> &'static str {
                match (
                    sel.holds(video),
                    sel.holds(audio),
                    sel.holds(ctx),
                    sel.holds(actx),
                ) {
                    (true, false, false, false) => "video",
                    (false, true, false, false) => "audio",
                    (false, false, true, false) => "context",
                    (false, false, false, true) => "audio_context",
                    other => panic!("{sku}: a selection over {other:?}"),
                }
            };
            let (q, kv) = (name(&q_sel), name(&kv_sel));
            let want_head = match q {
                "video" if kv == "video" || kv == "context" => d.head_dim,
                _ => d.audio_head_dim,
            };
            assert_eq!(*head_dim, want_head, "{sku}: {q} -> {kv}");
            let want_scale = (want_head as f32).sqrt().recip();
            assert!((sm_scale - want_scale).abs() < 1e-7, "{sku}: {q} -> {kv}");
            *pairs.entry((q, kv)).or_default() += 1;
        }
        let layers = d.layers as usize;
        let want: BTreeMap<(&str, &str), usize> = [
            (("video", "video"), layers),
            (("audio", "audio"), layers),
            (("video", "context"), layers),
            (("audio", "audio_context"), layers),
            (("video", "audio"), layers),
            (("audio", "video"), layers),
        ]
        .into_iter()
        .collect();
        assert_eq!(pairs, want, "{sku}: six attentions a block");
        assert_eq!(
            connector_reads,
            2 * d.conn_layers as usize,
            "{sku}: one read per connector layer, twice over"
        );
    }
}

/// (f)
#[test]
fn every_rope_is_one_ladder_across_the_row() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let mut seen: BTreeMap<([u32; 4], u32), usize> = BTreeMap::new();
        for node in &plan.nodes {
            let Operation::Elementwise(Elementwise::RopeAxes {
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                ..
            }) = &node.op
            else {
                continue;
            };
            assert_eq!(*form, RopeForm::SplitLadder, "{sku}");
            assert_eq!(*thetas, [model::ROPE_THETA; 4], "{sku}");
            assert_eq!(
                rotary_dim, head_dim,
                "{sku}: the ladder pairs rotate-half within a whole head"
            );
            *seen.entry((*dims, *head_dim)).or_default() += 1;
        }
        let layers = d.layers as usize;
        let conn = d.conn_layers as usize;
        // Per block: the two self-attentions and the cross-modal pair, q and
        // k apiece. Per connector layer: q and k. The text cross-attentions
        // turn nothing.
        let mut want: BTreeMap<([u32; 4], u32), usize> = BTreeMap::new();
        *want.entry((d.rope_dims(), d.head_dim)).or_default() += 2 * layers;
        *want
            .entry((d.audio_rope_dims(), d.audio_head_dim))
            .or_default() += 2 * layers;
        *want
            .entry((d.av_rope_dims(), d.audio_head_dim))
            .or_default() += 4 * layers;
        *want
            .entry(([d.cross_dim, 0, 0, 0], d.head_dim))
            .or_default() += 2 * conn;
        *want
            .entry(([d.audio_cross_dim, 0, 0, 0], d.audio_head_dim))
            .or_default() += 2 * conn;
        assert_eq!(seen, want, "{sku}");

        // The video ladder pads two slots; a one-axis ladder pads none.
        assert_eq!(model::rope_pad(d.dim(), model::ROPE_AXES), 2, "{sku}");
        assert_eq!(
            model::rope_pad(d.audio_dim(), model::AUDIO_ROPE_AXES),
            0,
            "{sku}"
        );
        let no_other = plan.nodes.iter().any(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeFull { .. })
                    | Operation::Elementwise(Elementwise::RopePartial { .. })
                    | Operation::Elementwise(Elementwise::RopeMrope { .. })
            )
        });
        assert!(!no_other, "{sku}: this text turns one kind of rope");
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
fn every_row_bakes_on_every_platform() {
    for platform in PLATFORMS {
        for sku in ROWS {
            let plan = trace(sku, platform);
            let compiled = model_compiler::compile(
                &plan,
                &budget(),
                &model_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{platform:?}: `{sku}` does not bake: {why}"));
            let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{platform:?} `{sku}`: the regions tile the node list once"
            );
        }
    }
}

/// (h)
#[test]
fn the_generative_facts_state_the_readings_the_latent_and_the_schedule() {
    for sku in ROWS {
        let facts = row(sku).generative.as_ref().expect("facts");
        let d = dims(sku);
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(usize::from(reading.index), at, "{sku}: dense from 0");
            assert!(
                !reading.has_kv && !reading.takes_tokens,
                "{sku}: every reading here binds float ports alone"
            );
            assert!(
                reading.positions.is_none(),
                "{sku}: LTX turns physical coordinates, which no convention names"
            );
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["denoise", "refine.video", "refine.audio"]);
        assert_eq!(usize::from(DENOISE), 0);
        assert_eq!(usize::from(REFINE_VIDEO), 1);
        assert_eq!(usize::from(REFINE_AUDIO), 2);
        let latent = facts.latent.expect("a latent space");
        assert_eq!(
            (
                latent.channels,
                latent.patch_t,
                latent.patch_h,
                latent.patch_w
            ),
            (d.channels, 1, 1, 1),
            "{sku}: a token is one latent cell"
        );
        assert_eq!(
            (latent.spatial_compression, latent.temporal_compression),
            (32, 8)
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.train_steps, model::TRAIN_STEPS);
        assert_eq!(schedule.boundary, None, "one backbone");
        assert_eq!(
            schedule.pinned_sigmas,
            model::DISTILLED_SIGMAS.to_vec(),
            "{sku}: the distilled row pins eight sigmas for BOTH modalities"
        );
        assert!(facts.max_rows >= 4096);
        validate(facts);
    }
}

/// What the runtime's `validate_generative` demands, restated here so the
/// facts are checked where they are written.
fn validate(facts: &models::Generative) {
    for reading in &facts.readings {
        assert!(reading.readout_width > 0);
        if !reading.takes_tokens {
            assert!(
                reading
                    .ports
                    .iter()
                    .any(|port| port.kind == PortKind::Latents),
                "a token-less reading states its rows through a latents port"
            );
        }
    }
}

/// (j)
///
/// `elementwise.add_bias` folds ITS BIAS IN PLACE (the IR aliases `out_out`
/// onto `out`). One adaLN head serves all 48 blocks, so a block that added
/// its own table straight onto that shared vector would hand block `n + 1`
/// the sum of every table before it. The text copies first; this is the
/// claim that it still does.
#[test]
fn every_block_table_folds_into_a_copy_of_the_vector_the_stack_shares() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let lane_shaped = |id: ValueId| {
            matches!(
                &plan.values[id.0 as usize].ty,
                Ty::Tensor { shape, .. } if shape.first() == Some(&Dim::Lanes)
            )
        };
        let mut folded = 0usize;
        for node in &plan.nodes {
            let Operation::Elementwise(Elementwise::AddBias { out, .. }) = &node.op else {
                continue;
            };
            if !lane_shaped(*out) {
                continue;
            }
            // How many nodes read the rectangle this fold writes over?
            let readers = plan
                .nodes
                .iter()
                .filter(|other| {
                    let mut ins = Vec::new();
                    other.op.inputs(&mut ins);
                    ins.contains(out)
                })
                .count();
            assert_eq!(
                readers,
                1,
                "{sku}: a bias folded in place on a lane vector {} other nodes also read",
                readers - 1
            );
            folded += 1;
        }
        let d = dims(sku);
        // Per block: four tables a stream. Outside them: the two heads' own
        // `[temb | temb]` folds and the eight adaLN projections' biases.
        assert!(
            folded >= 8 * d.layers as usize,
            "{sku}: {folded} lane-vector folds for {} blocks",
            d.layers
        );
    }
}

/// (i)
#[test]
fn the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
        let mut modulates = 0usize;
        for node in &plan.nodes {
            let Operation::Elementwise(Elementwise::Modulate {
                x, m, lane_of_row, ..
            }) = &node.op
            else {
                continue;
            };
            modulates += 1;
            let width = match ty(*x) {
                Ty::Tensor { shape, dtype } => {
                    assert_eq!(dtype, Dtype::Bf16, "{sku}: a bf16 trunk");
                    assert_eq!(shape[0], Dim::Tokens, "{sku}");
                    match shape[1] {
                        Dim::Const(w) => w,
                        other => panic!("{sku}: a modulated row of width {other:?}"),
                    }
                }
                other => panic!("{sku}: {other:?}"),
            };
            assert_eq!(
                ty(*m),
                Ty::Tensor {
                    shape: vec![Dim::Lanes, Dim::Const(2 * width)],
                    dtype: Dtype::F32,
                },
                "{sku}: a per-lane f32 scale/shift pair as wide as its rows"
            );
            let lanes = lane_of_row.expect("every modulation here is per lane");
            assert_eq!(
                plan.values[lanes.0 as usize].def,
                Def::Input(RuntimeInput::Geometry {
                    space: 0,
                    kind: GeomKind::RequestOfToken
                }),
                "{sku}: the broadcast is the fire's token->lane table"
            );
        }
        // Twelve sites a block (three per stream, one per context lane, two
        // per stream for the cross-modal pair) and one per head.
        assert_eq!(modulates, 12 * d.layers as usize + 2, "{sku}");

        let mut gates = 0usize;
        for node in &plan.nodes {
            match &node.op {
                Operation::Elementwise(Elementwise::GatedResidualAdd { .. }) => {
                    let mut pairs = Vec::new();
                    node.op.aliases(&mut pairs);
                    assert_eq!(
                        pairs.len(),
                        1,
                        "{sku}: a gated fold is in place on its residual"
                    );
                }
                Operation::Elementwise(Elementwise::GateSigmoidMulHeads {
                    head_dim,
                    scale,
                    ..
                }) => {
                    gates += 1;
                    assert_eq!(*scale, model::GATE_SCALE, "{sku}: `out * 2 sigmoid(W x)`");
                    assert!(
                        *head_dim == d.head_dim || *head_dim == d.audio_head_dim,
                        "{sku}: a gate logit per head"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            gates,
            6 * d.layers as usize + 2 * d.conn_layers as usize,
            "{sku}: every attention ends in a per-head gate"
        );
    }
}
