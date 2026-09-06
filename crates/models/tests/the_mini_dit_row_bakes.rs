//! **THE FIRST GENERATIVE ROW IN THE CATALOG TRACES, CLASSIFIES AND BAKES:
//! THREE STREAMS, ONE GROUP, NO KV, AND A VELOCITY WHERE THE LOGITS WOULD
//! BE.**
//!
//! ```text
//! cargo test -p models --test the_mini_dit_row_bakes
//! ```
//!
//! `mini-dit` is the synthetic family M0 is verified against
//! (`scripts/imagegen/mini_dit_ref.py`). It is the row that says the
//! substrate holds together end to end at the model layer: D2's per-stream
//! lanes and group CSR, D3's float ports and `velocity` readout, D6's
//! modulation and gated folds, D7's three-axis rope. What is asserted:
//!
//! ```text
//! (a) every platform traces, and the row declares NO cache row at all
//! (b) the readout is `velocity` and there is no `out` — a denoiser has no
//!     logits
//! (c) the ports are the five the family declares, at the widths it declares
//! (d) the three streams classify to three distinct classes and every merge
//!     resolves in each
//! (e) the joint attention reads the `!context` group CSR on both sides; the
//!     cross attention reads two different CSRs and spans two arms
//! (f) the rope states three axes summing to the head, interleaved
//! (g) the plan bakes on every platform, and the image class is the one the
//!     velocity export lives in
//! (h) the row's `Generative` facts say what the trace actually reads: one
//!     reading, three streams, five ports whose kind-relative indices are
//!     the ones the trace numbers, and a velocity readout at the plan's
//!     own width
//! ```

use std::collections::BTreeSet;

use model_dsl::{
    Attention, Classify, Def, Dim, Dtype, Elementwise, GeomKind, Guard, Operands, Operation,
    Platform, Request, RopeForm, RuntimeInput, Stream, Trace, Ty, ValueId, seam,
};
use models::mini_dit::forward::Facts;
use models::mini_dit::model;
use models::{PortKind, ReadoutKind};

const SKU: &str = "mini-dit-bf16-kv-bf16";

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn trace(platform: Platform) -> Trace {
    let row = models::sku(SKU).unwrap_or_else(|| {
        let names: Vec<&str> = models::skus().map(|row| row.name.as_str()).collect();
        panic!("this build ships no `{SKU}`; rows are {names:#?}")
    });
    (row.trace)(platform)
}

/// (a), (b)
#[test]
fn the_row_traces_on_every_platform_with_no_cache_and_a_velocity_readout() {
    for platform in PLATFORMS {
        let plan = trace(platform);
        assert!(!plan.nodes.is_empty(), "{platform:?}: an empty plan");
        assert!(
            plan.caches.is_empty(),
            "{platform:?}: a denoise reading holds nothing between fires, and this row \
             declared {:?}",
            plan.caches,
        );
        let seams: BTreeSet<&str> = plan.seams.iter().map(|s| s.seam.as_str()).collect();
        assert!(
            seams.contains(seam::VELOCITY.name),
            "{platform:?}: no velocity readout; seams are {seams:?}"
        );
        assert!(
            !seams.contains(seam::OUT.name),
            "{platform:?}: a denoiser has no logits, and `out` was planted anyway"
        );
    }
}

/// (c) — the five float ports, each once, at the widths `model::port` names.
#[test]
fn the_row_reads_exactly_the_five_ports_it_declares() {
    let plan = trace(Platform::Cuda);
    let ports: BTreeSet<String> = plan
        .values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(input) => match input {
                RuntimeInput::Latents { port, width } => Some(format!("latents[{port}]:{width}")),
                RuntimeInput::Context { port, width } => Some(format!("context[{port}]:{width}")),
                RuntimeInput::LaneVector { port, width } => {
                    Some(format!("lane_vector[{port}]:{width}"))
                }
                RuntimeInput::AxisPositions { port, axes } => {
                    Some(format!("axis_positions[{port}]:{axes}"))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    let want: BTreeSet<String> = [
        format!(
            "latents[{}]:{}",
            model::port::LATENTS,
            model::PATCH_FEATURES
        ),
        format!("context[{}]:{}", model::port::TEXT, model::TEXT_WIDTH),
        format!("context[{}]:{}", model::port::CONTEXT, model::CONTEXT_WIDTH),
        format!("lane_vector[{}]:1", model::port::TIMESTEP),
        format!(
            "axis_positions[{}]:{}",
            model::port::POSITIONS,
            model::ROPE_AXES
        ),
    ]
    .into_iter()
    .collect();
    assert_eq!(ports, want);

    // The timestep is embedded once and only once: `[Lanes, 1]` f32 into
    // `[Lanes, TIMESTEP_DIM]` f32, `[sin | cos]`, unscaled.
    let sinusoids: Vec<(u32, f32, bool, f32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::Sinusoid {
                dim,
                max_period,
                flip_sin_cos,
                scale,
                ..
            }) => Some((*dim, *max_period, *flip_sin_cos, *scale)),
            _ => None,
        })
        .collect();
    assert_eq!(
        sinusoids,
        vec![(
            model::TIMESTEP_DIM,
            model::TIMESTEP_MAX_PERIOD,
            model::TIMESTEP_FLIP_SIN_COS,
            model::TIMESTEP_SCALE
        )]
    );
}

/// (d) — three streams, three classes, every merge resolved in each.
#[test]
fn each_stream_classifies_into_its_own_class_and_every_merge_resolves() {
    let plan = trace(Platform::Cuda);
    let classes = model_dsl::resolve_classes(&plan).expect("every merge resolves");
    let row = models::sku(SKU).expect("the row is in the catalog");

    let mut seen = Vec::new();
    for stream in [Stream::Text, Stream::Image, Stream::Context] {
        let request = Request::new(1, false).on_stream(stream);
        // The catalog's own classify column and the family's `Facts` agree.
        let word = (row.classify)(&request);
        assert_eq!(word, Facts::of(&request).word(), "{stream:?}");
        let class = classes
            .class_of(word & classes.mask)
            .unwrap_or_else(|| panic!("a {stream:?} lane has no class"));
        seen.push((stream, class));
    }
    let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
    assert_eq!(
        distinct.len(),
        3,
        "the three streams share a class: {seen:?}"
    );
}

/// (e) — the two ragged reads, and which tables each one pairs its segments
/// by.
#[test]
fn the_joint_attention_is_self_paired_and_the_cross_attention_is_not() {
    let plan = trace(Platform::Cuda);
    let ragged: Vec<(ValueId, ValueId, u32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::Ragged {
                q_indptr,
                kv_indptr,
                head_dim,
                ..
            }) => Some((*q_indptr, *kv_indptr, *head_dim)),
            _ => None,
        })
        .collect();
    // Three joint self-attentions (blocks 0, 1, 2) and one cross-attention.
    assert_eq!(ragged.len(), 4, "one ragged read per attention sublayer");
    assert!(
        ragged
            .iter()
            .all(|(_, _, head_dim)| *head_dim == model::HEAD_DIM),
        "every head is {} wide",
        model::HEAD_DIM
    );
    let self_paired: Vec<&(ValueId, ValueId, u32)> =
        ragged.iter().filter(|(q, kv, _)| q == kv).collect();
    let crossed: Vec<&(ValueId, ValueId, u32)> =
        ragged.iter().filter(|(q, kv, _)| q != kv).collect();
    assert_eq!(self_paired.len(), 3);
    assert_eq!(
        crossed.len(),
        1,
        "one attention whose keys are another lane's"
    );

    // The joint blocks' CSR is the whole `!context` selection — the caption
    // and image lanes, packed as one sequence.
    let selection_of = |id: ValueId| match &plan.values[id.0 as usize].def {
        Def::Input(RuntimeInput::Geometry {
            kind: GeomKind::GroupIndptr { select },
            ..
        }) => *select,
        other => panic!("a ragged CSR that is not a group indptr: {other:?}"),
    };
    let joint = selection_of(self_paired[0].0);
    assert_eq!(
        selection_of(self_paired[1].0),
        joint,
        "blocks 0 and 1 pack one sequence"
    );
    // A lane's word satisfies a selection iff `word & mask == value`; the
    // joint selection is exactly "not on the context stream".
    let word = |stream: Stream| Facts::of(&Request::new(1, false).on_stream(stream)).word();
    assert!(joint.holds(word(Stream::Text)));
    assert!(joint.holds(word(Stream::Image)));
    assert!(!joint.holds(word(Stream::Context)));

    // The cross-attention's keys come off the context lane and its queries
    // off the image lane, which is exactly the two-arm join `Recorder::push`
    // makes an exception for.
    let (q_csr, kv_csr, _) = crossed[0];
    assert!(selection_of(*q_csr).holds(word(Stream::Image)));
    assert!(!selection_of(*q_csr).holds(word(Stream::Context)));
    assert!(selection_of(*kv_csr).holds(word(Stream::Context)));
    assert!(!selection_of(*kv_csr).holds(word(Stream::Image)));

    let joins = plan
        .nodes
        .iter()
        .find(|node| {
            matches!(
                &node.op,
                Operation::Attention(Attention::Ragged { q_indptr, kv_indptr, .. })
                    if q_indptr != kv_indptr
            )
        })
        .expect("the cross attention");
    assert!(
        matches!(joins.guard, Guard::Or(..)),
        "a cross attention spans both arms: {:?}",
        joins.guard
    );
}

/// (f) — the rope's numbers, stated once by the family and read back here.
#[test]
fn every_rope_turns_three_axes_of_the_whole_head_interleaved() {
    let plan = trace(Platform::Cuda);
    let ropes: Vec<([u32; 4], RopeForm, u32, u32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::RopeAxes {
                dims,
                form,
                rotary_dim,
                head_dim,
                ..
            }) => Some((*dims, *form, *rotary_dim, *head_dim)),
            _ => None,
        })
        .collect();
    // Two per joint self-attention sublayer (q and k), over blocks 0 and 2 as
    // one stream and block 1 as two.
    assert_eq!(ropes.len(), 8);
    for rope in &ropes {
        assert_eq!(
            *rope,
            (
                model::ROPE_DIMS,
                RopeForm::Interleaved,
                model::HEAD_DIM,
                model::HEAD_DIM
            )
        );
    }
    assert_eq!(
        model::ROPE_DIMS.iter().sum::<u32>(),
        model::HEAD_DIM,
        "the three axes cover the whole head"
    );

    // The cross-attention gets no rope (the Wan contract), so the rope count
    // above is exactly the self-attention count and not one more.
    let ropes_after_cross = plan
        .nodes
        .iter()
        .skip_while(|node| {
            !matches!(
                &node.op,
                Operation::Elementwise(Elementwise::Layernorm { .. })
            )
        })
        .filter(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeAxes { .. })
            )
        })
        .count();
    assert_eq!(
        ropes_after_cross, 0,
        "the affine cross norm is followed by no rope"
    );
}

/// (g) — the plan bakes, on every platform.
#[test]
fn the_plan_bakes_on_every_platform() {
    for platform in PLATFORMS {
        let plan = trace(platform);
        let budget = model_compiler::Budget {
            max_lanes: 256,
            max_tokens: 8192,
            buckets: vec![
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
            ],
            max_adapters: 0,
        };
        let compiled =
            model_compiler::compile(&plan, &budget, &model_compiler::DeviceProfile::default())
                .unwrap_or_else(|why| {
                    panic!("{platform:?}: the mini-dit plan does not bake: {why}")
                });
        assert!(
            !compiled.regions.is_empty(),
            "{platform:?}: a bake with no regions"
        );
        // Every node the trace states is tiled into exactly one region.
        let tiled: usize = compiled
            .regions
            .iter()
            .map(|region| region.nodes.len())
            .sum();
        assert_eq!(
            tiled,
            plan.nodes.len(),
            "{platform:?}: the regions tile the node list once"
        );
    }
}

/// The types the joint rectangle carries, read off the trace rather than
/// assumed: `[Tokens, HIDDEN]` bf16 everywhere in the trunk, and a modulation
/// vector that is `[Lanes, 2 · HIDDEN]` f32 because a lane-vector chain lands
/// f32 (`IMAGEGEN_CONTRACT.md` §3).
#[test]
fn the_modulation_is_a_per_lane_f32_pair_over_a_bf16_trunk() {
    let plan = trace(Platform::Cuda);
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
                shape: vec![Dim::Tokens, Dim::Const(u64::from(model::HIDDEN))],
                dtype: Dtype::Bf16,
            }
        );
        assert_eq!(
            ty(*m),
            Ty::Tensor {
                shape: vec![Dim::Lanes, Dim::Const(u64::from(2 * model::HIDDEN))],
                dtype: Dtype::F32,
            }
        );
        let lanes = lane_of_row.expect("every modulation here is per lane");
        assert_eq!(
            plan.values[lanes.0 as usize].def,
            Def::Input(RuntimeInput::Geometry {
                space: 0,
                kind: GeomKind::RequestOfToken
            }),
            "the broadcast is the fire's token→lane table"
        );
    }
    // Two per block (attention, MLP) over three blocks, plus the head; block
    // 1 states its two per stream.
    assert_eq!(seen, 9);

    // And every gated fold aliases the stream it folds into.
    for node in &plan.nodes {
        if !matches!(
            &node.op,
            Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
        ) {
            continue;
        }
        let mut pairs = Vec::new();
        node.op.aliases(&mut pairs);
        assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
    }
}

/// (h) — the guest-facing facts and the trace are one statement.
///
/// The runtime resolves `input("latents")` to `RuntimeInput::Latents { port }`
/// by counting ports of that kind in declaration order, so a reordered
/// `ports` list would feed the caption rows into the cross-attention and
/// still typecheck. This is what says it does not.
#[test]
fn the_generative_facts_are_the_ports_the_trace_reads() {
    let row = models::sku(SKU).expect("the row is in the catalog");
    let facts = row
        .generative
        .as_ref()
        .expect("the first generative row states its readings");
    assert_eq!(facts.readings.len(), 1, "one reading, today");
    let reading = &facts.readings[0];
    assert_eq!(reading.name, "denoise");
    assert_eq!(reading.index, models::mini_dit::forward::DENOISE_READING);
    assert!(!reading.has_kv, "a denoise pass binds no kv");
    assert!(!reading.takes_tokens, "and embeds no tokens");
    assert_eq!(
        reading.streams,
        vec![Stream::Text, Stream::Image, Stream::Context]
    );
    assert_eq!(reading.readout, ReadoutKind::Velocity);
    assert_eq!(reading.readout_width, model::PATCH_FEATURES);

    // The port a name resolves to, as the host resolves it: the index among
    // ports of the same kind, in declaration order.
    let at = |name: &str| {
        let (index, port) = reading.port(name).unwrap_or_else(|| {
            panic!("the reading declares no port `{name}`");
        });
        (index, port.kind, port.width)
    };
    assert_eq!(
        at("latents"),
        (
            model::port::LATENTS,
            PortKind::Latents,
            model::PATCH_FEATURES
        )
    );
    assert_eq!(
        at("text"),
        (model::port::TEXT, PortKind::Context, model::TEXT_WIDTH)
    );
    assert_eq!(
        at("context"),
        (
            model::port::CONTEXT,
            PortKind::Context,
            model::CONTEXT_WIDTH
        )
    );
    assert_eq!(
        at("timestep"),
        (model::port::TIMESTEP, PortKind::LaneVector, 1)
    );
    assert_eq!(
        at("positions"),
        (
            model::port::POSITIONS,
            PortKind::AxisPositions,
            u32::from(model::ROPE_AXES)
        )
    );

    // And the same five ports, at the same indices, are what the trace binds.
    let plan = trace(Platform::Cuda);
    let mut traced: Vec<(PortKind, u8, u32)> = plan
        .values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(RuntimeInput::Latents { port, width }) => {
                Some((PortKind::Latents, *port, *width))
            }
            Def::Input(RuntimeInput::Context { port, width }) => {
                Some((PortKind::Context, *port, *width))
            }
            Def::Input(RuntimeInput::LaneVector { port, width }) => {
                Some((PortKind::LaneVector, *port, *width))
            }
            Def::Input(RuntimeInput::AxisPositions { port, axes }) => {
                Some((PortKind::AxisPositions, *port, u32::from(*axes)))
            }
            _ => None,
        })
        .collect();
    traced.sort_by_key(|(kind, port, _)| (format!("{kind:?}"), *port));
    let mut declared: Vec<(PortKind, u8, u32)> = reading
        .ports_indexed()
        .map(|(index, port)| (port.kind, index, port.width))
        .collect();
    declared.sort_by_key(|(kind, port, _)| (format!("{kind:?}"), *port));
    assert_eq!(
        traced, declared,
        "the facts a guest binds by and the inputs the plan reads are one list"
    );

    // The latent space and the schedule the reference states.
    let latent = facts.latent.expect("a denoiser states its latent space");
    assert_eq!(
        (
            latent.channels,
            latent.patch_h,
            latent.patch_w,
            latent.patch_t
        ),
        (model::CHANNELS, model::PATCH, model::PATCH, 1)
    );
    assert_eq!(
        latent.channels * latent.patch_t * latent.patch_h * latent.patch_w,
        model::PATCH_FEATURES,
        "a latent row is exactly the head's output row"
    );
    let schedule = facts.schedule.as_ref().expect("and its schedule");
    assert_eq!(schedule.kind, models::ScheduleKind::Flow);
    assert_eq!(schedule.pinned_sigmas, vec![1.0, 0.75, 0.5, 0.25]);
}
