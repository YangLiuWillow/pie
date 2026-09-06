//! **THE PARITY HARNESS'S BISECTION KNOB: A TAPPED `mini-dit` PLANTS ITS
//! VELOCITY SEAM ON THE NAMED INTERMEDIATE, AT THAT RECTANGLE'S WIDTH, AND
//! STILL READS EVERY PORT THE READING DECLARES.**
//!
//! ```text
//! cargo test -p models --test the_mini_dit_tap_exports_an_intermediate
//! ```
//!
//! `scripts/imagegen/mini_dit_parity.py --tap <key>` bisects a parity
//! mismatch by reading an intermediate out in the velocity's place
//! (`forward::Tap`). What is asserted:
//!
//! ```text
//! (a) untapped, the velocity seam is `[rows, PATCH_FEATURES]`; tapped at a
//!     trunk rectangle it is `[rows, HIDDEN]`; the row's `Generative` facts
//!     state the same width, so the guest's `velocity(width)` agrees
//! (b) a tap plants ONE float readout and no `out`, like the model itself
//! (c) a tap that ends the trace at block 0 still reads block 2's context
//!     port: the plan declares every port the reading does, whatever the
//!     tap cuts, so the runtime's port feeds do not change under a probe
//! (d) an unknown key taps nothing: the plan is the model's
//! ```

use std::collections::BTreeSet;

use model_dsl::{Def, Dim, Dtype, Platform, RuntimeInput, Trace, Ty, seam, trace_hybrid};
use models::mini_dit::forward::{Tap, generative};
use models::mini_dit::model::{self, Model};

fn traced(tap: Option<&str>) -> Trace {
    let text = Model::mini(Dtype::Bf16, 1).tapped(tap.map(str::to_string));
    trace_hybrid("mini-dit", &text, Platform::Cuda)
}

/// The width of the one velocity seam's value.
fn velocity_width(plan: &Trace) -> u64 {
    let seams: Vec<_> = plan
        .seams
        .iter()
        .filter(|s| s.seam == seam::VELOCITY.name)
        .collect();
    assert_eq!(seams.len(), 1, "one velocity seam, found {seams:?}");
    let value = seams[0].values[0];
    match &plan.values[value.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.as_slice() {
            [Dim::Tokens, Dim::Const(width)] => *width,
            other => panic!("the velocity seam is a `[Tokens, width]` rectangle, not {other:?}"),
        },
        other => panic!("the velocity seam is a tensor, not {other:?}"),
    }
}

fn ports(plan: &Trace) -> BTreeSet<String> {
    plan.values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(RuntimeInput::Latents { port, .. }) => Some(format!("latents[{port}]")),
            Def::Input(RuntimeInput::Context { port, .. }) => Some(format!("context[{port}]")),
            Def::Input(RuntimeInput::LaneVector { port, .. }) => {
                Some(format!("lane_vector[{port}]"))
            }
            Def::Input(RuntimeInput::AxisPositions { port, .. }) => {
                Some(format!("axis_positions[{port}]"))
            }
            _ => None,
        })
        .collect()
}

/// (a)
#[test]
fn a_tap_moves_the_velocity_seam_to_the_intermediate_at_its_width() {
    assert_eq!(
        velocity_width(&traced(None)),
        u64::from(model::PATCH_FEATURES)
    );
    assert_eq!(
        velocity_width(&traced(Some("b0.norm1_out"))),
        u64::from(model::HIDDEN)
    );
    assert_eq!(
        velocity_width(&traced(Some("b2.out"))),
        u64::from(model::HIDDEN)
    );
    assert_eq!(
        velocity_width(&traced(Some("final.norm_out"))),
        u64::from(model::HIDDEN)
    );
    for (tap, width) in [
        (None, model::PATCH_FEATURES),
        (Some("b0.norm1_out"), model::HIDDEN),
        (Some("final.tokens"), model::PATCH_FEATURES),
    ] {
        assert_eq!(Tap::width(tap), width, "Tap::width({tap:?})");
        assert_eq!(
            generative(tap).readings[0].readout_width,
            width,
            "the reading's readout width under {tap:?}"
        );
    }
}

/// (b)
#[test]
fn a_tap_is_still_one_float_readout_and_no_logits() {
    for tap in ["x_embed", "b0.attn_heads", "b1.out_img", "b2.cross_q"] {
        let plan = traced(Some(tap));
        let seams: Vec<&str> = plan.seams.iter().map(|s| s.seam.as_str()).collect();
        assert_eq!(
            seams.iter().filter(|s| **s == seam::VELOCITY.name).count(),
            1,
            "{tap}: one velocity seam; seams are {seams:?}"
        );
        assert!(
            !seams.contains(&seam::OUT.name),
            "{tap}: a probe has no logits either; seams are {seams:?}"
        );
    }
}

/// (c)
#[test]
fn a_tap_that_cuts_the_trace_short_still_reads_every_port() {
    let whole = ports(&traced(None));
    assert!(
        whole.contains("context[1]"),
        "the model reads the context port: {whole:?}"
    );
    for tap in ["x_embed", "b0.in", "b0.norm1_out", "b1.joint_attn_heads"] {
        assert_eq!(
            ports(&traced(Some(tap))),
            whole,
            "{tap}: the ports the plan declares"
        );
    }
}

/// (d)
#[test]
fn an_unknown_key_taps_nothing() {
    let model = traced(None);
    let probe = traced(Some("not.a.dump.key"));
    assert_eq!(probe.nodes.len(), model.nodes.len());
    assert_eq!(velocity_width(&probe), velocity_width(&model));
}
