//! **A PER-LANE MODULATION VECTOR REACHES EVERY ROW OF ITS LANE THROUGH THE
//! FIRE'S OWN TOKEN→LANE MAP, KV SPACE OR NO KV SPACE.**
//!
//! ```text
//! cargo test -p model-dsl --test a_modulate_over_lanes_broadcasts_by_request_of_token
//! ```
//!
//! adaLN (D6) computes `[Lanes, k·width]` off a per-lane timestep and
//! applies it to `[Tokens, width]` rows. The broadcast is not a shape-algebra
//! operation — the IR has none — it is the op reading `request_of_token`.
//! A denoiser declares no kv space, so that table must be readable without
//! one:
//!
//! ```text
//! (a) the timestep chain types as stated: lane_vector [Lanes, 1] f32 →
//!     sinusoid [Lanes, dim] f32 → matmul [Lanes, 2·width]
//! (b) modulate over it names `lane_of_row` = RequestOfToken in space 0,
//!     on a plan with no cache row at all, and answers x's own type
//! (c) the per-token form (Wan TI2V) names no lane map and wants m on x's rows
//! (d) the gated residual fold aliases the stream it folds into
//! (e) a vector of the wrong width is refused at trace time, by width
//! ```

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Request, Value,
    Weight, ops, seam, trace_hybrid,
};
use model_ir::{Def, Dim, Elementwise, GeomKind, Operation, RuntimeInput, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

const WIDTH: u32 = 64;
const FREQ: u32 = 32;

/// Which broadcast the text under test states.
#[derive(Clone, Copy)]
enum By {
    Lane,
    Token,
}

struct AdaLn(By);

impl ForwardHybrid for AdaLn {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let ada = Weight::sym("ada", [2 * u64::from(WIDTH), u64::from(FREQ)], Dtype::Bf16);
        let gate = Weight::sym("gate", [u64::from(WIDTH), u64::from(FREQ)], Dtype::Bf16);
        let normed = ops::elemwise::layernorm_no_scale(&x, 1e-6);
        let (m, g, lanes) = match self.0 {
            By::Lane => {
                let t = inputs.lane_vector(0, 1);
                let emb = ops::elemwise::sinusoid(&t, FREQ, 10_000.0, true, 1.0);
                let emb = ops::elemwise::silu(&emb);
                (
                    ops::linear::matmul(&emb, &ada),
                    ops::linear::matmul(&emb, &gate),
                    Some(inputs.request_of_token()),
                )
            }
            By::Token => {
                // A per-token timestep: one scalar per row, embedded per row.
                let t = inputs.latents(1, 1, Dtype::F32);
                let emb = ops::elemwise::sinusoid(&t, FREQ, 10_000.0, false, 1000.0);
                (
                    ops::linear::matmul(&emb, &ada),
                    ops::linear::matmul(&emb, &gate),
                    None,
                )
            }
        };
        let y = ops::elemwise::modulate(&normed, &m, lanes.as_ref(), ModulateForm::ScaleShift);
        let r = ops::elemwise::gated_residual_add(&x, &g, &y, lanes.as_ref());
        seam::at(seam::VELOCITY, &[&r]);
        r
    }
}

fn ty(trace: &model_ir::Trace, id: model_ir::ValueId) -> Ty {
    trace.values[id.0 as usize].ty.clone()
}

fn tensor(rows: Dim, width: u64, dtype: Dtype) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(width)],
        dtype,
    }
}

/// (a), (b), (d).
#[test]
fn a_lane_vector_is_embedded_per_lane_and_broadcast_per_row() {
    let trace = trace_hybrid("adaln", &AdaLn(By::Lane), Platform::Cuda);
    assert!(
        trace.caches.is_empty(),
        "no kv space, and the lane map was still readable"
    );

    let sinusoid = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::Sinusoid { t, y, dim, .. }) => Some((*t, *y, *dim)),
            _ => None,
        })
        .expect("the timestep is embedded");
    assert_eq!(ty(&trace, sinusoid.0), tensor(Dim::Lanes, 1, Dtype::F32));
    assert_eq!(
        ty(&trace, sinusoid.1),
        tensor(Dim::Lanes, u64::from(FREQ), Dtype::F32)
    );
    assert_eq!(sinusoid.2, FREQ);

    let modulate = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            }) => Some((*x, *m, *lane_of_row, *form, *y)),
            _ => None,
        })
        .expect("the modulation is one node");
    assert_eq!(modulate.3, ModulateForm::ScaleShift);
    assert_eq!(
        ty(&trace, modulate.1),
        tensor(Dim::Lanes, 2 * u64::from(WIDTH), Dtype::F32)
    );
    assert_eq!(
        ty(&trace, modulate.4),
        ty(&trace, modulate.0),
        "y is x's type"
    );
    let lanes = modulate.2.expect("a per-lane vector names its lane map");
    assert_eq!(
        trace.values[lanes.0 as usize].def,
        Def::Input(RuntimeInput::Geometry {
            space: 0,
            kind: GeomKind::RequestOfToken
        }),
        "the broadcast is the fire's token→lane table in the token space"
    );
    assert_eq!(
        ty(&trace, lanes),
        Ty::Tensor {
            shape: vec![Dim::Tokens],
            dtype: Dtype::I32
        }
    );

    let fold = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::GatedResidualAdd {
                r,
                g,
                lane_of_row,
                r_out,
                ..
            }) => Some((*r, *g, *lane_of_row, *r_out)),
            _ => None,
        })
        .expect("the gated fold is one node");
    assert_eq!(fold.2, Some(lanes), "the gate broadcasts by the same map");
    assert_eq!(
        ty(&trace, fold.1),
        tensor(Dim::Lanes, u64::from(WIDTH), Dtype::F32)
    );
    let mut pairs = Vec::new();
    model_ir::Operands::aliases(
        &trace
            .nodes
            .iter()
            .find(|n| {
                matches!(
                    n.op,
                    Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
                )
            })
            .unwrap()
            .op,
        &mut pairs,
    );
    assert_eq!(pairs, vec![(fold.3, fold.0)], "in place on the stream");
}

/// (c).
#[test]
fn a_per_token_vector_names_no_lane_map() {
    let trace = trace_hybrid("adaln", &AdaLn(By::Token), Platform::Cuda);
    let (x, m, lanes) = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Elementwise(Elementwise::Modulate {
                x, m, lane_of_row, ..
            }) => Some((*x, *m, *lane_of_row)),
            _ => None,
        })
        .expect("the modulation is one node");
    assert_eq!(lanes, None);
    assert_eq!(
        ty(&trace, m),
        tensor(Dim::Tokens, 2 * u64::from(WIDTH), Dtype::F32)
    );
    assert_eq!(
        ty(&trace, x),
        tensor(Dim::Tokens, u64::from(WIDTH), Dtype::Bf16)
    );
    assert!(
        !trace.values.iter().any(|decl| matches!(
            decl.def,
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::RequestOfToken,
                ..
            })
        )),
        "no lane map was read"
    );
}

/// (e).
struct WrongWidth;

impl ForwardHybrid for WrongWidth {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let m = inputs.lane_vector(0, WIDTH); // one slice where ScaleShift wants two
        ops::elemwise::modulate(
            &x,
            &m,
            Some(&inputs.request_of_token()),
            ModulateForm::ScaleShift,
        )
    }
}

#[test]
fn a_vector_of_the_wrong_width_is_refused_by_width() {
    let refused = std::panic::catch_unwind(|| trace_hybrid("wrong", &WrongWidth, Platform::Cuda));
    let message = match refused {
        Ok(_) => panic!("a one-slice vector modulated a two-slice form"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default(),
    };
    assert!(message.contains("128-wide vector"), "{message}");
}
