//! **A GUMBEL-MAX DRAW READS THE LOGITS ONCE.** `gumbel_max(logits / T, state)`
//! traces as `intrinsic_val` → `div` → `add(·, rng_keyed gumbel)` →
//! `reduce_argmax`: four launches in the fused region, each writing a
//! vocabulary-wide row to scratch, and the argmax reading it back. The CUDA
//! emitter folds the head into `ptir_fast_gumbel_argmax_intrinsic`, which
//! draws the noise per element off the intrinsic and keeps nothing — the
//! same road a bare argmax takes through `ptir_fast_argmax_intrinsic`.
//!
//! The chain is recognised with and without the divide, and left alone when
//! the noise is uniform (not Gumbel) or when something else reads a value on
//! the way.

use eta_compiler::codegen::cuda::emit_fused_region;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, RngKind, Shape};
use eta_ir::validate::bind;

const VOCAB: u32 = 16;

fn channels() -> Vec<ChannelDecl> {
    vec![
        // 0: the sampled token, read by the host.
        ChannelDecl {
            shape: Shape::new(&[]).expect("a scalar shape"),
            dtype: ChanDType::Concrete(Dtype::I32),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        },
        // 1: the `[key, ctr]` rng state, written by the host.
        ChannelDecl {
            shape: Shape::vector(2),
            dtype: ChanDType::Concrete(Dtype::U32),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: false,
        },
    ]
}

/// `argmax(logits [/ T] + noise(kind))`, then the token out.
fn subject(divide: bool, kind: RngKind) -> TraceContainer {
    let mut ops = vec![
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        }, // 0
        Op::Reshape {
            value: 0,
            shape: Shape::vector(VOCAB),
        }, // 1
        Op::ChanTake(1), // 2: state
    ];
    let logits = if divide {
        ops.push(Op::Const(Literal::F32(0.7))); // 3
        ops.push(Op::Div(1, 3)); // 4
        4
    } else {
        1
    };
    let noise = ops.len() as u32;
    ops.push(Op::RngKeyed {
        state: 2,
        shape: Shape::vector(VOCAB),
        kind,
    });
    let perturbed = ops.len() as u32;
    ops.push(Op::Add(logits, noise));
    let token = ops.len() as u32;
    ops.push(Op::ReduceArgmax(perturbed));
    ops.push(Op::ChanPut {
        chan: 0,
        value: token,
    });
    TraceContainer {
        names: Vec::new(),
        channels: channels(),
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn emitted(container: TraceContainer) -> String {
    let mut profile = ModelProfile::dummy();
    profile.vocab = VOCAB;
    let bound = bind(container, profile).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let region = stage.fused.regions.first().expect("one fused region");
    let source = emit_fused_region("gumbel", stage, region).expect("the CUDA emitter serves it");
    // The runtime prologue defines every helper; only the entry point's body
    // says which ones this region calls.
    let entry = source
        .find("__global__ void gumbel")
        .expect("the entry point is in the source");
    source[entry..].to_string()
}

#[test]
fn a_scaled_gumbel_max_becomes_one_scan_of_the_intrinsic() {
    let source = emitted(subject(true, RngKind::Gumbel));
    assert!(
        source.contains("ptir_fast_gumbel_argmax_intrinsic("),
        "the head must fold into the one-scan form"
    );
    assert!(
        !source.contains("ptir_fast_argmax("),
        "no scratch-reading argmax is left behind"
    );
    let head = source.find("ptir_fast_gumbel_argmax_intrinsic(").unwrap();
    let call = &source[head..head + 700];
    assert!(call.contains("descriptors[p.a0]"), "the operand shape sizes the scan");
    assert!(!call.contains("nullptr"), "the divisor rides along");
}

#[test]
fn an_unscaled_gumbel_max_folds_too_with_no_divisor() {
    let source = emitted(subject(false, RngKind::Gumbel));
    let head = source.find("ptir_fast_gumbel_argmax_intrinsic(").expect("the head folds");
    assert!(source[head..head + 700].contains("nullptr"), "nothing to divide by");
}

#[test]
fn a_uniform_draw_is_not_a_gumbel_max_and_stays_as_traced() {
    let source = emitted(subject(true, RngKind::Uniform));
    assert!(!source.contains("ptir_fast_gumbel_argmax_intrinsic("));
    assert!(source.contains("ptir_fast_argmax("), "the traced argmax runs over scratch");
}
