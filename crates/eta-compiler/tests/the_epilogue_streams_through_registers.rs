//! A denoiser's epilogue over `[rows, vocab]` — temperature, softmax,
//! entropy, Gumbel, argmax, acceptance — used to be one loop per op over the
//! row, each a round trip through scratch. On CUDA a row-parallel region now
//! fuses the elementwise run into streams: one pass per reduction boundary,
//! intermediates in registers, a value stored only if something outside the
//! stream reads it.

use eta_compiler::codegen::cuda::fused::emit_fused_region;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::types::RngKind;
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::bind;

const ROWS: u32 = 256;
const VOCAB: u32 = 4096;

fn scalar_writer() -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::Writer,
        seeded: true,
    }
}

fn rows_out(dtype: Dtype) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(ROWS),
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

/// `scaled = logits / t; p = softmax(scaled); h = -Σ p·log p; argmax(scaled);
/// gumbel = scaled + g; sampled = argmax(gumbel); accept = h < bound`.
fn subject() -> TraceContainer {
    // The temperature first, as the DSL emits it: a scalar op after the
    // logits would end the row run before the arithmetic joins it.
    let ops = vec![
        // 0: the temperature, a host-set scalar channel; 1: as a rank-0 scalar
        Op::ChanRead(0),
        Op::Reshape {
            value: 0,
            shape: Shape::new(&[]).unwrap(),
        },
        // 2: logits [rows, vocab]
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        // 3: scaled = logits / t   (a scalar operand broadcasts)
        Op::Div(2, 1),
        // 4: row max [rows]
        Op::ReduceMax(3),
        // 5: [rows, vocab] of the max — the DSL's broadcast
        Op::Broadcast {
            value: 4,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        // 6: centred
        Op::Sub(3, 5),
        // 7: e = exp(centred)
        Op::Exp(6),
        // 8: z = Σ e
        Op::ReduceSum(7),
        // 9: z broadcast
        Op::Broadcast {
            value: 8,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        // 10: p = e / z
        Op::Div(7, 9),
        // 11: log p
        Op::Log(10),
        // 12: p log p
        Op::Mul(10, 11),
        // 13: Σ p log p
        Op::ReduceSum(12),
        // 14: h = -Σ
        Op::Neg(13),
        // 15: argmax of the scaled row
        Op::ReduceArgmax(3),
        // 16: gumbel noise
        Op::Rng {
            stream: 7,
            shape: Shape::matrix(ROWS, VOCAB),
            kind: RngKind::Gumbel,
        },
        // 17: noisy = scaled + g
        Op::Add(3, 16),
        // 18: sampled = argmax(noisy)
        Op::ReduceArgmax(17),
        // 19: the bound
        Op::Const(Literal::F32(0.1)),
        // 20: accept = h < bound
        Op::Lt(14, 19),
        Op::ChanPut { chan: 1, value: 15 },
        Op::ChanPut { chan: 2, value: 18 },
        Op::ChanPut { chan: 3, value: 20 },
        Op::ChanPut { chan: 4, value: 14 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            scalar_writer(),
            rows_out(Dtype::I32),
            rows_out(Dtype::I32),
            rows_out(Dtype::Bool),
            rows_out(Dtype::F32),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = VOCAB;
    profile
}

#[test]
fn the_row_parallel_epilogue_fuses_into_streams() {
    let bound = bind(subject(), profile()).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    // Every row-parallel generated region of the stage, emitted; the chain
    // may be cut over several by the partitioner, the streams are counted
    // across them all.
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.row_value.is_none() {
            continue;
        }
        let name = format!("k{index}");
        let emitted = emit_fused_region(&name, stage, region).expect("cuda emits");
        if std::env::var_os("PTIR_SHOW").is_some() {
            eprintln!("{emitted}");
        }
        // The body after the row-view preamble: the runtime text above it
        // defines the helpers this test asserts are not CALLED.
        let body = emitted
            .split("descriptors = ptir_rowdesc;")
            .nth(1)
            .expect("a row-parallel kernel has the row-view preamble");
        source.push_str(body);
    }
    assert!(!source.is_empty(), "no row-parallel region");
    let streams = source.matches("// stream of ").count();
    // The pass boundaries are the reductions something later reads: the row
    // max, the sum, the entropy sum — four passes at most for this chain,
    // never one loop per op.
    assert!(
        (2..=5).contains(&streams),
        "{streams} streams for a chain of a dozen elementwise ops; the emitter fell back to a loop per op or fused across a reduction"
    );
    // The two elementwise ops left to the helper are the per-row scalar
    // ones (`h = -Σ`, `h < bound`): one element a block, not the row.
    let helper_loops = source.matches("ptir_parallel_elementwise(").count();
    assert!(
        helper_loops <= 2,
        "{helper_loops} elementwise ops still run as their own loop over the row"
    );
    assert!(
        !source.contains("ptir_parallel_broadcast("),
        "a scalar's broadcast still materialises the row"
    );
    assert!(
        !source.contains("ptir_parallel_intrinsic("),
        "the logits are read only by the stream and must not be copied into scratch first"
    );
    assert!(
        !source.contains("offsets[2]"),
        "the logits value (2) never touches scratch"
    );
    // `centred` (6), `log p` (11) and `p log p` (12) are read only inside
    // their streams: no store lands them.
    for spent in [6u32, 11, 12] {
        assert!(
            !source.contains(&format!("offsets[{spent}]")),
            "value {spent} is spent inside its stream and must not touch scratch"
        );
    }
}

/// The DSL spells a row's scalar as `broadcast(reshape(m, [rows, 1]),
/// [rows, vocab])`. The reshape must fold away in normalization, or the
/// broadcast lands a whole `[rows, vocab]` plane in scratch per fire.
#[test]
fn a_reshaped_row_vector_broadcasts_through_the_stream() {
    let ops = vec![
        Op::ChanRead(0),
        Op::Reshape {
            value: 0,
            shape: Shape::new(&[]).unwrap(),
        },
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(ROWS, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Div(2, 1),
        Op::ReduceMax(3),
        // 5: the row max as a column, as the DSL emits it
        Op::Reshape {
            value: 4,
            shape: Shape::matrix(ROWS, 1),
        },
        Op::Broadcast {
            value: 5,
            shape: Shape::matrix(ROWS, VOCAB),
        },
        Op::Sub(3, 6),
        Op::Exp(7),
        Op::ReduceSum(8),
        Op::ChanPut { chan: 1, value: 9 },
    ];
    let container = TraceContainer {
        names: Vec::new(),
        channels: vec![scalar_writer(), rows_out(Dtype::F32)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    };
    let bound = bind(container, profile()).expect("binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    assert!(
        !stage.normalized.ops.iter().any(|op| matches!(op, Op::Reshape { shape, .. } if shape.rank() == 2)),
        "the [rows, 1] reshape survived normalization: {:?}",
        stage.normalized.ops
    );
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.row_value.is_none() {
            continue;
        }
        let emitted = emit_fused_region(&format!("k{index}"), stage, region).expect("cuda emits");
        source.push_str(emitted.split("descriptors = ptir_rowdesc;").nth(1).unwrap_or(""));
    }
    assert!(
        !source.contains("ptir_parallel_broadcast("),
        "the row max's broadcast still materialises the row"
    );
}
