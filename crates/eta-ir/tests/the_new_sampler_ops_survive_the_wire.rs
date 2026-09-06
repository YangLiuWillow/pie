//! **A NEW OP IS ONLY REAL ONCE IT SURVIVES THE CONTAINER.** `declare_ops!`
//! gives `sin`/`cos`/`sqrt`/`rsqrt` a tag and `RngKind::Normal` a kind byte;
//! neither is worth anything if the encoder writes bytes the decoder reads
//! back as something else. The crate's own `round_trip_every_op` sweeps
//! `representatives()`, which is one canned instance per row — this pins the
//! two things that sweep cannot: that the four new tags are the numbers the
//! wire froze (appended above `cast`, never renumbering a shipped op), and
//! that a `rng_keyed` carrying the new kind byte comes back carrying it and
//! not silently as `Uniform`.
//!
//! ```text
//! cargo test -p eta-ir --test the_new_sampler_ops_survive_the_wire
//! ```

use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer, decode, encode,
};
use eta_ir::op::{Op, tags};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, RngKind, Shape};

fn container(ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![ChannelDecl {
            shape: Shape::vector(4),
            dtype: ChanDType::Concrete(Dtype::U32),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: true,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

#[test]
fn the_appended_tags_are_the_numbers_the_wire_froze() {
    // Appended, not renumbered: every tag a shipped container can carry
    // still means what it meant, and the four new ones sit in the gap
    // `cast` left above itself.
    for (tag, name) in [
        (tags::EXP, "exp"),
        (tags::LOG, "log"),
        (tags::NEG, "neg"),
        (tags::RECIP, "recip"),
        (tags::ABS, "abs"),
        (tags::SIGN, "sign"),
        (tags::CAST, "cast"),
        (tags::SIN, "sin"),
        (tags::COS, "cos"),
        (tags::SQRT, "sqrt"),
        (tags::RSQRT, "rsqrt"),
    ] {
        assert_eq!(
            eta_ir::op::spec(tag).map(|row| row.name),
            Some(name),
            "tag {tag:#04x} is not `{name}`"
        );
    }
    assert_eq!(
        [tags::SIN, tags::COS, tags::SQRT, tags::RSQRT],
        [8, 9, 10, 11]
    );
    // `spec` binary-searches, so a row out of tag order answers `None` for
    // an op that exists.
    let tags: Vec<u8> = eta_ir::op::OP_TABLE.iter().map(|row| row.tag).collect();
    let mut sorted = tags.clone();
    sorted.sort_unstable();
    assert_eq!(tags, sorted, "OP_TABLE is no longer sorted by tag");
}

#[test]
fn a_sampler_epilogue_of_new_ops_round_trips() {
    let ops = vec![
        Op::ChanRead(0), // 0 state
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Normal,
        }, // 1
        Op::Sin(1),      // 2
        Op::Cos(2),      // 3
        Op::Sqrt(3),     // 4
        Op::Rsqrt(4),    // 5
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Gumbel,
        }, // 6
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Uniform,
        }, // 7
    ];
    let source = container(ops);
    let bytes = encode(&source);
    let back = decode(&bytes).expect("the container decodes");
    assert_eq!(back, source, "the epilogue did not survive the wire");

    // The kind byte in particular: an unknown kind decoding as `Uniform`
    // would turn a Gaussian latent into a uniform one with no diagnostic.
    let kinds: Vec<RngKind> = back.stages[0]
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::RngKeyed { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec![RngKind::Normal, RngKind::Gumbel, RngKind::Uniform]
    );
}

#[test]
fn an_unknown_rng_kind_is_refused_rather_than_read_as_uniform() {
    let source = container(vec![
        Op::ChanRead(0),
        Op::RngKeyed {
            state: 0,
            shape: Shape::vector(4),
            kind: RngKind::Normal,
        },
    ]);
    let bytes = encode(&source);
    let kind_byte = RngKind::Normal as u8;
    let at = bytes
        .iter()
        .rposition(|&b| b == kind_byte)
        .expect("the kind byte is on the wire");
    let mut mutant = bytes.clone();
    mutant[at] = 0x7f;
    assert!(
        decode(&mutant).is_err(),
        "a kind byte no `RngKind` claims decoded anyway"
    );
}
