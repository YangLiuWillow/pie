//! **A RAGGED ATTENTION MAY READ ITS QUERIES OFF ONE ARM AND ITS KEYS OFF
//! ANOTHER, AND IS THE ONLY OP THAT MAY.**
//!
//! ```text
//! cargo test -p model-dsl --test a_ragged_attention_joins_two_arms
//! ```
//!
//! `Recorder::push` refuses two operands from different split arms — the
//! rule that keeps a class's rows the only rows its nodes touch. D2 makes
//! one exception, for `attention.ragged`: cross-attention reads `q` off the
//! audio lanes' class and `k`/`v` off the video lanes', and the node must
//! run over BOTH windows. So:
//!
//! ```text
//! (a) the two-arm text traces and validates, with no kv space declared
//! (b) the ragged node's guard is the Or of the two arms' guards
//! (c) its answer is read back on the queries' arm: the unpack that follows
//!     carries the audio arm's guard, not the Or
//! (d) the packing tables it reads are keyed by each arm's own selection
//! (e) every other op still refuses two arms, at the line that mixed them
//! ```

mod common;

use common::{CrossAttention, HEAD_DIM, StreamFacts};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, Request, Selection,
    Stream, Value, Weight, ops, trace_hybrid,
};
use model_ir::{Attention, Def, GeomKind, Guard, Layout, Operation, RuntimeInput};

fn cond_of(p: &Predicate) -> Guard {
    match p {
        Predicate::Fact { bit } => Guard::Fact(*bit),
        Predicate::Not(a) => Guard::not(cond_of(a)),
        Predicate::And(a, b) => Guard::and(cond_of(a), cond_of(b)),
        Predicate::Rest => unreachable!(),
    }
}

/// (a), (b), (c), (d).
#[test]
fn queries_off_one_arm_and_keys_off_another_trace_under_the_or_of_both() {
    let trace = trace_hybrid("cross", &CrossAttention, Platform::Cuda);
    assert!(trace.caches.is_empty(), "a denoiser declares no kv space");

    // The two arms of the three-way split: audio, then video-and-not-audio.
    let audio = cond_of(&StreamFacts::on(Stream::Audio));
    let video = Guard::and(
        Guard::not(audio.clone()),
        cond_of(&StreamFacts::on(Stream::Video)),
    );

    let (at, ragged) = trace
        .nodes
        .iter()
        .enumerate()
        .find(|(_, node)| matches!(node.op, Operation::Attention(Attention::Ragged { .. })))
        .expect("the text holds the one ragged attention it was written for");
    let Operation::Attention(Attention::Ragged {
        head_dim,
        kv_heads,
        q_indptr,
        kv_indptr,
        ..
    }) = &ragged.op
    else {
        unreachable!()
    };
    assert_eq!(*head_dim, HEAD_DIM);
    assert_eq!(*kv_heads, 4, "kv heads are read off k's width");

    // (b) the node runs over both windows.
    let both = Guard::or(audio.clone(), video.clone());
    assert!(
        ragged.guard.equivalent(&both),
        "the ragged node is guarded by {:?}, not the join of audio and video",
        ragged.guard
    );
    assert!(
        !ragged.guard.equivalent(&audio) && !ragged.guard.equivalent(&video),
        "the join is wider than either arm"
    );

    // (c) the unpack after it reads the answer on the audio arm alone.
    let unpack = trace.nodes[at + 1..]
        .iter()
        .find(|node| matches!(node.op, Operation::Layout(Layout::UnpackRows { .. })))
        .expect("the answer is unpacked onto the audio rows");
    assert!(
        unpack.guard.equivalent(&audio),
        "the unpack is guarded by {:?}, not the queries' arm",
        unpack.guard
    );
    assert_eq!(
        unpack.guard, audio,
        "and spelled exactly as the audio arm spells itself, so the next op accepts it"
    );

    // (d) each side's CSR is that side's own selection.
    let selection = |id: model_ir::ValueId| match &trace.values[id.0 as usize].def {
        Def::Input(RuntimeInput::Geometry {
            space: 0,
            kind: GeomKind::LaneIndptr { select },
        }) => *select,
        other => panic!("a CSR is a lane indptr in the token space, not {other:?}"),
    };
    assert_eq!(selection(*q_indptr), Selection::of(&audio).unwrap());
    assert_eq!(selection(*kv_indptr), Selection::of(&video).unwrap());
    assert_ne!(selection(*q_indptr), selection(*kv_indptr));
    let perms = trace
        .values
        .iter()
        .filter(|decl| matches!(decl.def, Def::Input(RuntimeInput::RowPermutation { .. })))
        .count();
    assert_eq!(
        perms, 2,
        "one permutation per side, deduplicated per selection"
    );
}

/// (e): the rule stands for every other op.
struct TwoArmsIntoOneAdd;

impl ForwardHybrid for TwoArmsIntoOneAdd {
    type Facts = StreamFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (audio, video) = inputs.split(&StreamFacts::on(Stream::Audio));
        let w = Weight::sym("w", [8, 8], Dtype::Bf16);
        let a = ops::linear::matmul(&audio.latents(0, 8, Dtype::Bf16), &w);
        let b = ops::linear::matmul(&video.latents(0, 8, Dtype::Bf16), &w);
        ops::elemwise::add(&a, &b)
    }
}

#[test]
fn every_other_op_still_refuses_two_arms() {
    let refused =
        std::panic::catch_unwind(|| trace_hybrid("mixed", &TwoArmsIntoOneAdd, Platform::Cuda));
    let message = match refused {
        Ok(_) => panic!("an add over two arms traced"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    };
    assert!(
        message.contains("elementwise.add") && message.contains("different split arms"),
        "the refusal names the op and the rule: {message}"
    );
}

/// The stream facts round-trip through a `Request`: a lane on the video
/// stream lands in the video class and nowhere else.
#[test]
fn a_lanes_stream_is_its_fact_word() {
    let word = StreamFacts::of(&Request::new(16, false).on_stream(Stream::Video)).word();
    assert!(cond_of(&StreamFacts::on(Stream::Video)).holds(word));
    assert!(!cond_of(&StreamFacts::on(Stream::Audio)).holds(word));
    let text = StreamFacts::of(&Request::new(16, false)).word();
    assert!(
        cond_of(&StreamFacts::on(Stream::Text)).holds(text),
        "text is the default"
    );
}
