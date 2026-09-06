//! **MM-DiT JOINT ATTENTION IS A MERGE OF TWO STREAMS' PROJECTIONS, ONE
//! RAGGED ATTENTION OVER THE GROUP CSR, AND A SPLIT BACK ONTO THE ARMS —
//! AND THE SPLIT ARMS ARE SPELLED THE WAY THEIR SIBLINGS ARE.**
//!
//! ```text
//! cargo test -p model-dsl --test a_joint_attention_over_merged_streams_splits_back_onto_its_arms
//! ```
//!
//! FLUX's double block: the text lanes and the image lanes each run their
//! own qkv, `Value::merge` puts both in one `[Tokens, H·d]` rectangle, one
//! `attention.ragged` over the packed group order lets every lane of a
//! request see every other, and the answer is split per stream again for
//! the per-stream output projections. The merge's guard is `Or(text, image)`
//! and a value read under `text` afterwards must meet the text residual —
//! spelled `text` — without a "mixes arms" refusal:
//!
//! ```text
//! (a) the block traces; the ragged node's guard is the merge's Or
//! (b) the per-stream output projections carry exactly their arm's guard
//! (c) the residual folds meet the stream's own residual, one per arm
//! (d) the group CSR is the whole-fire selection when read off the joint
//!     value's arm — a merge of the two streams has no single mask
//! ```

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, RaggedMask, Request,
    Stream, Value, Weight, ops, seam, trace_hybrid,
};
use model_ir::{Attention, Def, Elementwise, GeomKind, Guard, Operation, RuntimeInput, Selection};

struct StreamFacts(Stream);

impl StreamFacts {
    fn on(stream: Stream) -> Predicate {
        Predicate::stream(0, stream)
    }
}

impl Classify for StreamFacts {
    fn of(r: &Request) -> StreamFacts {
        StreamFacts(r.stream())
    }
    fn word(&self) -> u64 {
        self.0.word(0)
    }
}

const WIDTH: u32 = 32;
const HEAD_DIM: u32 = 8;

struct DoubleBlock;

impl ForwardHybrid for DoubleBlock {
    type Facts = StreamFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (txt, img) = inputs.split(&StreamFacts::on(Stream::Text));
        let w = |name: &str| Weight::sym(name, [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let x_txt = txt.latents(0, WIDTH, Dtype::Bf16);
        let x_img = img.latents(1, WIDTH, Dtype::Bf16);
        let project = |x: &Value, prefix: &str| {
            (
                ops::linear::matmul(x, &w(&format!("{prefix}.q"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.k"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.v"))),
            )
        };
        let (qt, kt, vt) = project(&x_txt, "txt");
        let (qi, ki, vi) = project(&x_img, "img");
        let q = Value::merge(vec![qt, qi]);
        let k = Value::merge(vec![kt, ki]);
        let v = Value::merge(vec![vt, vi]);
        // Every lane of the fire joins the packing: the joint value belongs
        // to no single arm, so the whole-fire input's tables serve it.
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            0.35,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let (o_txt, o_img) = o.split(&StreamFacts::on(Stream::Text));
        let y_txt = ops::linear::matmul(&o_txt, &w("txt.o"));
        let y_img = ops::linear::matmul(&o_img, &w("img.o"));
        let r_txt = ops::elemwise::residual_add(&y_txt, &x_txt);
        let r_img = ops::elemwise::residual_add(&y_img, &x_img);
        let out = Value::merge(vec![r_txt, r_img]);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

#[test]
fn the_joint_attention_traces_and_its_answer_splits_back_onto_the_arms() {
    let trace = trace_hybrid("double", &DoubleBlock, Platform::Cuda);
    let text = Guard::Fact(Stream::Text.code());
    let image = Guard::not(text.clone());

    // (a)
    let ragged = trace
        .nodes
        .iter()
        .find(|node| matches!(node.op, Operation::Attention(Attention::Ragged { .. })))
        .expect("one joint attention");
    assert!(
        ragged
            .guard
            .equivalent(&Guard::or(text.clone(), image.clone()))
    );
    let Operation::Attention(Attention::Ragged {
        q_indptr,
        kv_indptr,
        mask,
        ..
    }) = &ragged.op
    else {
        unreachable!()
    };
    assert_eq!(*mask, RaggedMask::GroupBlockDiagonal);
    assert_eq!(q_indptr, kv_indptr, "self-attention: one CSR both sides");

    // (d)
    assert_eq!(
        trace.values[q_indptr.0 as usize].def,
        Def::Input(RuntimeInput::Geometry {
            space: 0,
            kind: GeomKind::GroupIndptr {
                select: Selection::ALL
            }
        })
    );

    // (b), (c): the four nodes after the unpack — two projections, two folds
    // — alternate text/image and carry exactly those guards.
    let folds: Vec<&Guard> = trace
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                Operation::Elementwise(Elementwise::ResidualAdd { .. })
            )
        })
        .map(|node| &node.guard)
        .collect();
    assert_eq!(
        folds,
        vec![&text, &image],
        "one fold per arm, spelled as the arm"
    );
    let projections: Vec<&Guard> = trace
        .nodes
        .iter()
        .rev()
        .filter(|node| matches!(node.op, Operation::Linear(model_ir::Linear::Matmul { .. })))
        .take(2)
        .map(|node| &node.guard)
        .collect();
    assert_eq!(projections, vec![&image, &text]);

    // The sweep resolves every merge: a text lane takes the text arm.
    let classes = model_dsl::resolve_classes(&trace).expect("every merge resolves");
    let word = StreamFacts(Stream::Text).word() & classes.mask;
    let class = classes.class_of(word).expect("a text lane has a class");
    for row in 0..classes.merges.len() {
        assert_eq!(
            classes.merge_arm[row][class],
            Some(0),
            "arm 0 is the text arm"
        );
    }
}
