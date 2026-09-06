//! **A `layout.pack_rows` / `layout.unpack_rows` RUNS IN EVERY CLASS ITS
//! GUARD ADMITS, EVEN WHEN ONE CLASS ALONE READS ITS ANSWER.**
//!
//! ```text
//! cargo test -p model-dsl --test every_row_permutation_runs_in_every_class_its_guard_admits
//! ```
//!
//! The two permutations are launched over the window of the classes they are
//! demanded in, and their permutation vector is fire-absolute over the whole
//! selection (`model_exec::fire::packing`). Narrow that window to one class
//! and the launch still walks packed rows `[start, start + rows)` — packed
//! rows, not that class's rows — so it writes another class's rows and
//! leaves its own tail carrying whatever was there before.
//!
//! A DiT's LAST joint block is exactly this shape: the head reads the image
//! rows alone (`pred[:, :S_img]`), so per-class demand would run that
//! block's unpack over the image window only. On FLUX.2-klein at 1024²
//! (512 text rows packed after 4096 image rows) that left the last 512 image
//! rows carrying the block's input and cost the step-0 velocity a cosine of
//! 0.9948 against the diffusers golden; rooting the node in both classes
//! (`model_ir::check::classes::spans_classes`) puts it at 0.99955, the bf16
//! floor.
//!
//! The claim: with a joint attention whose answer is split and only the
//! image arm read, both the packs and the unpack are demanded in the text
//! class as well as the image one.

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, RaggedMask, Request,
    Stream, Value, Weight, ops, seam, trace_hybrid,
};
use model_ir::{Layout, Operation};

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

/// One joint block whose answer only the image arm reads — the head of a
/// FLUX-shaped denoiser.
struct LastJointBlock;

impl ForwardHybrid for LastJointBlock {
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
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&Value::merge(vec![qt, qi]), &perm),
            &ops::layout::pack_rows(&Value::merge(vec![kt, ki]), &perm),
            &ops::layout::pack_rows(&Value::merge(vec![vt, vi]), &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            0.35,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        // The head: the text rows are dropped, so nothing downstream of here
        // is demanded in the text class.
        let (_, o_img) = o.split(&StreamFacts::on(Stream::Text));
        let out = ops::linear::matmul(&o_img, &w("img.o"));
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

#[test]
fn the_pack_and_unpack_of_a_head_only_block_are_demanded_in_both_streams() {
    let trace = trace_hybrid("last_joint", &LastJointBlock, Platform::Cuda);
    let classes = model_dsl::resolve_classes(&trace).expect("every merge resolves");
    let text = classes
        .class_of(StreamFacts(Stream::Text).word() & classes.mask)
        .expect("a text lane has a class");
    let image = classes
        .class_of(StreamFacts(Stream::Image).word() & classes.mask)
        .expect("an image lane has a class");
    assert_ne!(text, image, "the two streams are two classes");

    let permutations: Vec<usize> = trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.op,
                Operation::Layout(Layout::PackRows { .. } | Layout::UnpackRows { .. })
            )
        })
        .map(|(at, _)| at)
        .collect();
    assert_eq!(permutations.len(), 4, "three packs and one unpack");
    for at in permutations {
        let mask = &classes.node_mask[at];
        assert!(
            mask.contains(text) && mask.contains(image),
            "node {at} ({:?}) runs in classes {mask:?}; a row permutation's window is the \
             SELECTION's packed rectangle, so it must run in every class its guard admits",
            trace.nodes[at].op
        );
    }
}
