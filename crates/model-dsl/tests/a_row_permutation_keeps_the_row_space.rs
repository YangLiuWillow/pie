//! **PACKING ROWS BY GROUP AND UNPACKING THEM AGAIN CHANGES THE ORDER AND
//! NOTHING ELSE: THE ROW SPACE, WIDTH AND ELEMENT ARE THE INPUT'S.**
//!
//! ```text
//! cargo test -p model-dsl --test a_row_permutation_keeps_the_row_space
//! ```
//!
//! `layout.pack_rows` gathers by `RuntimeInput::RowPermutation` and
//! `layout.unpack_rows` scatters by the same vector (D2). Neither is a
//! reshape — the IR has none — so:
//!
//! ```text
//! (a) both answer exactly `x`'s type, on `Dim::Tokens`
//! (b) the permutation is `[Tokens]` i32, one per arm's selection, and the
//!     two ops of one arm read the same value
//! (c) both are fresh rectangles (no alias): a gather cannot run in place
//! (d) an rmsnorm-shaped chain around them types as before, so the packing
//!     is transparent to the ops on either side
//! ```

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, ops, seam,
    trace_hybrid,
};
use model_ir::{Def, Dim, Layout, Operands, Operation, RuntimeInput, Selection, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

struct PackThenUnpack;

impl ForwardHybrid for PackThenUnpack {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = inputs.latents(0, 24, Dtype::Bf16);
        let perm = inputs.row_permutation();
        let packed = ops::layout::pack_rows(&x, &perm);
        let normed = ops::elemwise::rmsnorm_no_scale(&packed, 24, 1e-6);
        let back = ops::layout::unpack_rows(&normed, &perm);
        seam::at(seam::VELOCITY, &[&back]);
        back
    }
}

#[test]
fn pack_and_unpack_keep_the_row_space() {
    let trace = trace_hybrid("pack", &PackThenUnpack, Platform::Cuda);
    let ty = |id: model_ir::ValueId| trace.values[id.0 as usize].ty.clone();
    let want = Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(24)],
        dtype: Dtype::Bf16,
    };

    let (pack, unpack) =
        trace
            .nodes
            .iter()
            .fold((None, None), |(pack, unpack), node| match &node.op {
                Operation::Layout(Layout::PackRows { x, perm, y }) => {
                    (Some((*x, *perm, *y)), unpack)
                }
                Operation::Layout(Layout::UnpackRows { x, perm, y }) => {
                    (pack, Some((*x, *perm, *y)))
                }
                _ => (pack, unpack),
            });
    let pack = pack.expect("the gather is one node");
    let unpack = unpack.expect("the scatter is one node");

    // (a)
    assert_eq!(ty(pack.0), want);
    assert_eq!(ty(pack.2), want, "the packed rectangle is the input's type");
    assert_eq!(ty(unpack.2), want, "and so is the unpacked one");
    assert_eq!(
        ty(unpack.0),
        want,
        "(d) the norm between them answered the same type"
    );

    // (b)
    assert_eq!(pack.1, unpack.1, "one permutation serves both directions");
    assert_eq!(
        trace.values[pack.1.0 as usize].def,
        Def::Input(RuntimeInput::RowPermutation {
            select: Selection::ALL
        }),
        "an unsplit input's selection is every lane"
    );
    assert_eq!(
        ty(pack.1),
        Ty::Tensor {
            shape: vec![Dim::Tokens],
            dtype: Dtype::I32
        }
    );

    // (c)
    for node in &trace.nodes {
        if matches!(
            node.op,
            Operation::Layout(Layout::PackRows { .. } | Layout::UnpackRows { .. })
        ) {
            let mut pairs = Vec::new();
            node.op.aliases(&mut pairs);
            assert!(
                pairs.is_empty(),
                "{} writes a fresh rectangle",
                node.op.name()
            );
        }
    }
}
