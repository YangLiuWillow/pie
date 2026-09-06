//! **A PLAN THAT READS A RECTANGLE AN IN-PLACE FOLD ALREADY OVERWROTE IS
//! REFUSED AT TRACE TIME — AND A FOLD ON A COPY, OR ON LANES THE READER
//! DOES NOT SERVE, IS NOT.**
//!
//! ```text
//! cargo test -p model-dsl --test an_in_place_fold_is_the_last_read_of_its_operand
//! ```
//!
//! `elementwise.add_bias` and its aliasing siblings declare `(out, in)` and
//! the arena folds every one of them onto its operand unconditionally
//! (`model_compiler::arena::fold_in_place`): no copy is minted for an
//! operand a later node still reads. That is what a biased projection
//! wants — the matmul output it folds into is its own — and it is exactly
//! what a SHARED conditioning vector does not want. An adaLN head computes
//! one timestep projection per fire and every block adds its own
//! `scale_shift_table` to it, so a block that folds into the projection
//! itself hands block `n + 1` the sum of every table before it: exact at
//! one block, drifting with depth, and invisible to a two-block miniature.
//! Three families wrote that bug independently, which is why the rule lives
//! in the validator rather than in each family's memory.
//!
//! ```text
//! (a) a per-block table folded onto the vector the stack shares is
//!     refused, naming the fold, its operand and the later reader
//! (b) the same text folding onto `ops::elemwise::copy` traces, and the
//!     copy is a real second rectangle (an `add` the tables never touch)
//! (c) the rule is guard-aware: two arms of one split folding one
//!     rectangle write disjoint rows, and are no fault
//! ```

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, Request, Stream, Value,
    Weight, ops, seam, trace_hybrid,
};
use model_ir::{Elementwise, Operands, Operation, Trace, ValueId};

const WIDTH: u32 = 64;
const BLOCKS: usize = 3;
const STREAM_BASE: u8 = 0;

struct Streams(Stream);

impl Classify for Streams {
    fn of(r: &Request) -> Streams {
        Streams(r.stream())
    }
    fn word(&self) -> u64 {
        self.0.word(STREAM_BASE)
    }
}

fn text() -> Predicate {
    Predicate::stream(STREAM_BASE, Stream::Text)
}

/// A stack of blocks whose only conditioning is one lane vector plus a
/// per-block table. `copy` says whether each block folds onto a copy.
struct Stack {
    copy: bool,
}

impl ForwardHybrid for Stack {
    type Facts = Streams;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<Streams>) -> Value {
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let lanes = inputs.request_of_token();
        // The vector the whole stack shares: `[Lanes, WIDTH]`, off the
        // timestep, computed once.
        let t = inputs.lane_vector(0, 1);
        let proj = ops::elemwise::sinusoid(&t, WIDTH, 10_000.0, true, 1.0);

        let mut x = x;
        for b in 0..BLOCKS {
            let table = Weight::sym(format!("table.{b}"), [u64::from(WIDTH)], Dtype::F32);
            let vector = if self.copy {
                ops::elemwise::copy(&proj)
            } else {
                proj.clone()
            };
            let m = ops::elemwise::add_bias(&table, &vector);
            x = ops::elemwise::gated_residual_add(&x, &m, &x, Some(&lanes));
        }
        seam::at(seam::VELOCITY, &[&x]);
        x
    }
}

/// Two streams of one sequence, each folding the SAME rectangle under its
/// own guard — a double block's two residual folds, in miniature.
struct TwoArms;

impl ForwardHybrid for TwoArms {
    type Facts = Streams;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<Streams>) -> Value {
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let (txt, img) = x.split(&text());
        let bias = Weight::sym("bias", [u64::from(WIDTH)], Dtype::Bf16);
        let other = Weight::sym("other", [u64::from(WIDTH)], Dtype::Bf16);
        let txt = ops::elemwise::add_bias(&bias, &txt);
        let img = ops::elemwise::add_bias(&other, &img);
        let x = Value::merge(vec![txt, img]);
        seam::at(seam::VELOCITY, &[&x]);
        x
    }
}

fn refusal(m: &(impl ForwardHybrid + std::panic::RefUnwindSafe)) -> String {
    let quiet = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let refused = std::panic::catch_unwind(|| trace_hybrid("stack", m, Platform::Cuda));
    std::panic::set_hook(quiet);
    match refused {
        Ok(_) => String::new(),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default(),
    }
}

/// (a)
#[test]
fn a_table_folded_onto_the_vector_the_stack_shares_is_refused() {
    let message = refusal(&Stack { copy: false });
    assert!(
        message.contains("elementwise.add_bias"),
        "a stack that folded three tables into one shared vector traced: {message:?}"
    );
    assert!(
        message.contains("overwrites") && message.contains("reads it afterwards"),
        "{message}"
    );
    // The first fold is named once per later reader — two here, since the
    // second and third blocks both read the projection after the first.
    let named = message.matches("elementwise.add_bias").count();
    assert!(named >= 2, "{named} folds named in:\n{message}");
}

/// (b)
#[test]
fn the_same_stack_folding_onto_a_copy_traces() {
    let plan: Trace = trace_hybrid("stack", &Stack { copy: true }, Platform::Cuda);
    let folds = plan
        .nodes
        .iter()
        .filter(|n| matches!(&n.op, Operation::Elementwise(Elementwise::AddBias { .. })))
        .count();
    assert_eq!(folds, BLOCKS, "one fold a block");
    // Every fold's operand is a rectangle of its own: nobody reads it after.
    let mut pairs = Vec::new();
    let mut ins = Vec::new();
    for (j, node) in plan.nodes.iter().enumerate() {
        pairs.clear();
        node.op.aliases(&mut pairs);
        for &(_, ValueId(input)) in &pairs {
            for later in &plan.nodes[j + 1..] {
                ins.clear();
                later.op.inputs(&mut ins);
                assert!(
                    !ins.contains(&ValueId(input)),
                    "node {j} folds into v{input}, which a later node reads"
                );
            }
        }
    }
    // The copy is a real second rectangle, not a re-spelling: `2v · ½`.
    let adds = plan
        .nodes
        .iter()
        .filter(|n| matches!(&n.op, Operation::Elementwise(Elementwise::Add { .. })))
        .count();
    assert_eq!(adds, BLOCKS, "one copy a block");
}

/// (c)
#[test]
fn two_arms_of_one_split_fold_one_rectangle_on_disjoint_rows() {
    let message = refusal(&TwoArms);
    assert!(
        message.is_empty(),
        "a guarded fold was read as a clobber: {message}"
    );
    // Non-vacuous only if the two arms really are one rectangle: a
    // guard-blind rule would have to fault this plan.
    let plan = trace_hybrid("arms", &TwoArms, Platform::Cuda);
    let operands: Vec<ValueId> = plan
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Operation::Elementwise(Elementwise::AddBias { out, .. }) => Some(*out),
            _ => None,
        })
        .collect();
    assert_eq!(operands.len(), 2, "two folds");
    assert_eq!(
        operands[0], operands[1],
        "the two arms fold different values, so the claim proves nothing"
    );
}
