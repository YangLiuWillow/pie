//! **THE SHAPE THE GROUPED ROUTED MATMUL ASSUMES, CHECKED WHERE IT IS
//! DECLARED.**
//!
//! `kernels_cuda::linear::moe::matmul_select` groups a wide fire by expert
//! instead of firing one GEMV per route, and it checks the bank it was
//! handed before it does: the shell flattens a `[experts, N, K]` declaration
//! on its LAST TWO axes, so a routed bank arrives as `[experts, N * K]` —
//! one row per expert, each row that expert's whole plane. The entry takes
//! `bank.rows` for the expert count and requires `bank.width` to be
//! `x.width * y.width`. A bank declared any other way fails that and the
//! entry falls back to the per-route GEMV.
//!
//! That the flattening is on the last two axes and not the first is not a
//! detail: reading it the other way makes the check reject every real fire,
//! and the optimisation is then dead in the engine while every kernel test
//! still passes. It was, until `engine-cuda`'s
//! `a_prefill_past_the_gemvs_grid_is_served` fired one and found out.
//!
//! That fallback is silent, and it is meant to be — a wrong guess about a
//! bank's layout must not be a wrong answer. But silence is also how a
//! family could lose the grouping without anyone noticing, so the shape the
//! entry assumes is asserted here, against the declaration itself, for
//! every routed select the catalog ships.
//!
//! It walks every SKU the catalog ships — several hundred routed selects
//! across the MoE families, deepseek's included — and it is a claim about
//! the DECLARATION, not about the kernel: the kernel's own answer is read
//! against a host dot in `kernels-cuda`'s
//! `the_grouped_expert_select_answers_the_routed_dot`.

use model_dsl::{Def, Dim, Linear, Operation, Platform, Trace, Ty, ValueId};

/// The last axis of a value's rectangle — the width the entry states to
/// cuBLAS.
fn width(trace: &Trace, v: ValueId) -> Option<u64> {
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.last() {
            Some(Dim::Const(w)) => Some(*w),
            _ => None,
        },
        _ => None,
    }
}

/// The rectangle a weight was declared at, or `None` for a value that is
/// not a weight.
fn declared(trace: &Trace, v: ValueId) -> Option<Vec<u64>> {
    match trace.values[v.0 as usize].def {
        Def::Weight(at) => Some(trace.params[at as usize].shape.clone()),
        _ => None,
    }
}

#[test]
fn a_routed_bank_is_the_rectangle_the_grouped_leg_reads() {
    let mut faults = Vec::new();
    let mut checked = 0usize;

    for row in models::skus() {
        let trace = (row.trace)(Platform::Cuda);
        for node in &trace.nodes {
            let Operation::Linear(Linear::MoeMatmulSelect { x, bank, y, .. }) = &node.op else {
                continue;
            };
            let (Some(bank_shape), Some(k), Some(n)) =
                (declared(&trace, *bank), width(&trace, *x), width(&trace, *y))
            else {
                faults.push(format!(
                    "`{}`: a routed select whose bank is not a declared weight, or whose \
                     activation and result are not constant-width rectangles — the grouped \
                     leg reads its expert count off exactly those three",
                    row.name,
                ));
                continue;
            };
            checked += 1;
            // `[experts, N, K]`, which the shell flattens to the `experts`
            // rows of `N * K` the entry is handed.
            match bank_shape.as_slice() {
                [experts, bank_n, bank_k] => {
                    if *bank_n != n || *bank_k != k {
                        faults.push(format!(
                            "`{}`: a routed bank of {bank_shape:?} against a {k}-wide \
                             activation and a {n}-wide result. The grouped leg wants \
                             `[experts, {n}, {k}]`, which reaches it as rows of \
                             {}; this one falls back to the GEMV and the family \
                             keeps the per-route read",
                            row.name,
                            n * k,
                        ));
                    } else if *experts == 0 {
                        faults.push(format!("`{}`: a routed bank of no experts", row.name));
                    }
                }
                other => faults.push(format!(
                    "`{}`: a routed bank declared {other:?}, not the three-axis \
                     `[experts, N, K]` the grouped leg divides",
                    row.name,
                )),
            }
        }
    }

    assert!(
        checked > 0,
        "no catalog SKU traces a routed select, so this gate proved nothing — either the \
         MoE families left the catalog or the op was renamed"
    );
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
