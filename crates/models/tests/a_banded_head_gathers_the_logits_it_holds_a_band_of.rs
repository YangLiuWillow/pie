//! The vocab-parallel readout's one invariant, checked without a checkpoint.
//!
//! A head banded across ranks lands only THIS rank's columns of the logits, so
//! every readout through one owes an `all_gather` before anything reads the
//! result. Miss it on one model and that model is silently wrong at tp > 1 —
//! the plan reads a shard as if it were the whole vocabulary, and no shape
//! check catches it because the declared width is what the gather would have
//! produced.
//!
//! Six models band their head today and only one of them (gemma-4) has a
//! checkpoint small enough to run here, so this is the check that covers the
//! other five, and the one that will catch a seventh model added later.

use model_dsl::{Collective, Def, Linear, Operation, Platform, Shard, Trace, ValueId};

/// Is `w` a weight the loader bands across ranks — cut on the vocabulary axis?
fn banded(trace: &Trace, w: ValueId) -> bool {
    match trace.values[w.0 as usize].def {
        Def::Weight(at) => matches!(
            trace.params[at as usize].shard,
            Shard::Cut { axis: 0, .. }
        ),
        _ => false,
    }
}

#[test]
fn a_banded_head_gathers_the_logits_it_holds_a_band_of() {
    let mut faults = Vec::new();

    for row in models::skus() {
        let trace = (row.trace)(Platform::Cuda);

        // Every value some `all_gather` consumes.
        let gathered: Vec<ValueId> = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Collective(Collective::AllGather { x, .. }) => Some(*x),
                _ => None,
            })
            .collect();

        for node in &trace.nodes {
            let Operation::Linear(Linear::LmHead { w, y, .. }) = &node.op else {
                continue;
            };
            if banded(&trace, *w) && !gathered.contains(y) {
                faults.push(format!(
                    "`{}` reads out through a vocab-banded head and nothing gathers \
                     the result: this rank lands only its columns of the logits, so \
                     the readout owes an `all_gather` before anything reads them",
                    row.name,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

/// The mirror: a single rank has no band, so nothing may be cut and nothing
/// gathered. Catches a banding predicate that forgot to ask about `tp`.
#[test]
fn a_single_rank_bands_nothing_and_gathers_nothing() {
    let mut faults = Vec::new();

    for row in models::skus() {
        if row.recipe.tp > 1 {
            continue;
        }
        let trace = (row.trace)(Platform::Cuda);

        if let Some(cut) = trace
            .params
            .iter()
            .find(|p| matches!(p.shard, Shard::Cut { axis: 0, .. }) && p.name.contains("head"))
        {
            faults.push(format!(
                "`{}` ships one rank and cuts `{}` on the vocabulary axis; there is \
                 no second rank to hold the other band",
                row.name, cut.name,
            ));
        }

        let gathers = trace
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.op,
                    Operation::Collective(Collective::AllGather { .. })
                )
            })
            .count();
        if gathers != 0 {
            faults.push(format!(
                "`{}` ships one rank and gathers {gathers} time(s); a gather over a \
                 world of one is a copy nobody asked for",
                row.name,
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
