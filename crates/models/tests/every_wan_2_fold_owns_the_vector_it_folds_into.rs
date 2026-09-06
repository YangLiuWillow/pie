//! **NO WAN 2.2 NODE FOLDS IN PLACE INTO A VECTOR A LATER NODE STILL
//! READS — THE THIRTY BLOCKS ADD THEIR `scale_shift_table` TO A COPY OF
//! THE ONE SHARED `timestep_proj`, NOT TO THE PROJECTION ITSELF.**
//!
//! ```text
//! cargo test -p models --test every_wan_2_fold_owns_the_vector_it_folds_into
//! ```
//!
//! `elementwise.add_bias` folds its bias IN PLACE: the IR aliases
//! `out_out` onto `out` (`model_ir::ops::elemwise`, `Operands::aliases`),
//! so the arena hands both ids one column and the node overwrites its
//! operand. That is exactly what a biased projection wants — the matmul
//! output it folds into is its own. It is exactly what a shared
//! conditioning vector does NOT want: `denoise` computes
//! `time_proj(silu(temb))` once a fire and every block adds its own
//! `scale_shift_table` to it, so folding into the projection itself
//! leaves block `k` modulating by `timestep_proj + sum(table_0..table_k)`.
//! That reads exact at one block and drifts with every block after: the
//! flagship answered cos 0.274 against the diffusers `dit.step0.out`
//! while the two-block miniature still answered 0.9999, which is why the
//! claim below is about the PLAN and not about a tolerance.
//!
//! The claim is general because the hazard is: any in-place fold whose
//! operand a later node reads is a wrong answer, silently, and the plan
//! says so without a GPU.

use model_dsl::{Operands, Platform, Trace, ValueId};

const ROWS: [&str; 3] = [
    "wan22-ti2v-5b-bf16-kv-bf16",
    "wan22-mini-d128-bf16-kv-bf16",
    "wan22-mini-nano-bf16-kv-bf16",
];

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn trace(sku: &str, platform: Platform) -> Trace {
    let row = models::sku(sku).unwrap_or_else(|| panic!("this build ships no `{sku}`"));
    (row.trace)(platform)
}

/// The last node that lists `id` among its inputs, per value.
fn last_read(t: &Trace) -> Vec<Option<usize>> {
    let mut last = vec![None; t.values.len()];
    let mut ins = Vec::new();
    for (j, node) in t.nodes.iter().enumerate() {
        ins.clear();
        node.op.inputs(&mut ins);
        for &ValueId(id) in &ins {
            last[id as usize] = Some(j);
        }
    }
    last
}

#[test]
fn every_wan_2_fold_owns_the_vector_it_folds_into() {
    for sku in ROWS {
        for platform in PLATFORMS {
            let t = trace(sku, platform);
            let last = last_read(&t);
            let mut pairs = Vec::new();
            let mut folds = 0usize;
            for (j, node) in t.nodes.iter().enumerate() {
                pairs.clear();
                node.op.aliases(&mut pairs);
                for &(_out, ValueId(input)) in &pairs {
                    folds += 1;
                    assert!(
                        last[input as usize] <= Some(j),
                        "{sku}/{platform:?}: node {j} `{}` folds in place into v{input}, \
                         which node {:?} reads afterwards — an in-place fold must own \
                         its operand (add to a copy, as `forward::copy_of` does)",
                        node.op.name(),
                        last[input as usize],
                    );
                }
            }
            assert!(
                folds > 0,
                "{sku}/{platform:?}: this plan folds nothing in place, so the claim \
                 is vacuous and the walk above must be wrong"
            );
        }
    }
}
