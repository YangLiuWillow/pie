//! **THE CATALOG'S `mini-dit-bf16-kv-bf16` ROW — whose adaLN is an f32
//! LANE CHAIN (`sinusoid → silu → matmul → add_bias → split_rows →
//! modulate`) — LOADS UNDER THE SERVING KNOBS: every body the arming pass
//! records answers what its own eager walk answers.**
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test the_mini_dit_row_arms_its_bodies -- --nocapture
//! ```
//!
//! Reads the artifact at `$PIE_MINI_DIT_ARTIFACT` (default
//! `/root/.cache/pie-imagegen/mini-dit.zt`) and skips by name when it is
//! absent or no device is present.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use model_dsl::{Platform, Trace};

const SKU: &str = "mini-dit-bf16-kv-bf16";

fn artifact() -> Option<PathBuf> {
    let path = std::env::var("PIE_MINI_DIT_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.cache/pie-imagegen/mini-dit.zt"));
    path.is_file().then_some(path)
}

fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor_compat::index(path).map_err(|why| why.to_string())?;
    checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

#[test]
fn every_armed_body_answers_its_eager_walk() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Some(path) = artifact() else {
        eprintln!("not asked: no mini-dit artifact (PIE_MINI_DIT_ARTIFACT)");
        return;
    };
    let sku = models::sku(SKU).expect("the catalog ships the mini-dit row");
    let trace = (sku.trace)(Platform::Cuda);
    let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for, |name| {
        models::sku(name).map(|sku| sku.classify)
    })
    .expect("the engine opens");
    let loaded = engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets::default(),
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 2,
        })
        .expect("the row loads: every armed body agrees with its eager walk");
    assert!(loaded.caps.profile.has_velocity);
}
