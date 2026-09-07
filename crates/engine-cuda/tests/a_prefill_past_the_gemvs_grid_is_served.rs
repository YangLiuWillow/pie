//! **THE ROUTED MATMUL'S GROUPED LEG, REACHED THROUGH THE ENGINE.**
//!
//! `kernels-cuda` reads the grouped leg's arithmetic against a host dot. What
//! that cannot say is whether a real fire ever gets there: the entry declines
//! for a streaming bank, under a body's staged geometry, and for a shape it
//! does not recognise, and every one of those declines is SILENT — it falls
//! back to the per-route GEMV, which is a slower right answer.
//!
//! So this fires a prefill the GEMV cannot serve at all. Its route run rides
//! the grid's y axis, which stops at 65535, and 16400 tokens at a fan-out of
//! four is 65600 routes. The claim is a pair:
//!
//!   * with the grouped leg, the fire lands and its logits are a rectangle
//!     something wrote;
//!   * with `PIE_NO_MOE_GROUP` set, the SAME fire is refused, and the refusal
//!     is the GEMV's own — naming the route run and the grid it does not fit.
//!
//! The second half is what makes the first mean anything. A test that only
//! fired would pass just as well if the engine had quietly stayed on the
//! GEMV at some narrower width, or if the model had no routed select at all.
//!
//! The model is `a3b_micro` — qwen a3b's routed text scaled down (four
//! layers, 32 experts, top-4, a 2048-token vocabulary), whose checkpoint no
//! one ships. It is written here from the trace's own params and read back
//! through `checkpoint_dsl::own_contract`, which is the door for a container
//! holding a plan's weights under the plan's own names.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Dtype, Platform, Request};
use model_ir::{ParamSource, Trace};

/// `a3b_micro`'s routed fan-out, and the token count whose route run passes
/// the GEMV's grid ceiling: `16_400 * 4 = 65_600`, and the ceiling is 65535.
const TOP_K: u32 = 4;
const WIDE: u32 = 16_400;

const PAGE: u32 = 16;
/// Room for the whole prefill, rounded to a whole page.
const CONTEXT: u32 = 16_512;

fn model() -> models::qwen_3::model::Model {
    models::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1)
}

fn word(query_len: u32) -> u64 {
    model_dsl::word_of(model, &Request::new(query_len, false))
}

/// The class word, as the shell's own `classify` field wants it.
fn classify(request: &Request) -> u64 {
    model_dsl::word_of(model, request)
}

/// A scratch directory of this process's own, collected however the test
/// leaves — the container under it is tens of megabytes.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let dir = std::env::temp_dir().join(format!("pie-moe-grouped-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    Scratch(dir)
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// **A CHECKPOINT FOR A TEXT NOBODY SHIPS**, written from the trace's own
/// params under the plan's own names.
///
/// A registered plane is one the checkpoint does NOT have — the shell
/// reserves and zeroes those — so only checkpoint-sourced params are
/// written. Norm scales sit near one and everything else is small: a norm
/// drawn near zero would make the logits a rectangle of noise, which is the
/// thing `finite` below exists to notice.
fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        assert_eq!(
            param.dtype,
            Dtype::Bf16,
            "`{}` is {:?}; this fixture writes a bf16 text",
            param.name,
            param.dtype
        );
        let mut bytes = Vec::with_capacity(count * 2);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
        }
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "{what}: logit {at} is {value}");
    }
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

struct Fixture {
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture() -> Fixture {
    let m = model();
    let trace = model_dsl::trace_hybrid("qwen35-a3b-micro", &m, Platform::Cuda);
    let dir = scratch();
    let container = dir.0.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        trace,
        contract,
        container,
        _dir: dir,
    }
}

fn load(fixture: &Fixture) -> engine_cuda::Result<Shell> {
    Shell::load(Boot {
        classify,
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, WIDE),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 1,
        pages: CONTEXT / PAGE,
        ordinal: 0,
        // The grouped leg declines under a body's staged geometry, and a
        // prefill this wide would never be captured anyway; off, so that the
        // one thing under test is the width and not the arming pass.
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // Full residency: an uncapped table answers `ExpertTable::RESIDENT`,
        // which is the reading the grouped leg accepts.
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
}

/// One prefill of `WIDE` tokens on a fresh shell.
fn wide_fire(fixture: &Fixture) -> engine_cuda::Result<Vec<Vec<f32>>> {
    let mut shell = load(fixture)?;
    shell.open(0).expect("slot 0 opens");
    let prompt: Vec<u32> = (0..WIDE).map(|t| (t * 7 + 11) % 2048).collect();
    shell.fire(&[Lane {
        slot: 0,
        word: word(WIDE),
        tokens: &prompt,
    }])
}

#[test]
fn a_prefill_past_the_gemvs_grid_is_served() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture();
    assert_eq!(
        WIDE * TOP_K > 65_535,
        true,
        "the fixture's route run has to pass the GEMV's grid ceiling or this proves nothing"
    );

    // (1) The grouped leg serves it.
    let logits = wide_fire(&fixture).expect(
        "a prefill of 65600 routes fires — if this refused, the engine never reached the \
         grouped leg and the routed matmul is still one GEMV per route",
    );
    finite(&logits[0], "the wide prefill");

    // (2) And without it, the same fire is the GEMV's own refusal. This is
    // what makes (1) a statement about the grouped leg rather than about
    // some narrower width the GEMV would have served anyway.
    //
    // SAFETY: one test, one thread, and the variable is read at fire time by
    // `linear::moe::matmul_select` and nowhere else.
    unsafe { std::env::set_var("PIE_NO_MOE_GROUP", "1") };
    let refused = wide_fire(&fixture);
    unsafe { std::env::remove_var("PIE_NO_MOE_GROUP") };

    let why = refused
        .err()
        .expect("with the grouped leg declined, a 65600-route fire is past the GEMV's grid")
        .to_string();
    assert!(
        why.contains("65600") || why.contains("65535"),
        "the refusal should be the GEMV's, naming the route run against the grid it does \
         not fit; it said: {why}"
    );
}
