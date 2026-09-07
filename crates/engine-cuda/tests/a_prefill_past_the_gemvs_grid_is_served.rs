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
use std::sync::{Mutex, MutexGuard, PoisonError};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Dtype, Platform, Request};
use model_ir::{ParamSource, Trace};

/// `a3b_micro`'s routed fan-out, and the token count whose route run passes
/// the GEMV's grid ceiling: `16_400 * 4 = 65_600`, and the ceiling is 65535.
const TOP_K: u32 = 4;
const WIDE: u32 = 16_400;

/// A width BOTH legs serve, for the arms that compare them: 32768 routes is
/// under the GEMV's ceiling, and 1024 routes per expert is well over the
/// four the grouped leg asks for.
const BOTH: u32 = 8_192;

/// One shell at a time per process: `kernels-cuda`'s scratch slabs are
/// process-global, and these tests set `PIE_NO_MOE_GROUP` around a fire.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

const PAGE: u32 = 16;
/// One synthetic routed text, with the two `fn` items a shell wants for it.
/// `classify` has to be a plain function pointer, so each text carries its
/// own pair rather than a closure over a `Dims`.
struct Text {
    name: &'static str,
    build: fn() -> models::qwen_3::model::Model,
    classify: model_ir::ClassifyFn,
    word: fn(u32) -> u64,
    /// The widest fire this text is planned for, and the context to match.
    ceiling: u32,
}

fn micro() -> models::qwen_3::model::Model {
    models::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1)
}
fn micro_classify(request: &Request) -> u64 {
    model_dsl::word_of(micro, request)
}
fn micro_word(len: u32) -> u64 {
    model_dsl::word_of(micro, &Request::new(len, false))
}

fn uncached() -> models::qwen_3::model::Model {
    models::qwen_3::model::Model::a3b_uncached_bank(Dtype::Bf16, Dtype::Bf16, 1)
}
fn uncached_classify(request: &Request) -> u64 {
    model_dsl::word_of(uncached, request)
}
fn uncached_word(len: u32) -> u64 {
    model_dsl::word_of(uncached, &Request::new(len, false))
}

/// Four layers, 32 experts, a bank of a few megabytes — small enough to sit
/// in a card's last-level cache.
const MICRO: Text = Text {
    name: "a3b_micro",
    build: micro,
    classify: micro_classify,
    word: micro_word,
    ceiling: WIDE,
};

/// One layer, the same 32 experts over a 2048-wide trunk: a 134 MiB gate/up
/// bank, past the 48 MiB an L40S carries.
const UNCACHED: Text = Text {
    name: "a3b_uncached_bank",
    build: uncached,
    classify: uncached_classify,
    word: uncached_word,
    ceiling: BOTH,
};

/// A scratch directory of this process's own, collected however the test
/// leaves — the container under it is tens of megabytes.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("pie-moe-{tag}-{}", std::process::id()));
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
    text: Text,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture(text: Text) -> Fixture {
    let m = (text.build)();
    let trace = model_dsl::trace_hybrid(text.name, &m, Platform::Cuda);
    let dir = scratch(text.name);
    let container = dir.0.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        text,
        trace,
        contract,
        container,
        _dir: dir,
    }
}

fn load(fixture: &Fixture) -> engine_cuda::Result<Shell> {
    Shell::load(Boot {
        classify: fixture.text.classify,
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, fixture.text.ceiling),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: fixture.text.ceiling.next_multiple_of(PAGE),
        // Two: one seat warms the fire's width, the other is the one timed.
        slots: 2,
        pages: 2 * (fixture.text.ceiling.next_multiple_of(PAGE) / PAGE),
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

/// One prefill of `tokens` on a fresh shell, and what it cost.
fn fire_at(fixture: &Fixture, tokens: u32) -> engine_cuda::Result<(Vec<Vec<f32>>, f64)> {
    let mut shell = load(fixture)?;
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let prompt: Vec<u32> = (0..tokens).map(|t| (t * 7 + 11) % 2048).collect();
    let word = (fixture.text.word)(tokens);
    // **WARM AT THE WIDTH THAT IS TIMED.** A narrow warm fire is a GEMV
    // fire: it takes none of the grouped leg's scratch slabs, so their first
    // allocation would land inside the measurement.
    shell.fire(&[Lane {
        slot: 0,
        word,
        tokens: &prompt,
    }])?;
    let start = std::time::Instant::now();
    let rows = shell.fire(&[Lane {
        slot: 1,
        word,
        tokens: &prompt,
    }])?;
    Ok((rows, start.elapsed().as_secs_f64() * 1e3))
}

/// The same, with the grouped leg declined.
///
/// SAFETY: the caller holds [`serialized`], and the variable is read at fire
/// time by `linear::moe::matmul_select` and nowhere else.
fn fire_at_ungrouped(
    fixture: &Fixture,
    tokens: u32,
) -> engine_cuda::Result<(Vec<Vec<f32>>, f64)> {
    unsafe { std::env::set_var("PIE_NO_MOE_GROUP", "1") };
    let answer = fire_at(fixture, tokens);
    unsafe { std::env::remove_var("PIE_NO_MOE_GROUP") };
    answer
}

#[test]
fn a_prefill_past_the_gemvs_grid_is_served() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(MICRO);
    assert_eq!(
        WIDE * TOP_K > 65_535,
        true,
        "the fixture's route run has to pass the GEMV's grid ceiling or this proves nothing"
    );

    // (1) The grouped leg serves it.
    let (logits, _) = fire_at(&fixture, WIDE).expect(
        "a prefill of 65600 routes fires — if this refused, the engine never reached the \
         grouped leg and the routed matmul is still one GEMV per route",
    );
    finite(&logits[0], "the wide prefill");

    // (2) And without it, the same fire is the GEMV's own refusal. This is
    // what makes (1) a statement about the grouped leg rather than about
    // some narrower width the GEMV would have served anyway.
    //
    let why = fire_at_ungrouped(&fixture, WIDE)
        .err()
        .expect("with the grouped leg declined, a 65600-route fire is past the GEMV's grid")
        .to_string();
    assert!(
        why.contains("65600") || why.contains("65535"),
        "the refusal should be the GEMV's, naming the route run against the grid it does \
         not fit; it said: {why}"
    );
}

/// **THE TWO LEGS ANSWER THE SAME MODEL**, read through the whole stack
/// rather than at the entry.
///
/// `kernels-cuda` checks the grouped leg's arithmetic against a host dot on
/// one op. This checks it where it actually lives: four layers, a router
/// that picks the experts, both legs of the MLP, and a readout — the same
/// prompt through the same weights, once each way.
///
/// They do not agree bit for bit and must not be asked to: a batched GEMM
/// accumulates in a different order from a per-route GEMV, and the logits
/// carry that. The tolerance below is on the SPREAD of the logits, so it
/// does not quietly pass a rectangle of noise.
///
/// It prints both prices too. One synthetic four-layer text is not a serving
/// number and is not offered as one; what it is good for is noticing if the
/// grouped leg ever becomes the slower way to answer.
#[test]
fn the_two_legs_answer_the_same_model() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(MICRO);

    let (grouped, grouped_ms) = fire_at(&fixture, BOTH).expect("the grouped leg fires");
    let (gemv, gemv_ms) = fire_at_ungrouped(&fixture, BOTH).expect("the per-route GEMV fires");
    finite(&grouped[0], "grouped");
    finite(&gemv[0], "gemv");

    assert_eq!(
        grouped[0].len(),
        gemv[0].len(),
        "the two legs read out different vocabularies"
    );
    let spread = gemv[0].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - gemv[0].iter().copied().fold(f32::INFINITY, f32::min);
    let worst = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut gaps: Vec<f32> = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .collect();
    gaps.sort_by(f32::total_cmp);
    let median = gaps[gaps.len() / 2];
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    let pick = |row: &[f32]| {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (at, v)| {
                if *v > best.1 { (at, *v) } else { best }
            })
            .0
    };
    eprintln!(
        "{BOTH} tokens, {} routes: grouped {grouped_ms:.1} ms, gemv {gemv_ms:.1} ms \
         ({:.2}x); gap worst {worst:.4} mean {mean:.4} median {median:.4} over a spread \
         of {spread:.2}; argmax {} vs {}",
        BOTH * TOP_K,
        gemv_ms / grouped_ms,
        pick(&grouped[0]),
        pick(&gemv[0]),
    );
    // **WHAT A REORDERED SUM LOOKS LIKE**, and what it does not. A batched
    // GEMM accumulates in a different order from a per-route GEMV, so the
    // logits differ — but the difference is spread over the whole vocabulary
    // rather than sitting in one place: the mean and the median land on top
    // of each other, and the worst is a small multiple of them. A leg that
    // dropped a route, read a bank at the wrong stride or landed a block in
    // the wrong rows would move the mean, not just the tail, so that is what
    // is bounded tightest here. The worst gets the looser bound because one
    // near-tie rounding the other way is not a fault.
    assert!(
        mean <= 0.01 * spread,
        "the two legs answer differently in the MEAN: {mean} over a spread of {spread}. \
         A reordered sum moves the tail, not the middle"
    );
    assert!(
        worst <= 0.05 * spread,
        "one logit differs by {worst} over a spread of {spread}, which is more than a \
         near-tie rounding the other way"
    );
}

/// **THE SAME COMPARISON WHERE THE BANK DOES NOT FIT IN CACHE**, which is
/// the case the whole change is about.
///
/// [`the_two_legs_answer_the_same_model`] runs on a bank of a few megabytes.
/// A serving card holds that in its last-level cache, so the per-route
/// GEMV's re-reads never reach memory and the thing grouping removes is not
/// there to remove. This text's gate/up bank is 134 MiB against an L40S's
/// 48 MiB, so a re-read is a fetch, and the ratio here is the one that says
/// what grouping is for.
///
/// Still not a serving number — one layer, synthetic weights, one fire. What
/// it is is the claim measured through the engine rather than at the entry.
#[test]
fn a_bank_past_the_cache_is_where_grouping_pays() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(UNCACHED);

    let (grouped, grouped_ms) = fire_at(&fixture, BOTH).expect("the grouped leg fires");
    let (gemv, gemv_ms) = fire_at_ungrouped(&fixture, BOTH).expect("the per-route GEMV fires");
    finite(&grouped[0], "grouped");
    finite(&gemv[0], "gemv");

    let spread = gemv[0].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - gemv[0].iter().copied().fold(f32::INFINITY, f32::min);
    let mean = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / grouped[0].len() as f32;

    eprintln!(
        "{} tokens, {} routes over a 134 MiB bank: grouped {grouped_ms:.1} ms, \
         gemv {gemv_ms:.1} ms ({:.1}x); mean logit gap {mean:.4} over a spread of {spread:.2}",
        BOTH,
        BOTH * TOP_K,
        gemv_ms / grouped_ms,
    );
    assert!(
        mean <= 0.01 * spread,
        "the two legs answer differently in the MEAN: {mean} over a spread of {spread}"
    );
    // Not a threshold on the speedup — a slower grouped leg here would mean
    // the dispatch is buying nothing where it was built to buy the most, and
    // that is worth failing over even though the exact ratio is not.
    assert!(
        grouped_ms < gemv_ms,
        "grouping a bank that does not fit in cache took {grouped_ms:.1} ms against the \
         per-route GEMV's {gemv_ms:.1} ms; this is the shape it exists for"
    );
}
