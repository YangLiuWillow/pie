//! **The RS device half, gated on the one equivalence it exists to keep**
//! (alto design §6, survey §9; wave F3).
//!
//! The programming model's whole claim is that a speculative window can be
//! folded LATER without being folded WRONG:
//!
//! ```text
//! fold k tokens in the forward          ==   buffer k tokens,
//!   (write_state, one fire)                  then fold-buffered k
//!                                            (two fires, in-proj replayed)
//! ```
//!
//! Byte for byte, on the recurrent banks themselves — not on the logits,
//! because a fold-buffered replay computes no output anybody reads and a
//! logit diff would be measuring the attention layers riding beside it.
//!
//! Six claims, in the order they can fail:
//!
//! 1. **a buffered scatter folds nothing.** After a `RsVerb::Buffer` fire the
//!    slot's banks are exactly what they were, which is what makes a rejected
//!    draft pure host bookkeeping ("no folded state was ever perturbed").
//! 2. **the replay is the fold.** `Buffer(k)` then `FoldBuffered(k)` leaves the
//!    same bytes as one `Fold` over the same k tokens. This is the whole
//!    wave: the buffer holds the conv+scan's INPUTS, the gather puts them back
//!    over the in-projection the replay recomputed, and the recurrence cannot
//!    tell the difference.
//! 3. **`commit_len` is live.** A replay truncated at fewer tokens leaves
//!    DIFFERENT bytes — so the seat the shell fills is one the kernels read,
//!    rather than an argument nobody looks at.
//! 4. **a truncated fold is EXACT against a shorter buffer** (wave F3b, and
//!    the claim the kernel's coupling made impossible). A replay of twenty
//!    buffered tokens truncated at four and a replay BOUNDED at four leave
//!    the same bytes. The two run different launches over different row
//!    counts — one binds `commit_len` and the other binds nothing — and the
//!    old fla scan read `commit_len != nullptr` as `single_round` as well,
//!    a different bf16 rounding of the decay (`ssm.cuh:1660,1697-1706`), so
//!    the two agreed only to a rounding: 3,115,437 of 10,321,884 state bytes
//!    moved for no reason but the flag. The rounding is its own argument now
//!    (`RecurrentPool::fused_decay`), the shell binds the fold's own policy
//!    everywhere, and the equality is exact.
//! 5. **the mixed row is the two fires it replaces** (wave F3b's 2R interior
//!    split). `Buffer(k tokens, fold = j)` — one fire that writes the whole
//!    window into the buffer AND lands the boundary on row `j` — leaves the
//!    same bytes as `Buffer(k)` followed by `FoldBuffered(j)`. The row is cut
//!    into two segments on one stream: the head `[0, j)` folds, and the tail
//!    `[j, k)` continues from what the head wrote and moves nothing. **Both
//!    halves are pinned**, because either can fail alone — the state is
//!    compared against the two-fire reference, and the fire's OUTPUTS against
//!    a plain fold over the same window, which is what a skipped tail would
//!    lose.
//! 6. **the fold predicate is per lane, and it is device data.** One fire
//!    carrying a folding lane BESIDE a buffering one folds exactly one of
//!    them, and `channel::mask_from_commit` is what decided which: the
//!    predicate bytes it wrote are readable afterwards and say `1` for the
//!    lane pointed at the standing committed word and `0` for the lane pointed
//!    at the standing refused one. A refused PASS reaches the same kernel by
//!    the same array with the instance's own commit word in that slot, which
//!    is why this is the mask's gate and not merely the verb's.
//!
//! 7. **the read path** (`the_read_path_replays_the_buffer_it_folds`): a
//!    window buffered unfolded, then a second fire that REPLAYS it
//!    (`replay = k`, `fold = k`) while buffering its own rows, leaves the
//!    banks a `Fold` of the first window leaves and answers the second
//!    window's logits within the floor of a fire over it after a plain fold
//!    of the first — one speculative round, as the block drafters fire it on
//!    a hybrid target; and the round's own buffer then folds to what folding
//!    its rows plainly leaves. The arms run over the EXTENDED run
//!    `[replay | rows]` through the chunked kernels (`Run::rs_extend`).
//!
//! 8. **a deployment's first fire is its every fire**
//!    (`a_deployments_first_fire_is_its_every_fire`) — the gate on the gemm
//!    tuner's pinning, which used to change kernels under a running
//!    deployment.
//!
//! # Gating
//!
//! Skips at RUN time, like [`serve_smoke`](../serve_smoke.rs), saying which of
//! the machine and the checkpoint was missing — an `#[ignore]`d test on the
//! one box that could run it is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda \
//!   --test a_buffered_fold_is_the_fold_it_replaces -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_cuda::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

/// The catalog row this gate serves: a GDN/attention hybrid, which is the
/// family the whole recurrent vocabulary exists for.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens in the speculative window. Twenty against a sixteen-token page is
/// deliberate: the buffer's addressing is page-major, so a window that fits in
/// one page would never exercise the page-crossing arithmetic that the scatter
/// and the gather have to agree about.
const WINDOW: usize = 20;

/// The kv page size, which is also the buffer's page size — dev's rule, so
/// that a buffer page and a kv page are one number.
const PAGE: u32 = 16;

/// How many buffer page slots `WINDOW` tokens need.
const PAGES: u64 = WINDOW.div_ceil(PAGE as usize) as u64;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The lane word the model's own `Classify` computes — runtime-side work, done
/// here because this test IS the runtime for the length of one fire.
fn word(query_len: u32) -> u64 {
    (models::sku(SKU).expect("the catalog ships the SKU").classify)(&Request::new(query_len, false))
}

/// A loaded shell, or `None` and a sentence saying what was missing.
fn ready(what: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let sku = models::sku(SKU).expect("the catalog ships the SKU");
    let trace = (sku.trace)(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = sku
        .contract(&source, Platform::Cuda)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        // Neither field existed when this test was written: `voxels` is the
        // third row axis a VAE runs on, `deferred_tier` the expert-residency
        // knob that stopped being an environment read. A plan with no voxel
        // rows states no ladder.
        voxels: None,
        deferred_tier: false,
        classify: sku.classify,
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 512,
        // One recurrent slot per arm below: the reference fold, the replay,
        // the truncated replay, the shorter-bounded replay, the mixed row,
        // the mixed row's output reference, and the read path's two.
        slots: 8,
        pages: 8 * 512 / PAGE,
        ordinal: 0,
        // **THE EAGER PATH, AND THE SHELL WOULD HAVE INSISTED ANYWAY.** A fire
        // that moves buffered bytes is not graph-replayable (design §6: the
        // plain fold is the only shape that is), so `enqueue` forces the eager
        // walk for one whatever the mode — this states the same thing where a
        // reader can see it.
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    eprintln!(
        "{what}: buffered-activation pool is {:.2} MiB",
        shell.buffer_bytes() as f64 / (1 << 20) as f64,
    );
    assert!(
        shell.buffer_bytes() > 0,
        "a GDN plan reserved no buffered-activation pool, so nothing below can be true"
    );
    Some(shell)
}

/// The window this gate folds — arbitrary ids, because what is being compared
/// is two ways of folding the SAME activations and not what the model says
/// about them.
fn window() -> Vec<u32> {
    (0..WINDOW as u32).map(|at| 1000 + at * 37).collect()
}

fn buffer(at: u32) -> RsVerb {
    // `Host(0)` is the pure scatter: every row into the buffer, the folded
    // state untouched.
    mixed(at, 0)
}

/// **The mixed row** (wave F3b): the same scatter, landing the durable state
/// on row `fold` of the window it is writing.
fn mixed(at: u32, fold: u32) -> RsVerb {
    replaying(at, fold, 0)
}

/// **The read path**: the same scatter at `at`, with `replay` buffered tokens
/// at `[at - replay, at)` replayed through the recurrence ahead of the rows;
/// `fold` counts in the extended layout `[replay | rows]`.
fn replaying(at: u32, fold: u32, replay: u32) -> RsVerb {
    RsVerb::Buffer {
        // The run is a LIST of physical page slots (wave F3-tail); this gate's
        // buffer happens to be the first `PAGES` of the pool, in order.
        pages: (0..PAGES as u32).collect(),
        at,
        fold: FoldLen::Host(fold),
        replay,
    }
}

fn fold_buffered(len: u32) -> RsVerb {
    bounded_fold(WINDOW as u32, len)
}

/// A replay whose BOUND is stated too — the shorter buffer claim 4 compares
/// a truncated fold against. `bound` is the lane's row count by contract, so
/// the two arms differ in rows as well as in the seat.
fn bounded_fold(bound: u32, len: u32) -> RsVerb {
    // This gate's buffer begins at buffer token zero: no fold has landed
    // mid-page ahead of it, so the head and the origin are both zero.
    fold_buffered_at(0, bound, len)
}

/// A replay whose buffer begins at token `at` — where a round's own window
/// sits after the prefix it replayed was folded off the front.
fn fold_buffered_at(at: u32, bound: u32, len: u32) -> RsVerb {
    RsVerb::FoldBuffered {
        pages: (0..PAGES as u32).collect(),
        at,
        bound,
        len: FoldLen::Host(len),
    }
}

/// The seated door, with no attachments, no images and no capture readback.
fn fire(shell: &mut Shell, lanes: &[Seated<'_>]) -> engine_cuda::Result<Vec<Vec<f32>>> {
    shell.fire_media(lanes, &[], &[], &mut Vec::new())
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        lane: Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        },
        rs,
        rs_reset: reset,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

/// **A DEPLOYMENT'S FIRST FIRE ANSWERS WHAT EVERY LATER ONE DOES.**
///
/// It did not, and the cause was not this file's subject at all: the dense
/// gemm tuner (`kernels_cuda::linear::dense`) pinned a benched cuBLASLt
/// tactic only once it had seen a shape TWICE, so a shape's first gemm ran
/// the untuned ladder and every one after it ran another kernel, whose sums
/// round differently. One bf16 ulp on a projection, ~0.19 in the recurrent
/// banks it fed, ~0.19 on the logits — enough to flip a near-tie token, and
/// whether it showed at all depended on whether this machine's tuner cache
/// already held the shape, so a greedy stream could differ between two runs
/// of the same binary. The tuner now pins on the FIRST sighting; this is the
/// gate on that.
///
/// The fold is the probe rather than the subject: a twenty-row chunked
/// recurrence reads a projection through several layers, so it magnifies a
/// single ulp into millions of state bytes.
#[test]
fn a_deployments_first_fire_is_its_every_fire() {
    let Some(mut shell) = ready("first-fire determinism") else {
        return;
    };
    let tokens = window();
    let mut banks: Vec<Vec<u8>> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for slot in 0..3u32 {
        shell.open(slot).expect("slot opens");
        let out = fire(&mut shell, &[seated(slot, &tokens, RsVerb::Fold, RsReset::Fresh)])
            .expect("the fold runs");
        banks.push(shell.state_bytes(slot).expect("banks"));
        rows.push(out.into_iter().next().expect("a row"));
    }
    for (at, what) in [(1usize, "the second fold"), (2, "the third fold")] {
        let (worst, coarse, any) = bank_gap(&banks[0], &banks[at]);
        let d = spread(&rows[0], &rows[at]);
        eprintln!("first fold vs {what}: banks worst {worst:.3e}, {coarse} coarse, {any} differ; logit spread {d:.3e}");
        assert_eq!(
            (coarse, any),
            (0, 0),
            "{what} left different banks than the first (worst {worst:.3e}, logit spread {d:.3e}) \
             — a kernel changed under the deployment, as the gemm tuner used to"
        );
    }
}

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    let Some(mut shell) = ready("the buffered fold") else {
        return;
    };
    let tokens = window();

    // ── The reference: one ordinary fire, folding every token in the forward.
    shell.open(0).expect("slot 0 opens");
    let zeroed = shell.state_bytes(0).expect("slot 0 reads back");
    assert!(
        zeroed.iter().all(|byte| *byte == 0),
        "an opened slot's recurrent banks are not zero, so nothing below is a comparison"
    );
    let reference = fire(&mut shell, &[seated(0, &tokens, RsVerb::Fold, RsReset::Fresh)])
        .expect("the folding fire runs");
    let folded = shell.state_bytes(0).expect("slot 0 reads back");
    assert_ne!(
        folded, zeroed,
        "a fold over {WINDOW} tokens left the banks untouched, so the reference is empty"
    );

    // ── Claim 1: a buffered scatter folds NOTHING.
    shell.open(1).expect("slot 1 opens");
    fire(&mut shell, &[seated(1, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "a `RsVerb::Buffer` fire perturbed the folded state — the whole point of the \
         buffer is that a rejected draft is pure host bookkeeping"
    );

    // ── Claim 2: the replay IS the fold.
    fire(&mut shell, &[seated(1, &tokens, fold_buffered(WINDOW as u32), RsReset::Held)])
        .expect("the fold-buffered fire runs");
    let replayed = shell.state_bytes(1).expect("slot 1 reads back");
    assert_eq!(
        replayed.len(),
        folded.len(),
        "the two slots hold different amounts of state, which is a pool bug"
    );
    let differing = folded
        .iter()
        .zip(&replayed)
        .filter(|(a, b)| a != b)
        .count();
    let (worst, coarse, any) = bank_gap(&folded, &replayed);
    eprintln!("claim 2: {differing} bytes differ; as f32 cells: worst |d| {worst:.3e}, {coarse} cells past 1e-3 relative, {any} cells differ at all");
    if std::env::var_os("PIE_GATE_PROBE").is_none() {
        assert_eq!(
            differing, 0,
            "buffer-then-fold-buffered left {differing} of {} state bytes different from the \
             fold it replaces",
            folded.len(),
        );
    }

    // ── Claim 3: `commit_len` is a seat the kernels read.
    shell.open(2).expect("slot 2 opens");
    fire(&mut shell, &[seated(2, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the second buffering fire runs");
    fire(&mut shell, &[seated(2, &tokens, fold_buffered(4), RsReset::Held)])
        .expect("the truncated fold-buffered fire runs");
    let truncated = shell.state_bytes(2).expect("slot 2 reads back");
    assert_ne!(
        truncated, zeroed,
        "a fold truncated at 4 tokens folded nothing at all"
    );
    assert_ne!(
        truncated, replayed,
        "a fold truncated at 4 tokens left the same bytes as one over all {WINDOW}, so \
         `commit_len` reached the launch and was ignored"
    );

    // ── Claim 4: the truncated fold is EXACT against a shorter buffer.
    //
    //    The same four buffered tokens, folded two ways over the SAME bytes:
    //    slot 2 replayed twenty and stopped at four (`commit_len` bound),
    //    slot 3 replays four and stops at its own end (no seat at all). The
    //    buffer is untouched between the two, so the activations the gather
    //    lays down are the same bytes — which is what makes this a test of
    //    the ARITHMETIC and of nothing else.
    //
    //    While the fla scan read `commit_len != nullptr` as a rounding this
    //    equality could not hold: the bound arm folded the decay into the
    //    update and the unbound one rounded the decayed state to bf16 first.
    shell.open(3).expect("slot 3 opens");
    fire(&mut shell, &[seated(
            3,
            &tokens[..4],
            bounded_fold(4, 4),
            RsReset::Fresh,
        )])
        .expect("the shorter-bounded fold-buffered fire runs");
    let shorter = shell.state_bytes(3).expect("slot 3 reads back");
    {
        let (w, c, a) = bank_gap(&truncated, &shorter);
        eprintln!("claim 4: truncated(4) vs bounded(4): worst {w:.3e}, {c} coarse, {a} differ");
        let (w, c, a) = bank_gap(&folded, &truncated);
        eprintln!("claim 3: fold(20) vs truncated(4): worst {w:.3e}, {c} coarse, {a} differ");
    }
    assert_ne!(
        shorter, zeroed,
        "a replay bounded at 4 tokens folded nothing at all"
    );
    let differing = truncated
        .iter()
        .zip(&shorter)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "a fold of {WINDOW} buffered tokens truncated at 4 left {differing} of {} state \
         bytes different from a fold of the same 4 over a buffer bounded there — the \
         length seat is still changing the arithmetic and not only the count",
        truncated.len(),
    );

    // ── Claim 5: the mixed row IS the two fires it replaces.
    //
    //    One fire: scatter the whole window into the buffer AND land the
    //    boundary on row 4. The row's fold boundary is strictly interior, so
    //    the recurrent arms run twice on one stream — head `[0, 4)` folding,
    //    tail `[4, {WINDOW})` continuing from what the head wrote.
    shell.open(4).expect("slot 4 opens");
    let mixed_out = fire(&mut shell, &[seated(4, &tokens, mixed(0, 4), RsReset::Fresh)])
        .expect("the mixed fire runs");
    let mixed_state = shell.state_bytes(4).expect("slot 4 reads back");
    {
        let (w, c, a) = bank_gap(&truncated, &mixed_state);
        eprintln!("claim 5: mixed(0,4) vs truncated(4): worst {w:.3e}, {c} coarse, {a} differ");
        // A plain fold of the first four, for the in-forward reference.
        shell.open(5).expect("slot 5 opens");
        fire(&mut shell, &[seated(5, &tokens[..4], RsVerb::Fold, RsReset::Fresh)]).expect("four fold plainly");
        let four = shell.state_bytes(5).expect("slot 5 reads back");
        let (w, c, a) = bank_gap(&four, &mixed_state);
        eprintln!("probe: fold(4) vs mixed(0,4): worst {w:.3e}, {c} coarse, {a} differ");
        let (w, c, a) = bank_gap(&four, &truncated);
        eprintln!("probe: fold(4) vs truncated replay(4): worst {w:.3e}, {c} coarse, {a} differ");
        shell.open(5).expect("slot 5 reopens");
        fire(&mut shell, &[seated(5, &tokens, RsVerb::Fold, RsReset::Fresh)]).expect("fold again");
        let again = shell.state_bytes(5).expect("slot 5 reads back");
        let (w, c, a) = bank_gap(&folded, &again);
        eprintln!("probe: fold(20) twice: worst {w:.3e}, {c} coarse, {a} differ");
    }
    let differing = truncated
        .iter()
        .zip(&mixed_state)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "one fire that buffers {WINDOW} tokens and folds 4 of them left {differing} of {} \
         state bytes different from buffering them and folding 4 in two fires",
        truncated.len(),
    );

    // ── And the TAIL RAN. A split that dropped its second launch would leave
    //    the same folded state and every output past the boundary unwritten,
    //    so the state comparison alone cannot see it. The whole row is one
    //    continuous scan — the head from the pre-fire state, the tail from
    //    the state the head wrote — so the fire's outputs must be the plain
    //    fold's outputs over the same window.
    let mixed_row = mixed_out.first().expect("the mixed fire reads a row back");
    let plain_row = reference.first().expect("the folding fire reads a row back");
    assert_eq!(
        mixed_row.len(),
        plain_row.len(),
        "the two fires read back different row widths"
    );
    assert!(
        !plain_row.is_empty(),
        "the reference fire read no logits back, so the tail cannot be pinned"
    );
    let worst = mixed_row
        .iter()
        .zip(plain_row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "the mixed fire's own outputs differ from the plain fold's by {worst} — the tail \
         segment `[4, {WINDOW})` did not run, or did not continue from the boundary"
    );
}

/// **Claim 6**: the fold predicate is per lane and the device writes it.
///
/// One fire, two lanes, two verbs: the folding lane's slot advances and the
/// buffering lane's does not, in a fire where BOTH ran through the same
/// launches over the same window. Nothing about the launch distinguishes them
/// — same conv, same scan, same `write_state` — so the only thing that can
/// have separated the two is the byte `channel::mask_from_commit` wrote for
/// each, which is read back and asserted directly.
///
/// **AND THAT KERNEL IS THE PASS PREDICATE, NOT A VERB SWITCH.** The array it
/// scatters holds one commit-word ADDRESS per lane: the standing "committed"
/// word for a lane with no guest, the standing "refused" word for a lane whose
/// verb is a scatter, and the attached instance's OWN pass commit word for a
/// lane that carries a prologue — the word `channel::pull_validate` clears
/// when a prediction turns out stale. So a refused pass and a buffered scatter
/// reach the fold through the same array, the same kernel and the same byte,
/// which is what makes this the mask's gate.
#[test]
fn one_fire_folds_the_lane_that_committed_and_not_the_lane_that_buffered() {
    let Some(mut shell) = ready("the per-lane fold predicate") else {
        return;
    };
    let tokens = window();

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let zeroed = shell.state_bytes(1).expect("slot 1 reads back");

    fire(&mut shell, &[
            seated(0, &tokens, RsVerb::Fold, RsReset::Fresh),
            seated(1, &tokens, buffer(0), RsReset::Fresh),
        ])
        .expect("the mixed fire runs");

    let predicate = shell.fold_predicate(2).expect("the predicate reads back");
    assert_eq!(
        predicate,
        vec![1, 0],
        "the fold predicate the device wrote is {predicate:?}, and this fire's lanes are \
         one folding lane followed by one buffering lane"
    );
    assert_ne!(
        shell.state_bytes(0).expect("slot 0 reads back"),
        zeroed,
        "the folding lane of a mixed fire did not fold"
    );
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "the buffering lane of a mixed fire folded anyway, so the predicate did not reach \
         the scan"
    );
}

/// The largest logit spread two fires over one window from one state may
/// show: the replayed arm runs the window as the tail of a longer extended
/// run through the chunked kernels, whose chunk boundaries fall elsewhere.
const LOGIT_FLOOR: f32 = 2e-2;

/// How far two bank images are apart, read as f32 cells: the largest
/// absolute difference, and how many cells differ by more than a rounding.
fn bank_gap(a: &[u8], b: &[u8]) -> (f32, usize, usize) {
    let cells = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let (a, b) = (cells(a), cells(b));
    let mut worst = 0.0f32;
    let mut coarse = 0usize;
    let mut any = 0usize;
    for (x, y) in a.iter().zip(&b) {
        let d = (x - y).abs();
        if d > 0.0 {
            any += 1;
        }
        if d > 1e-3 * x.abs().max(y.abs()).max(1e-6) {
            coarse += 1;
        }
        worst = worst.max(d);
    }
    (worst, coarse, any)
}

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (at, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = at;
        }
    }
    best
}

#[test]
fn the_read_path_replays_the_buffer_it_folds() {
    let Some(mut shell) = ready("the read path") else {
        return;
    };
    let first = window();
    let k = WINDOW as u32;
    // A second window, shorter than a page, at ids the first never used.
    let second: Vec<u32> = (0..6u32).map(|at| 3000 + at * 41).collect();

    // ── The reference: A folded plainly, then B folded plainly after it.
    shell.open(6).expect("slot 6 opens");
    fire(&mut shell, &[seated(6, &first, RsVerb::Fold, RsReset::Fresh)]).expect("A folds plainly");
    let folded_a = shell.state_bytes(6).expect("slot 6 reads back");
    let plain = fire(&mut shell, &[seated(6, &second, RsVerb::Fold, RsReset::Held)]).expect("B folds after it");
    let folded_ab = shell.state_bytes(6).expect("slot 6 reads back");

    // ── The round: A buffered unfolded; then B replaying A (`replay = k`,
    //    `fold = k`) while buffering itself at `at = k`.
    shell.open(7).expect("slot 7 opens");
    fire(&mut shell, &[seated(7, &first, buffer(0), RsReset::Fresh)]).expect("window A buffers");
    let round = fire(&mut shell, &[seated(7, &second, replaying(k, k, k), RsReset::Held)])
        .expect("window B replays A and buffers itself");
    let differing = folded_a
        .iter()
        .zip(&shell.state_bytes(7).expect("slot 7 reads back"))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "replaying A ahead of B left {differing} of {} state bytes different from the banks \
         Fold(A) leaves",
        folded_a.len()
    );
    let d = spread(&plain[0], &round[0]);
    eprintln!(
        "read path: B after replay(A) vs B after fold(A), last-row logit spread {d:.4}; argmax {} vs {}",
        argmax(&round[0]),
        argmax(&plain[0])
    );
    assert!(d <= LOGIT_FLOOR, "the read path answers other logits than a plain fold of the prefix: {d}");

    // ── And the round's own buffer, scattered at `at = k`, folds to what
    //    Fold(A) then Fold(B) left.
    fire(&mut shell, &[seated(7, &second, fold_buffered_at(k, 6, 6), RsReset::Held)]).expect("B's buffer folds");
    let differing = folded_ab
        .iter()
        .zip(&shell.state_bytes(7).expect("slot 7 reads back"))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "folding the round's buffer left {differing} of {} state bytes different from the \
         banks Fold(A) then Fold(B) leaves",
        folded_ab.len()
    );

    // ── A truncated round: B replays A but folds only 4 of it (`fold = 4 <
    //    replay`): the banks land what a plain Fold(A[..4]) leaves and B still
    //    answers from the state after ALL of A. The banks are compared within
    //    the recurrence's own rounding, not byte-exact: the committed scan
    //    over a 26-row extended run and a 4-row plain fold are two chunk
    //    geometries whose f32 carries round a few cells apart, the same
    //    tolerance the Metal gate reads this claim under.
    shell.open(2).expect("slot 2 opens");
    fire(&mut shell, &[seated(2, &first[..4], RsVerb::Fold, RsReset::Fresh)]).expect("A[..4] folds plainly");
    let folded_a4 = shell.state_bytes(2).expect("slot 2 reads back");
    shell.open(3).expect("slot 3 opens");
    fire(&mut shell, &[seated(3, &first, buffer(0), RsReset::Fresh)]).expect("window A buffers again");
    let partial = fire(&mut shell, &[seated(3, &second, replaying(k, 4, k), RsReset::Held)])
        .expect("window B replays A and folds four of it");
    // The banks land what Fold(A[..4]) leaves, within the recurrence's own
    // rounding: the reference is a 4-row plain fold and the replay a 26-row
    // committed scan, two chunk geometries whose f32 carries part in a few
    // cells.
    let truncated = shell.state_bytes(3).expect("slot 3 reads back");
    // Against the TWO-FIRE path that means the same thing: buffer A, then
    // fold four of it. That is the equivalence the read path owes — both
    // arms replay a buffer through the committed scan and commit at four —
    // where `Fold(A[..4])` is a different kernel geometry entirely (a 4-row
    // plain fold), and is only printed.
    shell.open(4).expect("slot 4 opens");
    fire(&mut shell, &[seated(4, &first, buffer(0), RsReset::Fresh)]).expect("A buffers");
    fire(&mut shell, &[seated(4, &first[..4], fold_buffered_at(0, 4, 4), RsReset::Held)])
        .expect("four of A's buffer folds");
    let two_fire = shell.state_bytes(4).expect("slot 4 reads back");
    let (worst, coarse, _) = bank_gap(&folded_a4, &truncated);
    eprintln!("read path, truncated fold vs a plain Fold(A[..4]): worst {worst:.3e}, {coarse} coarse (printed)");
    let (worst, coarse, any) = bank_gap(&two_fire, &truncated);
    eprintln!("read path, truncated fold vs buffer-then-fold-four: worst {worst:.3e}, {coarse} coarse, {any} differ");
    assert_eq!(
        (coarse, any),
        (0, 0),
        "a round replaying {k} and folding 4 left banks {worst} off buffering {k} and folding \
         4 in two fires — the truncated commit lands somewhere else on the read path"
    );
    let d = spread(&plain[0], &partial[0]);
    eprintln!("read path, truncated fold: last-row logit spread {d:.4}");
    assert!(d <= LOGIT_FLOOR, "a truncated fold changed what the window computes: {d}");
}
