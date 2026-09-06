//! Every diagnostic this shell has is a word on the boot document, and a
//! word nobody wrote leaves the record silent.
//!
//! `cargo test -p engine-cuda --test every_diagnostic_is_a_word_the_boot_states`
//!
//! The companion claim is `no_shell_reads_the_environment`, which says the
//! shell reads no `PIE_*` variable. That one says where a knob may NOT come
//! from; this one says a knob still arrives — that each of the fourteen
//! diagnostics that used to be an environment variable has a word, that the
//! word moves exactly the field it names, and that a deployment stating
//! nothing traces nothing. A record whose words nobody can spell would pass
//! the first gate and be useless.

use engine_cuda::{Diagnostics, Knobs};

/// Every word, with the record it should produce out of a default one.
const WORDS: &[(&str, fn(&Diagnostics) -> bool)] = &[
    ("golden-probe", |d| d.golden_probe),
    ("golden-skip", |d| d.golden_skip),
    ("arm-trace", |d| d.arm_trace),
    ("capture-serial", |d| d.capture_serial),
    ("boundary-trace", |d| d.boundary_trace),
    ("reap-trace", |d| d.reap_trace),
    ("trace-census", |d| d.trace_census),
    ("nan-check", |d| d.nan_check),
    ("ptr-trace=decode", |d| {
        d.ptr_trace.as_deref() == Some("decode")
    }),
    ("grid-trace=decode", |d| {
        d.grid_trace.as_deref() == Some("decode")
    }),
    ("plan-trace=all", |d| d.plan_trace.as_deref() == Some("all")),
    ("graph-dot=/tmp/dots", |d| {
        d.graph_dot.as_deref() == Some(std::path::Path::new("/tmp/dots"))
    }),
    ("fuse-chains=off", |d| !d.fuse_chains),
    ("gumbel-direct=off", |d| !d.gumbel_direct),
];

#[test]
fn every_diagnostic_is_a_word_the_boot_states() {
    // A deployment that states nothing traces nothing, and the shell's own
    // default is that deployment.
    let silent = Knobs::default().diagnostics;
    assert_eq!(silent, Diagnostics::default());
    assert!(!silent.any(), "silence is silence");
    assert_eq!(silent.words(), None);

    // Each word alone moves its own field and no other.
    for (word, reads) in WORDS {
        let one: Diagnostics = word
            .parse()
            .unwrap_or_else(|why| panic!("`{word}` is a word this shell speaks: {why}"));
        assert!(reads(&one), "`{word}` did not set the field it names");
        assert!(one.any(), "`{word}` is not silence");
        let mut moved = 0;
        for (_, other) in WORDS {
            if other(&one) != other(&Diagnostics::default()) {
                moved += 1;
            }
        }
        assert_eq!(moved, 1, "`{word}` moved {moved} fields, not one");
    }

    // The whole vocabulary at once, and its own spelling of itself parses
    // back to the same record — which is what a boot dump is read as.
    let all = WORDS
        .iter()
        .map(|(word, _)| *word)
        .collect::<Vec<_>>()
        .join(",");
    let every: Diagnostics = all.parse().expect("the vocabulary parses");
    for (word, reads) in WORDS {
        assert!(reads(&every), "`{word}` was lost in the full list");
    }
    let again: Diagnostics = every
        .words()
        .expect("something is on")
        .parse()
        .expect("the record's own spelling parses");
    assert_eq!(every, again, "the record round-trips through its words");

    // And a word this shell does not speak refuses by name, with the list —
    // the whole reason these are typed rather than read out of the air.
    let why = "goldenprobe"
        .parse::<Diagnostics>()
        .expect_err("a misspelling is not silence");
    assert!(why.contains("goldenprobe"), "the refusal names it: {why}");
    assert!(
        why.contains("golden-probe`"),
        "and lists the vocabulary: {why}"
    );
}
