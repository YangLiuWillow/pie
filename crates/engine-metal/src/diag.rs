//! The diagnostics a person debugging turns on, typed — and stated by the
//! boot document rather than read out of the environment (article 9).
//!
//! Twenty-six `PIE_*` environment variables were compiled into this shell:
//! traces, dumps, measurement arms and the numbers a sweep moves. They are one
//! typed record now, filled from the boot document the same way
//! `[metal.tuning]` already fills `kernels_metal::tuning` — the precedent this
//! module follows, and whose own doc comment says why ("swept via the boot
//! document rather than environment variables, which would need a rebuild per
//! arm"; an environment variable does not need a rebuild either, but it does
//! need a shell to read one, and that is what article 9 forbids).
//!
//! ```toml
//! [engine]
//! diagnostics = "tier-trace,kernel-profile=2"
//! ```
//!
//! or, per run and touching no file, `pie run … --diag tier-trace`. The words
//! reach this shell as `[metal] diagnostics` in the in-memory boot document
//! the runtime writes, are parsed here, and a word this shell does not speak
//! refuses the open by name.
//!
//! # Why a published record
//!
//! These are read from a window cut, a kernel compile, an allocator, a
//! keep-alive thread — places a boot never reaches. The record is published
//! once by [`crate::boot::open`] ([`publish`]) and read through [`on`]; the
//! provenance is the boot document, which is what article 9 asks for. The CUDA
//! shell's `engine_cuda::serve::diag` is the same module, one shell over.

use std::path::PathBuf;
use std::sync::OnceLock;

/// How the per-entrypoint kernel profile keys its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// Off: no per-kernel command buffers, no timings.
    #[default]
    Off,
    /// One row per entrypoint.
    On,
    /// One row per entrypoint AND scalar argument set (a matvec's K and N, a
    /// router's expert count…), so one entrypoint's time splits by shape.
    /// Prints sixty rows a fire instead of ten.
    Shaped,
}

impl Profile {
    /// Whether anything is timed at all.
    #[must_use]
    pub fn on(self) -> bool {
        !matches!(self, Profile::Off)
    }

    /// Whether the key carries the launch's scalars.
    #[must_use]
    pub fn shaped(self) -> bool {
        matches!(self, Profile::Shaped)
    }

    /// How many rows a fire's profile prints.
    #[must_use]
    pub fn rows(self) -> usize {
        match self {
            Profile::Shaped => 60,
            Profile::Off | Profile::On => 10,
        }
    }
}

/// Which kind of streamed dispatch `streamed_repeat` repeats. Mirrors
/// `program::launch::StepKind`, which is private to that module; this is the
/// word a person types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamedKind {
    Wide,
    Partial,
    Single,
    Reduce,
    Argmax,
}

/// What a person debugging turned on, for this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    /// `cut-trace`: every cut a fire makes — the rows of the value the cut
    /// carries, the pass it is, and the host time the blocking commit cost.
    pub cut_trace: bool,
    /// `tier-trace`: what a fire cost the streamed tier — seat copies, cuts,
    /// hits/misses, host time inside the cuts — as deltas, one line a fire;
    /// and, at load, what prefaulting and bank decoding took.
    pub tier_trace: bool,
    /// `rs-trace`: every non-fold recurrent lane plan, and every `rs_land`'s
    /// two rectangles.
    pub rs_trace: bool,
    /// `fire-trace`: `[fire t_us=…]` lines around a fire's phases, which
    /// `benches/pie_bench.py` reads off the server's stdout.
    pub fire_trace: bool,
    /// `region-trace`: one line per compiled region naming the form it took,
    /// and why the grouped form was declined when it was — the decline reason
    /// reaches nobody otherwise.
    pub region_trace: bool,
    /// `streamed-trace`: the streamed form's step words and grid sizes.
    pub streamed_trace: bool,
    /// `kernel-profile[=1|2]`: the device time of each fire by entrypoint. A
    /// measurement mode, not a serving one — it takes a command buffer per
    /// kernel.
    pub kernel_profile: Profile,
    /// `kernel-dump=<dir>`: every generated Metal source as `<entry>.metal`,
    /// and every streamed dispatch table as `<entry>.tables`, for a standalone
    /// harness to replay.
    pub kernel_dump: Option<PathBuf>,
    /// `route-dump=<file>`: the router's own vectors, as they were written.
    pub route_dump: Option<PathBuf>,
    /// `pass-half=off`: seat the WHOLE slab in one expert-major pass instead
    /// of half of it (which exists so the other half can be filled while this
    /// one runs). **On** by default; this is the A/B arm.
    pub pass_half: bool,
    /// `route-prefetch=off`: do not predict the next fire's routes. **On** by
    /// default; this is the A/B arm.
    pub route_prefetch: bool,
    /// `expert-passes=off`: no expert-major passes at all — one pass over
    /// every region. **On** by default; this is the A/B arm.
    pub expert_passes: bool,
    /// `keepalive=off`: stop the keep-alive spinner that holds the GPU's
    /// clocks up between fires. **On** by default; this is the A/B arm
    /// (measured: dsv4 237 → 163 ms/token with it, GLM 320 → 337 ms against).
    pub keepalive: bool,
    /// `scratch-no-zero`: skip the host memset of a fire's scratch, to measure
    /// what the host's touch of those pages costs the device. Off: the memset
    /// is what serving does.
    pub scratch_no_zero: bool,
    /// `host-rows`: stage a fire's rows through the host.
    pub host_rows: bool,
    /// `prefault`: walk every resident plane at load so the pager does the
    /// work up front instead of during the first fires.
    pub prefault: bool,
    /// `copy-resident`: copy each window into a Metal-allocated buffer instead
    /// of binding the mapping, to price the no-copy binding's first-use
    /// wiring.
    pub copy_resident: bool,
    /// `prefetch-k=<n>`: how many fires ahead the route predictor looks.
    /// `None` keeps the shell's own `PREFETCH_K`.
    pub prefetch_k: Option<usize>,
    /// `seat-threads=<n>`: threads the seat filler uses. `None` keeps the
    /// shell's own `SEAT_THREADS`.
    pub seat_threads: Option<usize>,
    /// `keepalive-iters=<n>`: the spinner's inner loop count, tuned so one
    /// dispatch is a few hundred microseconds. `None` keeps the shell's own.
    pub keepalive_iters: Option<u32>,
    /// `window-ceiling=<bytes>`: clamp `maxBufferLength` down, to cut a
    /// mapping into more windows than the device would ask for. `None` takes
    /// the device's own answer.
    pub window_ceiling: Option<u64>,
    /// `streamed-groups=<n>`: cap the blocks a wide dispatch spreads over.
    /// `None` keeps the shell's own `STREAMED_MAX_GROUPS`.
    pub streamed_groups: Option<u32>,
    /// `streamed-repeat=<n>`: issue each wide dispatch `n` times — all are
    /// idempotent, so this prices a dispatch. `1` is serving.
    pub streamed_repeat: usize,
    /// `streamed-repeat-kind=<wide|partial|single|reduce|argmax>`: narrow
    /// `streamed_repeat` to one kind of step. `None` repeats the wide ones.
    pub streamed_repeat_kind: Option<StreamedKind>,
    /// `streamed-limit=<k>`: run only the first `k` steps. This BREAKS the
    /// program and times what ran. `None` runs all of them.
    pub streamed_limit: Option<usize>,
    /// `streamed-nop=<n>`: `n` dispatches of one group that hit the kernel's
    /// `default: return` — the production floor of a dispatch with these
    /// bindings and no op behind it.
    pub streamed_nop: usize,
}

impl Default for Diagnostics {
    /// Nothing traced, nothing dumped, every arm at what serving does — byte
    /// for byte what this shell did before the record existed and nobody had
    /// set a `PIE_*` variable.
    fn default() -> Diagnostics {
        Diagnostics {
            cut_trace: false,
            tier_trace: false,
            rs_trace: false,
            fire_trace: false,
            region_trace: false,
            streamed_trace: false,
            kernel_profile: Profile::Off,
            kernel_dump: None,
            route_dump: None,
            pass_half: true,
            route_prefetch: true,
            expert_passes: true,
            keepalive: true,
            scratch_no_zero: false,
            host_rows: false,
            prefault: false,
            copy_resident: false,
            prefetch_k: None,
            seat_threads: None,
            keepalive_iters: None,
            window_ceiling: None,
            streamed_groups: None,
            streamed_repeat: 1,
            streamed_repeat_kind: None,
            streamed_limit: None,
            streamed_nop: 0,
        }
    }
}

impl Diagnostics {
    /// Whether anything at all is stated.
    #[must_use]
    pub fn any(&self) -> bool {
        *self != Diagnostics::default()
    }
}

/// Whether a `word=value` that reads as a switch means on.
fn switch(word: &str, value: &str) -> std::result::Result<bool, String> {
    match value {
        "" | "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "`{word}={other}` is not a switch; write `{word}` or `{word}=off`"
        )),
    }
}

/// A number, or a refusal naming the word that wanted one.
fn number<T: std::str::FromStr>(word: &str, value: &str) -> std::result::Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("`{word}` takes a number; `{value}` is not one"))
}

/// The vocabulary, for the refusal that lists it.
const WORDS: &str = "`cut-trace`, `tier-trace`, `rs-trace`, `fire-trace`, \
     `region-trace`, `streamed-trace`, `kernel-profile[=1|2]`, \
     `kernel-dump=<dir>`, `route-dump=<file>`, `pass-half=off`, \
     `route-prefetch=off`, `expert-passes=off`, `keepalive=off`, \
     `scratch-no-zero`, `host-rows`, `prefault`, `copy-resident`, \
     `prefetch-k=<n>`, `seat-threads=<n>`, `keepalive-iters=<n>`, \
     `window-ceiling=<bytes>`, `streamed-groups=<n>`, `streamed-repeat=<n>`, \
     `streamed-repeat-kind=<wide|partial|single|reduce|argmax>`, \
     `streamed-limit=<n>`, `streamed-nop=<n>`";

impl std::str::FromStr for Diagnostics {
    type Err = String;

    /// A comma-separated word list: `tier-trace,kernel-profile=2,pass-half=off`.
    ///
    /// A bare word turns its knob on; `word=off` turns it off; a word that
    /// takes a value takes it after `=`. Empty words are skipped. **An unknown
    /// word refuses by name and lists the vocabulary.**
    fn from_str(list: &str) -> std::result::Result<Diagnostics, String> {
        let mut diag = Diagnostics::default();
        for term in list.split(',') {
            let term = term.trim();
            if term.is_empty() {
                continue;
            }
            let (word, value) = match term.split_once('=') {
                Some((word, value)) => (word.trim(), value.trim()),
                None => (term, ""),
            };
            match word {
                "cut-trace" => diag.cut_trace = switch(word, value)?,
                "tier-trace" => diag.tier_trace = switch(word, value)?,
                "rs-trace" => diag.rs_trace = switch(word, value)?,
                "fire-trace" => diag.fire_trace = switch(word, value)?,
                "region-trace" => diag.region_trace = switch(word, value)?,
                "streamed-trace" => diag.streamed_trace = switch(word, value)?,
                "pass-half" => diag.pass_half = switch(word, value)?,
                "route-prefetch" => diag.route_prefetch = switch(word, value)?,
                "expert-passes" => diag.expert_passes = switch(word, value)?,
                "keepalive" => diag.keepalive = switch(word, value)?,
                "scratch-no-zero" => diag.scratch_no_zero = switch(word, value)?,
                "host-rows" => diag.host_rows = switch(word, value)?,
                "prefault" => diag.prefault = switch(word, value)?,
                "copy-resident" => diag.copy_resident = switch(word, value)?,
                "kernel-profile" => {
                    diag.kernel_profile = match value {
                        "" | "1" | "on" | "true" | "yes" => Profile::On,
                        "2" | "shaped" => Profile::Shaped,
                        "off" | "0" | "false" | "no" => Profile::Off,
                        other => {
                            return Err(format!(
                                "`kernel-profile={other}` is not a mode; write \
                                 `kernel-profile` (per entrypoint), \
                                 `kernel-profile=2` (per entrypoint and shape), \
                                 or `kernel-profile=off`"
                            ));
                        }
                    };
                }
                "streamed-repeat-kind" => {
                    diag.streamed_repeat_kind = Some(match value {
                        "wide" => StreamedKind::Wide,
                        "partial" => StreamedKind::Partial,
                        "single" => StreamedKind::Single,
                        "reduce" => StreamedKind::Reduce,
                        "argmax" => StreamedKind::Argmax,
                        other => {
                            return Err(format!(
                                "`streamed-repeat-kind={other}` does not name a step \
                                 kind; the spellings are `wide`, `partial`, \
                                 `single`, `reduce`, `argmax`"
                            ));
                        }
                    });
                }
                "kernel-dump" | "route-dump" => {
                    if value.is_empty() {
                        return Err(format!("`{word}` takes a path: `{word}=<path>`"));
                    }
                    match word {
                        "kernel-dump" => diag.kernel_dump = Some(PathBuf::from(value)),
                        _ => diag.route_dump = Some(PathBuf::from(value)),
                    }
                }
                "prefetch-k" => diag.prefetch_k = Some(number(word, value)?),
                "seat-threads" => {
                    // Zero threads is a typo, not a request; the shell's own
                    // figure is what an absent word means.
                    let threads: usize = number(word, value)?;
                    if threads == 0 {
                        return Err("`seat-threads` must be > 0".to_string());
                    }
                    diag.seat_threads = Some(threads);
                }
                "keepalive-iters" => diag.keepalive_iters = Some(number(word, value)?),
                "window-ceiling" => {
                    let bytes: u64 = number(word, value)?;
                    if bytes == 0 {
                        return Err(
                            "`window-ceiling` must be > 0; omit it for the device's own \
                             `maxBufferLength`"
                                .to_string(),
                        );
                    }
                    diag.window_ceiling = Some(bytes);
                }
                "streamed-groups" => {
                    let groups: u32 = number(word, value)?;
                    if groups == 0 {
                        return Err("`streamed-groups` must be > 0".to_string());
                    }
                    diag.streamed_groups = Some(groups);
                }
                "streamed-repeat" => diag.streamed_repeat = number::<usize>(word, value)?.max(1),
                "streamed-limit" => diag.streamed_limit = Some(number(word, value)?),
                "streamed-nop" => diag.streamed_nop = number(word, value)?,
                other => {
                    return Err(format!(
                        "`{other}` does not name a diagnostic; the words are {WORDS}"
                    ));
                }
            }
        }
        Ok(diag)
    }
}

/// The record this process serves under, published by the first boot.
static PUBLISHED: OnceLock<Diagnostics> = OnceLock::new();

/// State this process's diagnostics, from the boot document that carried them.
///
/// One process hosts one deployment, so the first boot to state a record
/// states it for the process. A later boot asking for a different one is told
/// so and ignored.
pub(crate) fn publish(stated: &Diagnostics) {
    let published = PUBLISHED.get_or_init(|| stated.clone());
    if published != stated {
        eprintln!(
            "engine-metal: this process already published a diagnostics record, \
             so the different one this boot states is ignored"
        );
    }
}

/// What this process's boot stated. All-off before any boot has, which is what
/// a unit test that never opens a device reads.
#[must_use]
pub fn on() -> &'static Diagnostics {
    static OFF: OnceLock<Diagnostics> = OnceLock::new();
    PUBLISHED
        .get()
        .unwrap_or_else(|| OFF.get_or_init(Diagnostics::default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_list_turns_nothing_on() {
        let diag: Diagnostics = "".parse().expect("silence parses");
        assert_eq!(diag, Diagnostics::default());
        assert!(!diag.any());
    }

    #[test]
    fn a_word_list_names_each_knob() {
        let diag: Diagnostics = "tier-trace, kernel-profile=2 ,pass-half=off,streamed-nop=3"
            .parse()
            .expect("the four words parse");
        assert!(diag.tier_trace);
        assert_eq!(diag.kernel_profile, Profile::Shaped);
        assert!(diag.kernel_profile.shaped());
        assert_eq!(diag.kernel_profile.rows(), 60);
        assert!(!diag.pass_half);
        assert_eq!(diag.streamed_nop, 3);
        assert!(diag.keepalive, "an unnamed arm keeps its default");
    }

    #[test]
    fn an_unknown_word_refuses_by_name() {
        let why = "tier-traces".parse::<Diagnostics>().expect_err("refused");
        assert!(why.contains("tier-traces"), "{why}");
        assert!(
            why.contains("tier-trace`"),
            "and lists the vocabulary: {why}"
        );
    }

    #[test]
    fn a_number_that_is_not_one_refuses() {
        assert!("prefetch-k=many".parse::<Diagnostics>().is_err());
        assert!("seat-threads=0".parse::<Diagnostics>().is_err());
        assert!("window-ceiling=0".parse::<Diagnostics>().is_err());
    }

    #[test]
    fn the_streamed_kind_takes_one_of_five_words() {
        let diag: Diagnostics = "streamed-repeat=4,streamed-repeat-kind=reduce"
            .parse()
            .expect("parses");
        assert_eq!(diag.streamed_repeat, 4);
        assert_eq!(diag.streamed_repeat_kind, Some(StreamedKind::Reduce));
        assert!(
            "streamed-repeat-kind=narrow"
                .parse::<Diagnostics>()
                .is_err()
        );
    }
}
