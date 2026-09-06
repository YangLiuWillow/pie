//! The diagnostics a person debugging turns on, typed — and stated by the
//! boot document rather than read out of the environment (article 9).
//!
//! Every knob here was an environment variable until this module existed, and
//! that is exactly the smell the constitution names: a shell reading
//! `PIE_GOLDEN_PROBE` takes a knob from nowhere, so nothing declares it,
//! nothing validates it, no boot dump records it, and a typo is silence. The
//! knobs themselves were never the problem — the golden probe found a live
//! arena defect and `capture-serial` localised a body-replay one — so they are
//! all still here, under one name each, spelled the same way in a config file
//! and on the command line:
//!
//! ```text
//! pie serve  --diag golden-probe,arm-trace
//! pie run …  --diag 'ptr-trace=decode,graph-dot=/tmp/dots'
//! ```
//!
//! or, for a deployment that wants one standing:
//!
//! ```toml
//! [engine]
//! diagnostics = "boundary-trace,reap-trace"
//! ```
//!
//! Both fill one [`Diagnostics`], which rides `Knobs::diagnostics` onto every
//! [`Boot`](super::Boot) — so `--diag` needs no rebuild, a verbose boot dumps
//! it with the rest of the boot, and a word this shell does not speak is
//! refused **by name** at boot instead of being ignored for an afternoon.
//!
//! # Why a published record and not a parameter
//!
//! Most of these are read from places no boot reaches: a capture loop deep in
//! [`crate::record`], a dispatch arm building an attention plan, a frame mark
//! on the enqueue path. Threading a `&Diagnostics` down every one of those
//! call chains would be a large diff whose only effect is to carry a word that
//! never changes after boot. So the record is *published* once, by the boot
//! that states it ([`publish`]), and read through [`on`]. The provenance is
//! still the boot document — `Knobs::diagnostics` is this module's only source
//! — which is what article 9 asks for. `[metal.tuning]`
//! (`engine_metal::boot::tuning`) is the same pattern one shell over, and
//! [`crate::record::REPLAY_UPTO`] is the same pattern one file over.

use std::path::PathBuf;
use std::sync::OnceLock;

/// What a person debugging turned on, for this process.
///
/// Every field is off/absent by default except the two A/B arms
/// (`fuse_chains`, `gumbel_direct`), which name a fusion that is *on* and can
/// only be switched off — the arm is the diagnostic, the fusion is the
/// deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    /// `golden-probe`: on a golden disagreement, ask two more questions before
    /// the verdict — is each arm repeatable, and does the body's lane `i`
    /// answer some OTHER lane of the walk (a lane order that moved, not a
    /// wrong number)? Also prints one `[body-probe]` line per route decision,
    /// and is what the [`crate::record::REPLAY_FROM`] /
    /// [`crate::record::REPLAY_UPTO`] bisection is driven under.
    pub golden_probe: bool,
    /// `golden-skip`: a body that disagrees with its own walk is dropped and
    /// its key refused (it walks eagerly) instead of failing the load, so one
    /// boot lists EVERY disagreement rather than stopping at the first.
    pub golden_skip: bool,
    /// `arm-trace`: what the arming pass did, key by key — the classes it
    /// landed, the buckets it refused, the bodies it armed.
    pub arm_trace: bool,
    /// `ptr-trace=<substring of a body key>`: per node of a matching body's
    /// capture and of its golden's eager arm, the op and the device pointers
    /// it read and wrote. The one trace that answers "the same launches, but
    /// over different memory?".
    pub ptr_trace: Option<String>,
    /// `graph-dot=<dir>`: every captured exec of the `ptr-trace` key written
    /// to `<dir>/exec<n>.dot`. Needs `ptr-trace` to name a key — a dump of
    /// every body of a load is not a diagnostic, it is a disk.
    pub graph_dot: Option<PathBuf>,
    /// `grid-trace=<substring of a body key>`: per launch of a matching body,
    /// the live span beside the ceiling grid it was captured at — the two
    /// numbers a replay disagreeing with its walk is read by.
    pub grid_trace: Option<String>,
    /// `plan-trace=<rows>` or `plan-trace=all`: the attention plan a fire of
    /// exactly that many rows built. Both arms of a golden print theirs, which
    /// is how a body's schedule is read beside its walk's.
    pub plan_trace: Option<String>,
    /// `capture-serial`: capture on ONE stream — the fork/join event points
    /// the stream pass baked are not walked — while still writing the capture
    /// down. A body that agrees with its walk only under this names the stream
    /// plan as what it disagrees over.
    pub capture_serial: bool,
    /// `boundary-trace`: where the host spends the frame boundary, one line
    /// per frame (see [`crate::serve::btrace`]).
    pub boundary_trace: bool,
    /// `reap-trace`: which door waited, and how long, per reap.
    pub reap_trace: bool,
    /// `trace-census`: the op histogram of the trace this load compiles, after
    /// fusion — one line, at load.
    pub trace_census: bool,
    /// `nan-check`: after every dispatched node, scan its outputs for a NaN
    /// and name the op that first produced one. Expensive: it synchronises and
    /// reads back every output.
    pub nan_check: bool,
    /// `fuse-chains=off`: land the traced launches instead of the
    /// norm-add-scale-norm, gemm-epilogue and modulation folds. **On** by
    /// default; this is the A/B arm.
    pub fuse_chains: bool,
    /// `gumbel-direct=off`: keep a program's Gumbel-max head as the launches
    /// it was traced as. **On** by default; this is the A/B arm.
    pub gumbel_direct: bool,
}

impl Default for Diagnostics {
    /// Nothing traced, both fusions on: what a deployment that says nothing
    /// gets, and byte for byte what every deployment got before this record
    /// existed.
    fn default() -> Diagnostics {
        Diagnostics {
            golden_probe: false,
            golden_skip: false,
            arm_trace: false,
            ptr_trace: None,
            graph_dot: None,
            grid_trace: None,
            plan_trace: None,
            capture_serial: false,
            boundary_trace: false,
            reap_trace: false,
            trace_census: false,
            nan_check: false,
            fuse_chains: true,
            gumbel_direct: true,
        }
    }
}

impl Diagnostics {
    /// Whether anything at all is stated — what a boot log reports.
    #[must_use]
    pub fn any(&self) -> bool {
        *self != Diagnostics::default()
    }

    /// The words that are on, in the spelling `--diag` takes. `None` when
    /// nothing is, so a caller says "none" in its own voice.
    #[must_use]
    pub fn words(&self) -> Option<String> {
        let mut words: Vec<String> = Vec::new();
        for (word, on) in [
            ("golden-probe", self.golden_probe),
            ("golden-skip", self.golden_skip),
            ("arm-trace", self.arm_trace),
            ("capture-serial", self.capture_serial),
            ("boundary-trace", self.boundary_trace),
            ("reap-trace", self.reap_trace),
            ("trace-census", self.trace_census),
            ("nan-check", self.nan_check),
        ] {
            if on {
                words.push(word.to_string());
            }
        }
        for (word, value) in [
            ("ptr-trace", self.ptr_trace.clone()),
            ("grid-trace", self.grid_trace.clone()),
            ("plan-trace", self.plan_trace.clone()),
            (
                "graph-dot",
                self.graph_dot.as_ref().map(|dir| dir.display().to_string()),
            ),
        ] {
            if let Some(value) = value {
                words.push(format!("{word}={value}"));
            }
        }
        if !self.fuse_chains {
            words.push("fuse-chains=off".to_string());
        }
        if !self.gumbel_direct {
            words.push("gumbel-direct=off".to_string());
        }
        (!words.is_empty()).then(|| words.join(","))
    }
}

/// Whether a `word=value` that reads as a switch means on.
///
/// `on`/`1`/`true`/`yes` and an empty value are on; `off`/`0`/`false`/`no` are
/// off; anything else is a refusal, because `capture-serial=maybe` is a
/// question the shell cannot answer by guessing.
fn switch(word: &str, value: &str) -> std::result::Result<bool, String> {
    match value {
        "" | "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "`{word}={other}` is not a switch; write `{word}` or `{word}=off`"
        )),
    }
}

/// The vocabulary, for the refusal that lists it.
const WORDS: &str = "`golden-probe`, `golden-skip`, `arm-trace`, \
     `ptr-trace=<key substring>`, `graph-dot=<dir>`, `grid-trace=<key substring>`, \
     `plan-trace=<rows|all>`, `capture-serial`, `boundary-trace`, `reap-trace`, \
     `trace-census`, `nan-check`, `fuse-chains=off`, `gumbel-direct=off`";

impl std::str::FromStr for Diagnostics {
    type Err = String;

    /// A comma-separated word list: `golden-probe,ptr-trace=decode,fuse-chains=off`.
    ///
    /// A bare word turns its knob on; `word=off` turns it off; a word that
    /// takes a value takes it after `=`. Empty words (a trailing comma,
    /// `--diag ""`) are skipped, so a key set to the empty string is "nothing
    /// on" rather than a refusal. **An unknown word refuses by name and lists
    /// the vocabulary** — the whole point of typing these is that a
    /// misspelling is not silence.
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
                "golden-probe" => diag.golden_probe = switch(word, value)?,
                "golden-skip" => diag.golden_skip = switch(word, value)?,
                "arm-trace" => diag.arm_trace = switch(word, value)?,
                "capture-serial" => diag.capture_serial = switch(word, value)?,
                "boundary-trace" => diag.boundary_trace = switch(word, value)?,
                "reap-trace" => diag.reap_trace = switch(word, value)?,
                "trace-census" => diag.trace_census = switch(word, value)?,
                "nan-check" => diag.nan_check = switch(word, value)?,
                "fuse-chains" => diag.fuse_chains = switch(word, value)?,
                "gumbel-direct" => diag.gumbel_direct = switch(word, value)?,
                // The valued words. An empty value refuses rather than
                // meaning "match everything": `ptr-trace=` would trace every
                // body of a load, which is a way to fill a disk, not to read
                // one.
                "ptr-trace" | "grid-trace" | "plan-trace" | "graph-dot" => {
                    if value.is_empty() {
                        return Err(format!(
                            "`{word}` takes a value: `{word}=<{}>`",
                            match word {
                                "graph-dot" => "directory",
                                "plan-trace" => "rows|all",
                                _ => "substring of a body key",
                            }
                        ));
                    }
                    match word {
                        "ptr-trace" => diag.ptr_trace = Some(value.to_string()),
                        "grid-trace" => diag.grid_trace = Some(value.to_string()),
                        "plan-trace" => diag.plan_trace = Some(value.to_string()),
                        _ => diag.graph_dot = Some(PathBuf::from(value)),
                    }
                }
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
/// One process hosts one deployment — a tensor-parallel group's ranks are
/// threads of it, booted from one `[engine]` table — so the first boot to
/// state a record states it for all of them. A later boot asking for a
/// *different* one is told so on stderr and ignored, rather than silently
/// changing what a running rank traces.
pub(crate) fn publish(stated: &Diagnostics) {
    let published = PUBLISHED.get_or_init(|| stated.clone());
    if published != stated {
        eprintln!(
            "engine-cuda: this process already published a diagnostics record \
             ({}), so the one this boot states ({}) is ignored; every rank of a \
             group traces the same thing",
            published.words().unwrap_or_else(|| "none".to_string()),
            stated.words().unwrap_or_else(|| "none".to_string()),
        );
    }
}

/// What this process's boot stated. All-off before any boot has, which is what
/// a unit test that never opens a shell reads.
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
        assert_eq!(diag.words(), None);
    }

    #[test]
    fn a_word_list_names_each_knob() {
        let diag: Diagnostics = "golden-probe, ptr-trace=decode:c0 ,fuse-chains=off"
            .parse()
            .expect("the three words parse");
        assert!(diag.golden_probe);
        assert_eq!(diag.ptr_trace.as_deref(), Some("decode:c0"));
        assert!(!diag.fuse_chains);
        assert!(diag.gumbel_direct, "an unnamed arm keeps its default");
    }

    #[test]
    fn an_unknown_word_refuses_by_name() {
        let why = "golden-probes".parse::<Diagnostics>().expect_err("refused");
        assert!(why.contains("golden-probes"), "{why}");
        assert!(
            why.contains("golden-probe`"),
            "and lists the vocabulary: {why}"
        );
    }

    #[test]
    fn a_valued_word_with_no_value_refuses() {
        assert!("ptr-trace".parse::<Diagnostics>().is_err());
        assert!("graph-dot=".parse::<Diagnostics>().is_err());
    }

    #[test]
    fn the_words_round_trip() {
        let list = "arm-trace,capture-serial,plan-trace=all,gumbel-direct=off";
        let diag: Diagnostics = list.parse().expect("parses");
        let again: Diagnostics = diag
            .words()
            .expect("something is on")
            .parse()
            .expect("its own spelling parses");
        assert_eq!(diag, again);
    }
}
