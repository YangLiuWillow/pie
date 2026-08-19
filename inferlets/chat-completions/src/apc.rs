//! Automatic prefix cache for the PER-REQUEST arm.
//!
//! Strategy A serves one inferlet instance per request and drops it, so every
//! turn re-prefills the whole conversation. Measured on `django-13089`,
//! Coder-30B: turns 3–5 cost 33.9 s, 63.3 s and 57.3 s with `cached=0` on every
//! one, against 4.5–6.1 s for the equivalent turns under the session arm. The
//! work is real and it is repeated.
//!
//! This module removes the repetition WITHOUT a long-lived process, by parking
//! the KV in the engine's own index instead of in guest memory.
//!
//! ## Why this is not Strategy B's mechanism
//!
//! `opencode-session` keeps `Retained { state, boundaries }` in a `Vec` inside
//! its own process and resumes by scanning it — which is why it needs a
//! long-lived process, a sticky WebSocket and a shim. That map cannot outlive
//! the instance, so a per-request arm cannot use it.
//!
//! `WorkingSet::update_index` / `from_index` are ENGINE-level and do outlive
//! the instance. That is the whole basis of this module, and it was verified on
//! this build rather than assumed: `explicit_prefix_index_survives_across_
//! inferlet_instances` in `runtime/engine/tests/e2e.rs` publishes from one
//! inferlet instance and reads it back from a second. (The pre-existing
//! `explicit_prefix_index_submits_only_the_suffix` publishes and looks up
//! inside ONE invocation, so it never tested this.)
//!
//! ## Page alignment is not a detail
//!
//! A message boundary lands wherever the template puts it; KV is parked in
//! whole pages. [`publish`] parks a `slice(0, n_pages)`, so a cut is only
//! usable if it falls on a page edge. Every candidate here is therefore aligned
//! DOWN to a page multiple before it is addressed — and the address is computed
//! over the aligned token prefix, never over the message boundary that
//! suggested it. Addressing `tokens[..b]` and parking `pages[..b/page]` would
//! publish KV that covers less than its own name claims, and the next turn
//! would graft a suffix onto a prefix that ends in the wrong place: fluent
//! output, wrong context, nothing to catch it.
//!
//! ## The key
//!
//! `prefix_addresses` hashes `model_id ‖ template_marker ‖ token_ids`, so a hit
//! means an identical token prefix under an identical template on an identical
//! model. Any divergence — a different system prompt, an edited turn, a
//! tokenizer change — changes the tokens, changes the address, and misses
//! cleanly. A false hit that returns the wrong KV is not unlikely; it is
//! unrepresentable.
//!
//! Shared with the session arm on purpose: both arms address prefixes the same
//! way, so an A/B measures where the KV lives rather than how it is named.
//!
//! ## Only the history is addressable
//!
//! The render is `history ‖ cue`. The cue is generation scaffolding: replaying
//! this turn as history renders no cue at all, so KV that includes it sits
//! under an address no future render can produce — a permanent miss that also
//! costs the pages. Every cut here is therefore bounded by the history length,
//! and the caller is responsible for passing that boundary.

use inferlet::ptir::attention::prelude::*;
use pie_openai_serving::{TEMPLATE_MARKER, prefix_addresses};

/// How many aligned cuts to park per turn.
///
/// **One**, and the cost that decides it is contention, not pages.
///
/// Every `slice` and every `update_index` takes the engine's GLOBAL KV mutex
/// (`store_registry::with_kv_lock`, `host/kv_working_set.rs`), which the engine
/// already treats as a throughput hazard — `fire.rs` goes out of its way to
/// keep a single integer load off it, noting "42979 exclusive acquisitions per
/// D/512 run". Parking eight cuts is sixteen acquisitions per turn.
///
/// That was fine in a serial probe and not fine under opencode, which issues
/// concurrent requests. The measured result was
/// `Metal forward timed out before its completion fence`, then a poison epoch,
/// then every later request refused by the gateway with `cluster saturated` —
/// a wedged server, from a cache that is supposed to be an optimisation.
///
/// What one cut gives up is real but unmeasured: a DIFFERENT conversation can
/// only share opencode's head, and only the ~2.1k tokens ahead of the
/// per-session environment block, so it needs a SHALLOW cut parked. Parking the
/// deepest cut alone serves the next turn of this conversation — which is the
/// win that was actually measured at 99.9% reuse — and gives up the
/// cross-conversation one. That trade should be revisited by measuring
/// cross-conversation hit rate, not by assuming it back.
/// `APC_OFF=1 cargo build` parks nothing, which disables the cache end to end:
/// a lookup can only hit what a publish parked. That is the CONTROL arm, built
/// from this same source so an A/B cannot accidentally compare two builds that
/// differ in more than the cache.
const PUBLISH_MAX: usize = if option_env!("APC_OFF").is_some() { 0 } else { 1 };

/// Extra cut candidates emitted every this many tokens, independent of where
/// the render's message boundaries fall.
///
/// Message-unit boundaries alone are too coarse to share anything ACROSS
/// conversations, for a reason that is a position accident rather than a
/// content one. opencode's head is one render op — `EquipAfterSystem` folds the
/// system turn and the tool schemas together — so its only unit boundary is the
/// whole ~7.3k-token head, and that never matches between two sessions because
/// the system prompt carries a per-session environment block (working
/// directory, git status, today's date) 8,695 chars into a 9,648-char system
/// message. Everything after it — 21,188 chars of byte-identical tool schemas —
/// is shifted by however much that block differs, so it is not a shared prefix
/// at all. A cache can reach only the 28% ahead of it, and only if a boundary
/// exists there.
///
/// Striding puts one there without knowing anything about opencode: wherever
/// two token streams agree, some stride point lands inside the agreement.
/// Same idea as vLLM's block-level APC, at coarser granularity because each
/// boundary costs a digest snapshot rather than a page-table entry.
///
/// Kept equal to the session arm's stride on purpose: the two arms must produce
/// the same cut set for the same conversation, or an A/B measures the cut
/// policy instead of where the KV lives.
pub const BOUNDARY_STRIDE: u32 = 256;

/// A resume point: how many tokens of the render are already in KV, the
/// working set holding them, and the index address it was found under.
///
/// The address is not diagnostics — it is the lifecycle handle. The entry it
/// names pins the PREVIOUS turn's page chain, and extending the resumed set
/// privatizes the shared stratum, so while both live the pool carries two
/// generations of the conversation. Measured: a single ramped conversation
/// under strategy A died with a 503 at 30,855 tokens — half of B's 61,711 on
/// the identical pool — with the live chain (30,855) plus the previous entry
/// (28,640) plus the reservation summing to the pool exactly. The turn that
/// successfully parks its own deeper cut must therefore REMOVE the entry it
/// resumed from, which needs this address.
pub struct Resume {
    pub cached_tokens: u32,
    pub ws: WorkingSet,
    pub address: String,
}

/// The addressed cut ladder for one render.
pub struct Plan {
    /// `(cut, address)`, ascending. Every cut is page-aligned, non-zero, and
    /// within the history.
    cuts: Vec<(u32, String)>,
}

impl Plan {
    /// Address every usable cut of `history`.
    ///
    /// `boundaries` are ascending token counts into `history` (render-unit
    /// edges and stride points); `page` is the driver's KV page size.
    pub fn build(model_id: &str, history: &[u32], boundaries: &[u32], page: u32) -> Plan {
        let cuts = aligned_cuts(boundaries, page, history.len() as u32);
        Plan {
            cuts: prefix_addresses(model_id, TEMPLATE_MARKER, history, &cuts),
        }
    }

    /// Longest-first lookup. Returns the first cut whose address is parked.
    ///
    /// Every cut is tried; there is no ladder cap. The cap in RatioThink's
    /// `chat-apc` (4) and OpenHands (8) buys back the cost of a MISS, and their
    /// misses are expensive because a probe there materialises a context. A
    /// miss here is `self.indexes.get(key)` on a `HashMap`
    /// (`runtime/engine/src/store/kv.rs:451`) returning `None` before anything
    /// is built — a host call, not a materialisation. Only the single hit that
    /// ends the scan builds a working set.
    ///
    /// Capping would cost real hits, and quietly. Cuts are ~256 tokens apart,
    /// so a cap of 8 reaches back only ~2k tokens — and one agent turn that
    /// pastes a file into a tool result adds more than that, which would put
    /// the previous turn's own parked prefix out of reach on exactly the turns
    /// where prefill is most expensive.
    ///
    /// A lookup error is treated as a miss, not a fault: the cache is an
    /// optimisation, and a turn that cannot resume is still a correct turn that
    /// pays a rebuild. The alternative — failing the request because the cache
    /// is unavailable — trades a slow answer for no answer.
    pub fn resume(&self) -> Option<Resume> {
        for (cut, address) in self.cuts.iter().rev() {
            match WorkingSet::from_index(address.as_bytes()) {
                Ok(Some(ws)) => {
                    return Some(Resume {
                        cached_tokens: *cut,
                        ws,
                        address: address.clone(),
                    });
                }
                Ok(None) => continue,
                Err(e) => {
                    eprintln!("[apc] from_index failed at cut {cut}: {e}; treating as a miss");
                    continue;
                }
            }
        }
        None
    }

    /// The cuts to park after prefill: at most [`PUBLISH_MAX`], sampled evenly
    /// across the ladder so both ends survive (see [`PUBLISH_MAX`]).
    pub fn publish_set(&self) -> Vec<(u32, String)> {
        sample(&self.cuts, PUBLISH_MAX)
    }

    /// Number of addressed cuts — the ladder depth actually available.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }
}

/// Evenly spaced subsequence of `items` of length at most `k`, always including
/// the first and last element.
fn sample<T: Clone>(items: &[T], k: usize) -> Vec<T> {
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    if items.len() <= k {
        return items.to_vec();
    }
    if k == 1 {
        return vec![items[items.len() - 1].clone()];
    }
    let last = items.len() - 1;
    let mut out: Vec<T> = (0..k).map(|i| items[i * last / (k - 1)].clone()).collect();
    out.truncate(k);
    out
}

/// Page-aligned cut points for `boundaries`, ascending and deduplicated.
///
/// Aligns DOWN: a cut past the KV that backs it is unusable, a cut short of it
/// merely reuses less. Drops zero — an empty prefix reuses nothing.
///
/// `history_len` itself IS a candidate when it is page-aligned: resuming there
/// leaves the cue as the whole delta, which is the best case, not a degenerate
/// one. (The session arm excludes its own tip because its tip is the full
/// render INCLUDING the cue, where the delta really would be empty.)
pub fn aligned_cuts(boundaries: &[u32], page: u32, history_len: u32) -> Vec<u32> {
    if page == 0 {
        return Vec::new();
    }
    let mut cuts: Vec<u32> = boundaries
        .iter()
        .filter(|&&b| b <= history_len)
        .map(|&b| (b / page) * page)
        .filter(|&c| c > 0)
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Park the history prefix under its address.
///
/// `cut` must be the page-aligned token count this address was computed over,
/// and `ws` must already hold at least that many tokens of KV.
///
/// Publishes a SLICE, not the working set: the live set also holds the
/// generation cue and the tokens about to be generated, and the next render of
/// this conversation reproduces neither.
///
/// Failure is logged and swallowed for the same reason a miss is: the turn is
/// correct either way, and a cache that cannot park is a slow turn, not a
/// wrong one.
pub fn publish(pipe: &Pipeline, ws: &WorkingSet, cut: u32, address: &str, page: u32) -> bool {
    if page == 0 || cut == 0 || cut % page != 0 {
        eprintln!("[apc] refusing to publish an unaligned cut {cut} (page {page})");
        return false;
    }
    let pages = cut / page;
    if ws.page_len() < pages {
        eprintln!(
            "[apc] refusing to publish {pages} page(s) from a {}-page set",
            ws.page_len()
        );
        return false;
    }
    match ws.slice(pipe, 0, pages) {
        Ok(prefix) => match prefix.update_index(address.as_bytes()) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[apc] update_index failed at cut {cut}: {e}");
                false
            }
        },
        Err(e) => {
            eprintln!("[apc] slice(0, {pages}) failed: {e}");
            false
        }
    }
}

/// Reuse fraction, the number to report instead of a hit flag.
///
/// RatioThink shipped two prefix-cache defects that both reported a healthy
/// `outcome == "hit"` while doing the cold turn's work — one left some boundary
/// reachable and re-prefilled most of the history anyway (a 25.7× TTFT
/// regression behind a green check). A boolean cannot see that; a fraction can.
pub fn reuse_fraction(cached_tokens: u32, total_prompt: u32) -> f32 {
    if total_prompt == 0 {
        return 0.0;
    }
    cached_tokens as f32 / total_prompt as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_align_down_and_run_ascending() {
        // page 16: 7 -> 0 (dropped), 20 -> 16, 33 -> 32, 40 -> 32 (dup).
        assert_eq!(aligned_cuts(&[7, 20, 33, 40], 16, 100), vec![16, 32]);
    }

    #[test]
    fn an_aligned_history_tip_is_a_candidate() {
        // Resuming at the whole history leaves the cue as the delta — the best
        // case. Only cuts PAST the history are dropped, because their KV would
        // cover generation scaffolding no later render reproduces.
        assert_eq!(aligned_cuts(&[64], 16, 64), vec![64]);
        assert!(aligned_cuts(&[80], 16, 64).is_empty());
    }

    #[test]
    fn a_zero_page_size_yields_no_candidates_rather_than_dividing_by_zero() {
        assert!(aligned_cuts(&[16, 32], 0, 100).is_empty());
    }

    #[test]
    fn one_park_per_turn_is_the_deepest_cut() {
        let cuts: Vec<u32> = (1..=40).map(|i| i * 256).collect();
        let picked = sample(&cuts, PUBLISH_MAX);
        // Whatever PUBLISH_MAX is, the DEEPEST cut must survive the sampling:
        // it is the one the next turn of this conversation resumes at, and it
        // is the only reuse this arm has actually measured.
        assert_eq!(picked.last(), cuts.last());
        assert!(picked.len() <= PUBLISH_MAX);
    }

    #[test]
    fn the_sample_keeps_both_ends_when_it_has_room() {
        // The policy itself is still ends-preserving; only the budget shrank.
        // Kept as a test so raising PUBLISH_MAX restores cross-conversation
        // reach rather than eight copies of the tail.
        let cuts: Vec<u32> = (1..=40).map(|i| i * 256).collect();
        let picked = sample(&cuts, 8);
        assert_eq!(picked.len(), 8);
        assert_eq!(picked[0], cuts[0]);
        assert_eq!(picked[7], cuts[cuts.len() - 1]);
        assert!(picked.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn a_ladder_shorter_than_the_budget_is_published_whole() {
        assert_eq!(sample(&[16u32, 32, 48], 8), vec![16, 32, 48]);
        // A render with nothing to park must not park something anyway.
        assert!(sample::<u32>(&[], PUBLISH_MAX).is_empty());
        assert!(sample::<u32>(&[], 8).is_empty());
    }

    #[test]
    fn reuse_fraction_is_zero_on_an_empty_prompt() {
        assert_eq!(reuse_fraction(0, 0), 0.0);
        assert_eq!(reuse_fraction(8_192, 8_192), 1.0);
        assert!((reuse_fraction(11_041, 11_237) - 0.982).abs() < 0.001);
    }
}
