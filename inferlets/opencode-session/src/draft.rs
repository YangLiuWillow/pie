//! Prompt-lookup drafting: guess the next few tokens by finding where the
//! recent output has occurred before, and copying what followed.
//!
//! ## Why this, on this workload
//!
//! Decode is where an agent turn's time goes — 69% of model time on
//! `django-13089`, measured by regression over 44 calls (R² 0.984). The cost
//! per token splits into a fixed term (~7.4 ms, the MoE weight read, already
//! running at ~203 GB/s and therefore near this machine's peak) and a context
//! term (~3.21 ms per 1k tokens of KV). Qwen3-Coder-30B carries 96 KiB of KV
//! per token across its 48 layers, so at a 22k-token context a decode step
//! reads over 2 GB and lands at ~12.5 tok/s.
//!
//! Nothing in that is fixable by decoding *faster* — the weight half is at
//! hardware speed. It is fixable by decoding *fewer times*: a verified draft
//! pays one weight read and one KV read for every token it gets accepted.
//!
//! ## Why a lookup table rather than a draft model
//!
//! A draft model costs weights, memory and a second forward pass. Prompt lookup
//! costs a hash map over tokens the turn already has. It works when the output
//! copies the input, and an agent turn copies constantly: file paths it just
//! listed, identifiers from the code it just read, the diff it is echoing back.
//! Where it does not apply it drafts nothing and the loop degrades to exactly
//! the per-token path it replaces.
//!
//! ## The property that makes this safe
//!
//! The draft is never trusted. Every drafted token is checked against the
//! model's own argmax in the same fire (`specverify`'s cross-row `cumprod`
//! prefix-AND), and a mismatch at row j discards the whole suffix. So output is
//! identical to what the per-token loop would have produced, and a bad drafter
//! costs throughput, never correctness.

/// How many tokens to propose at once.
///
/// Each drafted token adds a row to the verify fire, and rows are nearly free
/// against the KV read they share — but an unaccepted row is wasted embed and
/// projection work, so the useful ceiling is set by how far ahead a copy
/// typically runs. Four matches `specverify`'s default and the lengths that
/// actually repeat in agent output (a path segment, an identifier, a short
/// code span).
pub const DRAFT_K: usize = 4;

/// Length of the token suffix used as the lookup key.
///
/// Too short and it matches noise — a 1-gram key matches everywhere and drafts
/// garbage. Too long and it never fires on the first repetition of a phrase.
/// Two independent n-gram sizes are tried, longest first, so a precise match
/// wins when it exists and a looser one still fires when it does not.
const NGRAM_LONG: usize = 3;
const NGRAM_SHORT: usize = 2;

/// Propose up to [`DRAFT_K`] continuations of `tail`, by finding the most
/// recent earlier occurrence of its last n tokens in `haystack`.
///
/// `haystack` is the turn's token history (prompt + what has been generated so
/// far); `tail` is the same sequence's end. Searching from the RIGHT is
/// deliberate: the most recent occurrence is the most likely continuation in a
/// conversation that is currently talking about one thing.
///
/// Returns an empty vector when nothing matches, which the caller must treat as
/// "decode one token the ordinary way" rather than as an error.
pub fn draft(haystack: &[u32], tail: &[u32], k: usize) -> Vec<u32> {
    for n in [NGRAM_LONG, NGRAM_SHORT] {
        if tail.len() < n {
            continue;
        }
        let key = &tail[tail.len() - n..];
        // `saturating_sub(1)` keeps the search off the key's own position: a
        // window that ends where the tail ends would "predict" the tokens that
        // are already there.
        let last = haystack.len().saturating_sub(n).saturating_sub(1);
        for start in (0..=last).rev() {
            if &haystack[start..start + n] != key {
                continue;
            }
            let from = start + n;
            let take = k.min(haystack.len().saturating_sub(from));
            if take > 0 {
                return haystack[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

/// Running acceptance, reported per turn.
///
/// A speedup number alone cannot tell "drafting works" from "drafting is off
/// and something else got cheaper" — the same trap as reporting a prefix-cache
/// hit flag instead of a reuse fraction. Acceptance is the honest quantity:
/// `accepted / proposed` is what actually converts into throughput, and
/// `fires_saved` is what it bought.
#[derive(Default, Clone, Copy)]
pub struct Accounting {
    pub proposed: u32,
    pub accepted: u32,
    pub verify_fires: u32,
}

impl Accounting {
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        self.proposed += proposed as u32;
        self.accepted += accepted as u32;
        self.verify_fires += 1;
    }

    pub fn acceptance(&self) -> f32 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f32 / self.proposed as f32
        }
    }

    /// Decode fires avoided: every accepted draft token is a fire not run.
    pub fn fires_saved(&self) -> u32 {
        self.accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_the_continuation_of_a_repeated_phrase() {
        // "pkg/mod_7.py" said once, then begun again — the draft should finish it.
        let hay: Vec<u32> = vec![10, 11, 12, 13, 14, 99, 10, 11, 12];
        let draft = draft(&hay, &hay, 4);
        assert_eq!(draft, vec![13, 14, 99, 10]);
    }

    #[test]
    fn prefers_the_most_recent_occurrence() {
        // Same key twice with different continuations; the later one wins,
        // because a conversation's recent context predicts better than its old.
        let hay: Vec<u32> = vec![1, 2, 3, 40, 41, 1, 2, 3, 50, 51, 1, 2, 3];
        assert_eq!(draft(&hay, &hay, 2), vec![50, 51]);
    }

    #[test]
    fn falls_back_to_a_shorter_key_before_giving_up() {
        // No 3-gram match for [7,8,9]; the 2-gram [8,9] does occur earlier.
        let hay: Vec<u32> = vec![8, 9, 77, 78, 5, 6, 7, 8, 9];
        assert_eq!(draft(&hay, &hay, 2), vec![77, 78]);
    }

    #[test]
    fn drafts_nothing_rather_than_guessing() {
        // Nothing repeats: the caller must fall back to ordinary decoding.
        let hay: Vec<u32> = vec![1, 2, 3, 4, 5];
        assert!(draft(&hay, &hay, 4).is_empty());
        // And a tail shorter than the shortest key cannot be looked up at all.
        assert!(draft(&[1], &[1], 4).is_empty());
    }

    #[test]
    fn never_predicts_from_the_keys_own_position() {
        // The key sits at the very end. Matching it there would "predict" the
        // tokens already present and always appear to be a perfect draft — the
        // exact self-fulfilling bug this search's bounds exist to prevent.
        let hay: Vec<u32> = vec![4, 5, 6];
        assert!(draft(&hay, &hay, 2).is_empty());
    }

    #[test]
    fn acceptance_is_a_fraction_not_a_flag() {
        let mut a = Accounting::default();
        a.record(4, 4);
        a.record(4, 0); // a miss must drag the number down
        a.record(4, 2);
        assert_eq!(a.proposed, 12);
        assert_eq!(a.accepted, 6);
        assert!((a.acceptance() - 0.5).abs() < 1e-6);
        assert_eq!(a.fires_saved(), 6);
        assert_eq!(Accounting::default().acceptance(), 0.0);
    }
}
