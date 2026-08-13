//! Content addressing over **rendered token ids**, and the prefix-boundary
//! scan that turns one render into every candidate resume point.
//!
//! ## Why this exists next to [`crate::session`]
//!
//! `session::snapshot_address` hashes canonicalized *messages* plus tool
//! schemas. That is one level removed from what the KV actually holds, and the
//! gap is a real hazard: a chat-template edit, a `cue_no_think` change or a
//! tokenizer swap moves the tokens but **not** the message-level address, so a
//! resume would hand the model KV rendered by the old template — fluent, and
//! wrong in a way nothing else would catch.
//!
//! Hashing the token ids closes it by construction. In the words of the
//! OpenHands port, which arrived here first
//! (`openhands-integration-updated:inferlets/openhands-coder-session/src/prefix_cache.rs`):
//!
//! > a snapshot's name is a hash of the *exact* token sequence its KV holds, so
//! > a name match ⇒ identical tokens for the same model ⇒ identical KV. A false
//! > hit is impossible; any divergence (model, template, history, tools)
//! > changes the tokens ⇒ changes the hash ⇒ a clean miss.
//!
//! [`TEMPLATE_MARKER`] carries the part the token ids cannot: bump it whenever
//! the same messages could render to different tokens for reasons the ids do
//! not reveal.
//!
//! ## Why the digest is streaming
//!
//! A turn wants the address of *many* prefixes of one render — every safe
//! render-unit boundary, so a retry or a truncation can re-hit an earlier one.
//! FNV-1a is a streaming hash, so [`prefix_addresses`] walks the token stream
//! once and snapshots the running state at each boundary: N addresses for the
//! cost of one pass, instead of N passes.

/// Folded into every address alongside the model id. Bump on any change that
/// could make the same messages render to different tokens — a new chat
/// template, different tool-schema wrapping, a tokenizer swap — so stale state
/// misses cleanly instead of returning KV for a different byte stream.
pub const TEMPLATE_MARKER: &str = "opencode-session/qwen-chatml/v1";

const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
const FNV64_OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
/// Distinct basis for lane B (any offset works; this is A byte-reversed).
const FNV64_OFFSET_B: u64 = 0x2523_2284_e49c_f2cb;

/// Two correlated FNV-1a lanes over a token stream, rendered as 32 hex chars.
///
/// The lanes share [`FNV64_PRIME`] and differ only in offset basis, so their
/// effective width is wider than one 64-bit lane but is not a proven 2^-128.
/// That is ample here: a collision can only mislead *within one model and
/// template*, between two token streams the same conversation actually
/// produces, and every hit is additionally gated on the retained length
/// matching the boundary — so a collision has to agree on the length too.
#[derive(Clone, Copy, Debug)]
pub struct TokenDigest {
    a: u64,
    b: u64,
}

impl TokenDigest {
    /// Seed with the model identity and template marker. Two different models
    /// must never share an address even for identical token ids, because the
    /// KV those ids produce is not the same KV.
    pub fn new(model_id: &str, template: &str) -> Self {
        let mut d = TokenDigest { a: FNV64_OFFSET_A, b: FNV64_OFFSET_B };
        d.write(model_id.as_bytes());
        // 0xFF cannot appear in UTF-8, so it is a domain separator no field
        // content can forge — without it ("ab","c") and ("a","bc") collide.
        d.write(&[0xFF]);
        d.write(template.as_bytes());
        d.write(&[0xFF]);
        d
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.a ^= byte as u64;
            self.a = self.a.wrapping_mul(FNV64_PRIME);
            self.b ^= byte as u64;
            self.b = self.b.wrapping_mul(FNV64_PRIME);
        }
    }

    /// Absorb tokens, little-endian. Token ids rather than decoded text: the
    /// ids are what the KV is keyed on, and two different id sequences can
    /// decode to the same string.
    pub fn update(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.write(&t.to_le_bytes());
        }
    }

    /// The address of everything absorbed so far. Cheap and non-consuming, so
    /// a streaming walk can snapshot at every boundary.
    pub fn finish(&self) -> String {
        format!("{:016x}{:016x}", self.a, self.b)
    }
}

/// Address every prefix of `tokens` named by `boundaries`, in one pass.
///
/// `boundaries` are token counts, and must be ascending and within
/// `tokens.len()`; out-of-range entries are skipped rather than clamped, since
/// a boundary past the render is a caller bug and silently addressing a
/// shorter prefix would be worse than returning nothing for it.
///
/// Returns `(boundary, address)` pairs in the order given.
pub fn prefix_addresses(
    model_id: &str,
    template: &str,
    tokens: &[u32],
    boundaries: &[u32],
) -> Vec<(u32, String)> {
    let mut out = Vec::with_capacity(boundaries.len());
    let mut digest = TokenDigest::new(model_id, template);
    let mut pos = 0usize;
    for &b in boundaries {
        let end = b as usize;
        if end > tokens.len() || end < pos {
            continue;
        }
        digest.update(&tokens[pos..end]);
        pos = end;
        out.push((b, digest.finish()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_is_stable_and_length_sensitive() {
        let toks: Vec<u32> = (0..100).collect();
        let a = prefix_addresses("m", TEMPLATE_MARKER, &toks, &[100]);
        let b = prefix_addresses("m", TEMPLATE_MARKER, &toks, &[100]);
        assert_eq!(a, b);
        assert_eq!(a[0].1.len(), 32);

        // A different length is a different address, even though one stream is
        // a prefix of the other.
        let short = prefix_addresses("m", TEMPLATE_MARKER, &toks, &[99]);
        assert_ne!(a[0].1, short[0].1);
    }

    #[test]
    fn model_and_template_are_part_of_the_address() {
        let toks: Vec<u32> = (0..40).collect();
        let base = prefix_addresses("m1", TEMPLATE_MARKER, &toks, &[40]);
        // Same tokens, different model: the KV is not interchangeable.
        assert_ne!(base, prefix_addresses("m2", TEMPLATE_MARKER, &toks, &[40]));
        // Same tokens, bumped template marker: this is the template-drift
        // guard the message-level address cannot provide.
        assert_ne!(base, prefix_addresses("m1", "other/v2", &toks, &[40]));
    }

    #[test]
    fn streaming_prefixes_match_addressing_each_prefix_alone() {
        let toks: Vec<u32> = (0..64).map(|i| i * 7 + 3).collect();
        let bounds = [8u32, 16, 40, 64];
        let streamed = prefix_addresses("m", TEMPLATE_MARKER, &toks, &bounds);
        for (b, addr) in streamed {
            let alone = prefix_addresses("m", TEMPLATE_MARKER, &toks[..b as usize], &[b]);
            assert_eq!(alone[0].1, addr, "boundary {b} disagreed with a single-shot hash");
        }
    }

    #[test]
    fn field_separator_prevents_shift_collisions() {
        let toks: Vec<u32> = vec![1, 2, 3];
        // Without the 0xFF separator these would collide.
        let x = prefix_addresses("ab", "c", &toks, &[3]);
        let y = prefix_addresses("a", "bc", &toks, &[3]);
        assert_ne!(x, y);
    }

    #[test]
    fn out_of_range_boundaries_are_skipped_not_clamped() {
        let toks: Vec<u32> = (0..10).collect();
        let got = prefix_addresses("m", TEMPLATE_MARKER, &toks, &[5, 50]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 5);
    }

    #[test]
    fn token_ids_are_hashed_not_decoded_text() {
        // Two ids that could plausibly decode to the same string must not
        // share an address.
        let a = prefix_addresses("m", TEMPLATE_MARKER, &[1u32, 2], &[2]);
        let b = prefix_addresses("m", TEMPLATE_MARKER, &[2u32, 1], &[2]);
        assert_ne!(a, b);
    }
}
