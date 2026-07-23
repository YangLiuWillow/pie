//! Content-addressed KV **prefix cache** ("automatic prefix caching") for the
//! openevolve-generation inferlet.
//!
//! A snapshot's name embeds a hash of the *exact* token sequence its KV holds,
//! so a name match ⇒ identical tokens for the same model ⇒ identical KV. This
//! replaces the earlier host-coordinated key (`topk_sig`, a backend hash of the
//! top-K ids + parent code + metrics): the inferlet now self-keys L1p/L1g from
//! the actual rendered prompt tokens, so cross-worker reuse no longer depends on
//! the backend computing a matching key — a different render simply changes the
//! hash and misses cleanly.
//!
//! Shared with openhands-coder-session's `prefix_cache` in spirit; kept as a
//! small local copy so each inferlet stays self-contained (no shared crate).

/// Template / tokenizer format marker folded into every snapshot name. Bump
/// this whenever the L1p render for the same sections could change (chat
/// template, section join, tokenizer) so stale snapshots cleanly miss.
pub const TEMPLATE_MARKER: &str = "oe-generation/qwen3coder/v1";

const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
const FNV64_OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
// Distinct basis for lane B (any offset works; this is A byte-reversed).
const FNV64_OFFSET_B: u64 = 0x2523_2284_e49c_f2cb;

#[inline]
fn fnv64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV64_PRIME);
    }
    h
}

/// Stable 128-bit-rendered content digest (two correlated FNV-1a lanes),
/// emitted as 32 lowercase hex chars. No external crate — deterministic across
/// builds/processes, which a snapshot name keyed across short-lived per-call
/// wasm instances (and across isolated worker processes) requires.
pub fn content_hash(model_id: &str, template: &str, prefix_tokens: &[u32]) -> String {
    // 0xFF is not a valid UTF-8 continuation lead, so it cannot appear inside
    // the model/template strings — a clean domain separator that prevents
    // field-shift ambiguity, e.g. ("ab","c") vs ("a","bc").
    let sep = [0xFFu8];
    let mut a = fnv64(FNV64_OFFSET_A, model_id.as_bytes());
    let mut b = fnv64(FNV64_OFFSET_B, model_id.as_bytes());
    a = fnv64(a, &sep);
    b = fnv64(b, &sep);
    a = fnv64(a, template.as_bytes());
    b = fnv64(b, template.as_bytes());
    a = fnv64(a, &sep);
    b = fnv64(b, &sep);
    for &t in prefix_tokens {
        let le = t.to_le_bytes();
        a = fnv64(a, &le);
        b = fnv64(b, &le);
    }
    format!("{a:016x}{b:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_prefix_sensitive() {
        let a = content_hash("m", TEMPLATE_MARKER, &[1, 2, 3]);
        assert_eq!(a, content_hash("m", TEMPLATE_MARKER, &[1, 2, 3]));
        assert_ne!(a, content_hash("m", TEMPLATE_MARKER, &[1, 2, 3, 4]));
        assert_ne!(a, content_hash("other", TEMPLATE_MARKER, &[1, 2, 3]));
        assert_ne!(a, content_hash("m", "other/v2", &[1, 2, 3]));
    }

    #[test]
    fn field_separation_prevents_shift_ambiguity() {
        assert_ne!(content_hash("ab", "c", &[]), content_hash("a", "bc", &[]));
    }
}
