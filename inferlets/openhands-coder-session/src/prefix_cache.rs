//! Content-addressed cross-call KV **prefix cache** ("automatic prefix
//! caching") for the OpenHands coder-session inferlet.
//!
//! Adapted from RatioThink's `chat-apc/src/chat/prefix_cache.rs`. The idea:
//! a snapshot's name is a hash of the *exact* token sequence its KV holds, so
//! a name match ⇒ identical tokens for the same model ⇒ identical KV. A false
//! hit is impossible; any divergence (model, template, history, tools) changes
//! the tokens ⇒ changes the hash ⇒ a clean miss.
//!
//! This replaces the previous *host-coordinated single-slot* scheme (one
//! `oh-session-{id}` snapshot refreshed in place, with the host echoing
//! `session_prev_len`/`session_prev_hash`). Content addressing makes the
//! inferlet self-keying and lets many boundaries coexist, so retry / branch /
//! truncate re-hit their still-valid earlier boundary automatically instead of
//! rebuilding, and delegation (a sub-agent that shares a task prefix) hits the
//! parent's boundary with no explicit fork protocol.
//!
//! # Name schema
//!
//! ```text
//! name = "apc/{key}/{compat}/{hex(hash(model_id ‖ template ‖ prefix_token_ids))}"
//! ```
//!
//! - `key`    — per-conversation namespace (the session id). Isolates chats.
//! - `compat` — a version marker the host bumps to invalidate everything on
//!              app-schema / template drift. Empty ⇒ `"0"`.
//! - `hash`   — 128-bit-rendered digest over the model id, the template marker,
//!              and the prefix token ids.
//!
//! # Correctness
//!
//! The caller only ever names candidate prefixes by slicing its own full
//! render at recorded render-unit boundaries, so a candidate is a literal
//! token prefix of the full render by construction — a bad split can never
//! plant a wrong suffix. It additionally gates each hit on the opened
//! snapshot's `seq_len()` equalling the sliced length, so a name collision or
//! a truncated snapshot is rejected rather than trusted. Snapshot loss (engine
//! restart) and cross-model/tokenizer drift are natural misses.

/// Template / tokenizer format marker folded into every snapshot name. Bump
/// this whenever `render_prompt`'s output for the same messages could change
/// (new chat template, tool-schema wrapping, tokenizer) so stale snapshots
/// cleanly miss instead of returning KV for a different byte stream.
pub const TEMPLATE_MARKER: &str = "oh-coder-session/qwen3coder/v1";

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
/// builds/processes, which a snapshot name keyed across the short-lived
/// per-call wasm instances requires.
///
/// The lanes share [`FNV64_PRIME`] and differ only in offset basis, so their
/// effective collision resistance is wider than one 64-bit lane but is not a
/// proven 2^-128. That is ample here: a collision can only mislead within the
/// same `(key, compat, model)` namespace — two distinct histories the same
/// conversation actually sends — for which 64+ effective bits is far out of
/// reach.
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

/// Namespace every snapshot for `(key, compat)` lives under, with the trailing
/// `/` that makes `Context::delete` treat it as a prefix.
///
/// Teardown deletes this one name instead of the per-call names: those are
/// derived from token content nobody retains after the call, so enumerating
/// them later is impossible. Built from the same pieces as [`snapshot_name`]
/// so the two can't drift out of agreement.
pub fn namespace(key: &str, compat: &str) -> String {
    let compat = if compat.is_empty() { "0" } else { compat };
    format!("apc/{key}/{compat}/")
}

/// Full snapshot name: `apc/{key}/{compat}/{content_hash}`.
pub fn snapshot_name(key: &str, compat: &str, model_id: &str, prefix_tokens: &[u32]) -> String {
    let h = content_hash(model_id, TEMPLATE_MARKER, prefix_tokens);
    format!("{}{h}", namespace(key, compat))
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
        assert_ne!(
            content_hash("ab", "c", &[]),
            content_hash("a", "bc", &[]),
        );
    }

    #[test]
    fn name_shape() {
        let n = snapshot_name("chatA", "", "model-x", &[1, 2, 3]);
        assert!(n.starts_with("apc/chatA/0/"));
        assert_eq!(n.rsplit('/').next().unwrap().len(), 32);
    }

    /// Teardown deletes the namespace; every name the session saved must sit
    /// under it, or its KV pages leak.
    #[test]
    fn namespace_is_a_prefix_of_every_name_it_covers() {
        let ns = namespace("chatA", "v2");
        assert!(ns.ends_with('/'), "prefix delete needs the trailing slash");
        for toks in [&[][..], &[1][..], &[1, 2, 3][..]] {
            assert!(snapshot_name("chatA", "v2", "model-x", toks).starts_with(&ns));
        }
        // Another conversation's namespace must not be swept up with it.
        assert!(!snapshot_name("chatB", "v2", "model-x", &[1]).starts_with(&ns));
        // Nor another compat generation's.
        assert!(!snapshot_name("chatA", "v3", "model-x", &[1]).starts_with(&ns));
    }
}
