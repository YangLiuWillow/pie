mod common;

use std::sync::Arc;

use common::{MergeFormat, byte_level_json, gemma_json, unigram_json};
use serde_json::json;
use tokenizer::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn assert_exact(json: &serde_json::Value, texts: &[&str]) {
    let bytes = serde_json::to_vec(json).unwrap();
    let pie = Arc::new(
        std::str::from_utf8(&bytes)
            .unwrap()
            .parse::<Tokenizer>()
            .unwrap(),
    );
    let hf = HfTokenizer::from_bytes(&bytes).unwrap();

    assert_eq!(pie.vocab_size(), hf.get_vocab_size(true));
    for &text in texts {
        let ids = pie.encode(text);
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ids, hf_ids, "encoding {text:?}");
        assert_eq!(
            pie.decode(&hf_ids, false),
            hf.decode(&hf_ids, false).unwrap(),
            "HF→Pie decoding {text:?}"
        );
        assert_eq!(
            pie.decode(&ids, false),
            hf.decode(&ids, false).unwrap(),
            "Pie→HF decoding {text:?}"
        );
        assert_eq!(
            pie.decode(&hf_ids, true),
            hf.decode(&hf_ids, true).unwrap(),
            "special-token filtering {text:?}"
        );

        let mut decoder = pie.decoder(false);
        let mut incremental = String::new();
        for token in &hf_ids {
            incremental.push_str(&decoder.feed(std::slice::from_ref(token)));
        }
        incremental.push_str(&decoder.finish());
        assert_eq!(
            incremental,
            pie.decode(&hf_ids, false),
            "incremental decoding {text:?}"
        );
    }
}

#[test]
fn qwen3_profile_is_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        false,
        MergeFormat::Tuple,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "a\u{0301}", "<|special|>abc"]);
}

#[test]
fn qwen36_string_merges_are_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[\p{L}\p{M}]+|[^\p{L}\p{M}\p{N}]+"],
        false,
        MergeFormat::String,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "a\u{0301}", "Hello!"]);
}

#[test]
fn glm_and_nemotron_ignore_merges_are_exact() {
    let tokenizer = byte_level_json(
        serde_json::Value::Null,
        &[r"\p{N}{1,3}|[^\p{N}]+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "abc<|special|>"]);

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.encode("abc"), vec![259]);
    assert_eq!(pie.encode("1234"), vec![257, b'4' as u32]);
}

#[test]
fn deepseek_multi_regex_profile_is_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "Sequence", "normalizers": []}),
        &[
            r"\p{N}{1,3}",
            r"[一-龥぀-ゟ゠-ヿ]+",
            r"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s+",
        ],
        false,
        MergeFormat::Tuple,
        true,
    );
    assert_exact(&tokenizer, &["abc", "1234", "你好1234", "<|special|>abc"]);
}

#[test]
fn gemma_byte_fallback_profile_is_exact() {
    let tokenizer = gemma_json();
    assert_exact(&tokenizer, &["a b", "叫", "<special>a b"]);

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.decode(&[0xE5 + 6, 0x8F + 6], false), "��");
}

#[test]
fn grammar_bytes_are_decoder_aware_and_exclude_specials() {
    let byte_level = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    let pie: Tokenizer = byte_level.to_string().parse().unwrap();
    assert_eq!(pie.decoded_token_bytes(0xC3), Some(&[0xC3][..]));
    assert_eq!(pie.decoded_token_bytes(0xA9), Some(&[0xA9][..]));
    assert_eq!(pie.decoded_token_bytes(260), None);
    assert!(!pie.sorted_token_ids().contains(&260));

    let gemma: Tokenizer = gemma_json().to_string().parse().unwrap();
    assert_eq!(gemma.decoded_token_bytes(4), Some(&b"a "[..]));
    assert_eq!(gemma.decoded_token_bytes(6 + 0xE5), Some(&[0xE5][..]));
}

/// **THE UNIGRAM WALK AND ITS WRAPPING, AGAINST HUGGING FACE ITSELF.**
///
/// `assert_exact` is the judge: pie's ids must equal `tokenizers`' ids and
/// both decodes must agree, for every string. What each string is for:
///
/// - `"ab"` — the walk must not be greedy. `ab` is one piece and matches
///   further, and it scores worse than `a` + `b`.
/// - `"a red"` and `"red"` — `prepend_scheme = "always"` means a leading word
///   and an interior one take the SAME piece, which is the whole point of the
///   scheme and the thing a `FirstSegment` prefix would get wrong.
/// - `"a  b"` — the normalizer collapses the run of spaces before anything
///   else sees it.
/// - `""` — a template post-processor still appends its tail; the answer is
///   `[</s>]` and not nothing.
/// - `"aQb"` — `Q` is in no piece, so it is one `unk` and the walk carries on
///   past it. Byte fallback is off, so it is not a run of `<0xNN>`.
///
/// Compared against `tokenizers` WITH special tokens, unlike `assert_exact`'s
/// other callers. That is not a loosening: a `TemplateProcessing` post-
/// processor is part of what the model reads, and umT5's answer for any
/// string ends with `</s>`. The BPE families here have no post-processor —
/// their specials arrive as chat-template TEXT — which is why the shared
/// helper asks for none.
#[test]
fn a_unigram_walks_and_wraps_exactly_as_hugging_face_does() {
    let json = unigram_json();
    let bytes = serde_json::to_vec(&json).unwrap();
    let pie = std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap();
    let hf = HfTokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(pie.vocab_size(), hf.get_vocab_size(true));

    for text in ["ab", "a red", "red", "a  b", "", "aQb", "abab", "d"] {
        let ids = pie.encode(text);
        let hf_ids = hf.encode(text, true).unwrap().get_ids().to_vec();
        assert_eq!(ids, hf_ids, "encoding {text:?}");
        assert_eq!(
            pie.decode(&ids, true),
            hf.decode(&ids, true).unwrap(),
            "decoding {text:?}"
        );
    }
}
