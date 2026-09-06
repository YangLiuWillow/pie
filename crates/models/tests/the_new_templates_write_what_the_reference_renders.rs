//! The ATEM (Muse Glimmer) and Inkling templates write the token sequences
//! transformers' `apply_chat_template` renders for the same turn, checked
//! against the real tokenizers (the miniature snapshots' `tokenizer.json`,
//! skipped when a box does not hold them). The expected ids below are the
//! reference renderings, recorded once:
//!
//! ```text
//! tok.apply_chat_template([{"role": "user", "content": "What is 17 times 23?"}],
//!                         add_generation_prompt=True)
//! ```
//!
//! Inkling's is the whole rendering (the effort line, the user message, the
//! cue). Glimmer's reference injects a dated system block first, which is
//! not a template's to write; the user turn and cue after it are what the
//! template writes, and they are compared as that suffix.

use std::path::PathBuf;
use std::sync::Arc;

use chat_template::Instruct;
use tokenizer::Tokenizer;

const MESSAGE: &str = "What is 17 times 23?";

fn snapshot(repo: &str, revision: &str) -> Option<Arc<Tokenizer>> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let path = home
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join("tokenizer.json");
    if !path.is_file() {
        eprintln!("skipping: no tokenizer at {}", path.display());
        return None;
    }
    Some(Arc::new(Tokenizer::from_file(&path).expect("the tokenizer loads")))
}

#[test]
fn inkling_writes_the_effort_line_the_user_turn_and_the_cue() {
    let Some(tokenizer) = snapshot("thinkingmachines/Inkling", "mini-l7-e8") else {
        return;
    };
    let template = chat_template::inkling::Inkling::new(tokenizer);
    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200002, 200004, 133850, 6942, 3211, 25, 220, 15, 13, 24, 200010, 200000, 200004, 4827,
        382, 220, 1422, 4238, 220, 1860, 30, 200010, 200001,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![200006, 199999]);
}

#[test]
fn atem_writes_the_user_turn_and_the_cue_after_the_opening() {
    let Some(tokenizer) = snapshot("meta-models/Muse-Glimmer-30B", "mini-l8-ends") else {
        return;
    };
    let template = chat_template::atem::Atem::new(tokenizer);
    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 1556, 200023, 3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022,
        140680,
    ];
    assert_eq!(got, want);
    assert_eq!(template.prefix(), vec![200000]);
    assert_eq!(template.seal(), vec![200008, 200001]);

    // A system turn: the reference appends the reasoning strength (when the
    // message states none) and the recipient list before closing it.
    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 15651, 200023, 4662, 583, 260, 6201, 1574, 34956, 300, 9762, 38, 2244,
        1574, 15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023,
        3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022, 140680,
    ];
    assert_eq!(got, want);
    // A stated strength is kept, not doubled.
    let mut got = template.system_user("You are terse.\n\nReasoning strength: low.", MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 15651, 200023, 4662, 583, 260, 6201, 1574, 34956, 300, 9762, 38, 4463,
        1574, 15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023,
        3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022, 140680,
    ];
    assert_eq!(got, want);
}
