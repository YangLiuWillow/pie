//! pie's rendered token ids against the checkpoint's own chat template.
//!
//! The oracle is HuggingFace `apply_chat_template`, and it does NOT run here.
//! It ran once, deliberately, when a human regenerated
//! `tests/fixtures/render/` — the same trade
//! `model/config/tests/differential.rs` makes against the C++ normalizer it
//! replaces, and `scripts/sync-wit.sh` makes against vendored WIT: the foreign
//! toolchain runs when someone asks, and the PR path only ever reads
//! checked-in bytes.
//!
//! Why the oracle stays foreign: pie's renderer is a hand-written Rust
//! reimplementation of a Jinja template, and the thing being tested is whether
//! the reimplementation drifted. Checking it with a RUST Jinja engine would
//! compare pie's Rust against more Rust, where a shared assumption passes
//! silently — and if pie ever adopted that engine, the test would be the engine
//! against itself, which proves determinism and nothing about fidelity.
//!
//! What this costs to run: no Python, no network, no HuggingFace cache. The
//! tokenizers are carried as `pie.tokenizer/1`, pie's own compiled form, which
//! is a third of the bytes of `tokenizer.json` and loads ~400x faster (96–138 ms
//! parsing HF JSON, under a millisecond here).
//!
//! Regenerate with:
//!
//! ```text
//! python integrations/opencode/parity/check_render_matrix.py \
//!     --bin target/debug/render-tokens --emit-fixtures tests/fixtures/render
//! ```
//!
//! A diff in the fixtures is NEWS, not noise: it means a checkpoint's template
//! moved. Read it before committing it, and never hand-edit ids to make this
//! pass.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pie_model_common::instruct::Instruct;
use pie_openai_serving::render::{RenderOp, plan_render};
use pie_openai_serving::types::ChatCompletionRequest;
use pie_tokenizer::Tokenizer;
use pie_tokenizer::canonical::CanonicalTokenizer;
use serde_json::Value;
use std::sync::Arc;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/fixtures/render")
}

/// Read the flat container `render-tokens --compile` writes: per object, a
/// little-endian `u32` name length, the name, a `u32` data length, the data.
fn load_pietok(path: &Path) -> Arc<Tokenizer> {
    let blob = std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut objects: HashMap<String, Vec<u8>> = HashMap::new();
    let mut at = 0usize;
    while at < blob.len() {
        let take = |at: &mut usize, n: usize| -> Vec<u8> {
            let out = blob[*at..*at + n].to_vec();
            *at += n;
            out
        };
        let n = u32::from_le_bytes(take(&mut at, 4).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&mut at, n)).expect("object name is utf-8");
        let n = u32::from_le_bytes(take(&mut at, 4).try_into().unwrap()) as usize;
        objects.insert(name, take(&mut at, n));
    }
    let canonical = CanonicalTokenizer::from_objects(|n| objects.get(n).cloned())
        .expect("the fixture is a valid pie.tokenizer/1");
    Arc::new(Tokenizer::from_canonical(&canonical).expect("rebuilding the tokenizer"))
}

/// The renderer the ENGINE would bind, from the registry itself.
///
/// Not a reconstructed `ChatMLConfig`. The first version of this test copied
/// the qwen row's ten fields, which defeats the point: porting a model means
/// adding a row to `instruct::create`, and a test carrying its own copy would
/// happily pass against config nobody serves. `arch` is the driver's arch stem
/// — `architectures[0]` lowercased with the task suffix removed — recorded in
/// the fixture because only the generator knows it.
fn instruct(tokenizer: Arc<Tokenizer>, arch: &str, deploy: &str) -> Arc<dyn Instruct> {
    pie_model::instruct::create(arch, deploy, tokenizer)
}

fn render(inst: &dyn Instruct, body: &Value) -> Vec<u32> {
    let request: ChatCompletionRequest =
        serde_json::from_value(body.clone()).expect("fixture body deserializes");
    let ops = plan_render(&request).expect("fixture body plans");
    let mut ids = Vec::new();
    for op in &ops {
        ids.extend(match op {
            RenderOp::EquipAfterSystem { system, tools } => {
                inst.equip_after_system(system.as_deref(), tools)
            }
            RenderOp::User(m) => inst.user(m),
            RenderOp::Assistant(m, p) => inst.assistant_at(m, p.after_query, p.is_last),
            RenderOp::AssistantWithToolCalls { content, calls, pos } => inst
                .assistant_with_tool_calls_at(
                    content.as_deref(),
                    calls,
                    pos.after_query,
                    pos.is_last,
                ),
            RenderOp::AnswerBatch(r) => inst.answer_batch(r),
            RenderOp::Cue(thinking) => {
                if *thinking { inst.cue() } else { inst.cue_no_think() }
            }
        });
    }
    ids
}

#[test]
fn rendered_tokens_match_each_checkpoints_own_chat_template() {
    let dir = fixtures();
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("cases.json")).expect("fixtures are checked in"),
    )
    .expect("cases.json parses");

    // One tokenizer load per blob, not per case: loading is cheap but not free,
    // and two arms often share a blob.
    let mut toks: HashMap<String, Arc<Tokenizer>> = HashMap::new();
    for (digest, rel) in doc["tokenizers"].as_object().unwrap() {
        toks.insert(digest.clone(), load_pietok(&dir.join(rel.as_str().unwrap())));
    }
    let mut arms: HashMap<String, Arc<dyn Instruct>> = HashMap::new();
    for (arm, meta) in doc["arms"].as_object().unwrap() {
        let t = toks[meta["tokenizer"].as_str().unwrap()].clone();
        arms.insert(
            arm.clone(),
            instruct(t, meta["arch"].as_str().unwrap(), meta["deploy"].as_str().unwrap()),
        );
    }

    let cases = doc["cases"].as_array().expect("cases is an array");
    assert!(!cases.is_empty(), "the fixture corpus is empty — regenerate it");

    let mut failures = Vec::new();
    for case in cases {
        let arm = case["arm"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = render(arms[arm].as_ref(), &case["body"]);
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b);
            failures.push(format!(
                "  {arm}/{}/think={}  pie {} vs template {} tokens, first diff at {:?}",
                case["shape"].as_str().unwrap(),
                case["thinking"].as_bool().unwrap(),
                got.len(),
                want.len(),
                at
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} cells diverge from the checkpoint's own chat template.\n{}\n\
         This is pie asking the model a DIFFERENT QUESTION than the template \
         asks — it does not show up in throughput, and it does not raise an \
         error. It shows up as accuracy, attributed to the engine.\n\
         Run the Python harness for a token-level diff:\n  \
         python integrations/opencode/parity/check_render_matrix.py \
         --bin target/debug/render-tokens -v",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
