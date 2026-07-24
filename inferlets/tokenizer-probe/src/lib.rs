//! `tokenizer-probe` — Step 0 diagnostic for Pie ↔ HuggingFace tokenizer parity.
//!
//! Encodes each input string with Pie's tokenizer (`tokenizer_mini`) and returns
//! the token ids plus a decode round-trip, as JSON. A companion Python script
//! (`pie-rl/step0_tokenizer_parity/parity_check.py`) diffs these ids against
//! HuggingFace `AutoTokenizer` to catch silent tokenization drift before it can
//! corrupt an RL loop (misaligned `env_mask` / logprobs → wrong gradient).
//!
//! Input : `{ "texts": ["…", "…"] }`
//! Output: `{ "model": "<name>", "results": [{ "text", "ids", "roundtrip" }, …] }`
//!
//! The output JSON is also printed to stdout wrapped in sentinels so the harness
//! can extract it regardless of how `pie run` surfaces the return value.

use inferlet::model::Model;
use inferlet::{Result, runtime};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Strings to tokenize.
    texts: Vec<String>,
}

const BEGIN: &str = "<<<TOKPROBE_JSON>>>";
const END: &str = "<<<TOKPROBE_END>>>";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?.clone();
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();

    let mut results = Vec::with_capacity(input.texts.len());
    for text in &input.texts {
        let ids = tokenizer.encode(text);
        let roundtrip = tokenizer
            .decode(&ids)
            .unwrap_or_else(|e| format!("<DECODE_ERR: {e}>"));
        results.push(serde_json::json!({
            "text": text,
            "ids": ids,
            "roundtrip": roundtrip,
        }));
    }

    let out = serde_json::json!({
        "model": model_name,
        "results": results,
    });
    let json = serde_json::to_string(&out).map_err(|e| e.to_string())?;

    println!("{BEGIN}{json}{END}");

    Ok(json)
}
