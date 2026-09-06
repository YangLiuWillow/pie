//! ATEM: Muse Glimmer's channelled turn format.
//!
//! `<|begin_of_text|>` once at the head, then
//! `<|start|>{role}<|message|>{text}<|eot|>` per turn. The assistant's
//! header names a recipient: ` to=self` opens a reasoning channel that ends
//! in `<|eom|>` (end of message, the turn goes on), ` to=user` the reply,
//! ` to={tool}` a call; a turn ends in `<|eot|>`. Tool results come back as
//! `<|start|>tool {name}<|message|><tool_output name="{name}">…</tool_output><|eot|>`.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::decode::{GenericChatDecoder, NoopToolDecoder, ThinkingDecoder};
use crate::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, special, specials};

/// What ends a turn: `<|eot|>` (the model's own end of turn) and the
/// vocabulary's end of text. `<|eom|>` is NOT one — it closes a message
/// inside a turn that continues.
pub const STOP_TOKENS: &[&str] = &["<|eot|>", "<|end_of_text|>"];

/// Every other marker the format spells, for a tokenizer contract to check.
pub const MARKERS: &[&str] = &["<|begin_of_text|>", "<|start|>", "<|message|>", "<|eom|>"];

pub struct Atem {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    eot: u32,
    eom: u32,
    stop_ids: Vec<u32>,
    /// ` to=self<|message|>` — what follows the cue when the model reasons.
    reasoning_open: Vec<u32>,
    /// `<|start|>assistant` — the model picks the recipient itself.
    generation_prefix: Vec<u32>,
}

impl Atem {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let stop_ids = specials(&tokenizer, STOP_TOKENS);

        let start = special(&tokenizer, "<|start|>");
        let message = special(&tokenizer, "<|message|>");
        let eot = special(&tokenizer, "<|eot|>");
        let eom = special(&tokenizer, "<|eom|>");

        let header = |role: &str| -> Vec<u32> {
            let mut tokens = vec![start];
            tokens.extend(tokenizer.encode(role));
            tokens.push(message);
            tokens
        };

        let mut reasoning_open = tokenizer.encode(" to=self");
        reasoning_open.push(message);

        let mut generation_prefix = vec![start];
        generation_prefix.extend(tokenizer.encode("assistant"));

        Self {
            bos: special(&tokenizer, "<|begin_of_text|>"),
            system_prefix: header("system"),
            user_prefix: header("user"),
            // A replayed reply names its recipient, as the reference template
            // writes it (`recipient` defaults to `user`).
            assistant_prefix: header("assistant to=user"),
            eot,
            eom,
            stop_ids,
            reasoning_open,
            generation_prefix,
            tokenizer,
        }
    }

    fn turn(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.push(self.eot);
        tokens
    }

    /// The turn that opens a conversation, and the only place `<|begin_of_text|>`
    /// is written.
    fn opening(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn(prefix, msg));
        tokens
    }
}

impl Instruct for Atem {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    /// A system message carries two lines the model was trained to see at
    /// its end, as the reference template writes them: the reasoning
    /// strength (`high` unless the message states one) and the recipient
    /// list. A caller steers the reasoning by writing its own
    /// `Reasoning strength: low.` line.
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut body = msg.to_string();
        if !msg.to_ascii_lowercase().contains("reasoning strength") {
            body.push_str("\n\nReasoning strength: high.");
        }
        body.push_str("\n\n# Valid recipients: \"self\", \"user\".");
        self.opening(&self.system_prefix, &body)
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.opening(&self.user_prefix, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.user_prefix, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.turn(&self.assistant_prefix, msg)
    }

    fn cue(&self) -> Vec<u32> {
        self.generation_prefix.clone()
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        let mut tokens = vec![self.generation_prefix[0]];
        tokens.extend(self.tokenizer.encode(&format!("tool {name}")));
        tokens.push(self.reasoning_open[self.reasoning_open.len() - 1]);
        tokens.extend(
            self.tokenizer
                .encode(&format!("<tool_output name=\"{name}\">\n{value}\n</tool_output>")),
        );
        tokens.push(self.eot);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.reasoning_open.clone(),
            self.eom,
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}
