//! HunyuanImage 3's prompt rendering: the `hunyuan-image-3` conversation
//! template, which is a plain `Role: message` transcript rather than a
//! marker-delimited chat (`tokenization_hunyuan_image_3.py:517-530`) —
//!
//! ```text
//! {system}\n\nUser: {prompt}\n\nAssistant:
//! ```
//!
//! `SeparatorStyle::ADD_COLON_SPACE_SINGLE` with `sep = "\n\n"`,
//! `sep_sp = "\n\n"` after the system block, `sep2 = "<|endoftext|>"`
//! closing an assistant turn, roles `("User", "Assistant")` and
//! `stop_token_ids = [127957]`. The generation cue is `"Assistant:"` with
//! the trailing space the reference's `f"{role}:"` + `" "` leaves — the
//! image sequence (`<boi> <img_size> <img_ratio> <timestep> <img>… <eoi>`)
//! is appended by the guest after it, since which ratio token to write is
//! either the AR resolution head's answer or the guest's own choice.
//!
//! `prefix()` is `<|startoftext|>`: the reference prepends bos 127958 to
//! every sequence it builds. It is written HERE and nowhere else, so a
//! caller cannot double it.
//!
//! No system prompt is rendered by default: the base T2I row's
//! `use_system_prompt` is `None`. The Instruct row's `en_unified` prompt is
//! ~1k tokens of policy text and belongs to the guest that wants it, which
//! passes it through [`Instruct::system`].

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::{
    ChatDecoder, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ToolDecoder, special, specials,
};

use super::tokenizer::{END_OF_TEXT, START_OF_TEXT, STOP_TOKENS};

/// The `hunyuan-image-3` transcript.
pub struct HunyuanImage3 {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    /// `<|endoftext|>`: `sep2`, written at the end of an assistant turn.
    eos: u32,
    /// `"\n\n"` — `sep` and `sep_sp` are the same string here.
    sep: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl HunyuanImage3 {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let sep = tokenizer.encode("\n\n");
        Self {
            bos: special(&tokenizer, START_OF_TEXT),
            eos: special(&tokenizer, END_OF_TEXT),
            sep,
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    /// `"{role}: {message}"` followed by `sep`.
    fn turn(&self, role: &str, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("{role}: {}", msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }
}

impl Instruct for HunyuanImage3 {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    /// The system block is the bare message plus `sep_sp`, with no role
    /// header (`system_template = "{system_message}"`).
    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.tokenizer.encode(msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn("User", msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("User", msg)
    }

    /// An assistant turn closes on `sep2`, not on `sep`.
    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("Assistant: {}", msg.trim()));
        tokens.push(self.eos);
        tokens
    }

    /// What a generation continues from: the assistant's header alone.
    fn cue(&self) -> Vec<u32> {
        self.tokenizer.encode("Assistant:")
    }

    fn seal(&self) -> Vec<u32> {
        vec![self.eos]
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(HunyuanImage3::new(tokenizer))
}
