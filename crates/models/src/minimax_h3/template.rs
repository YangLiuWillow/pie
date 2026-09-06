//! MiniMax H3's prompt rendering: there is none. `t2va` is the raw prompt
//! tokens with `add_special_tokens=False` — no chat template, no system
//! prompt, no generation cue (`.../minimax_h3/presentation.py:109-113`).
//! What the reference DOES build around a prompt is a *presentation*: the
//! keyframe and reference blocks
//! (`"<Picture i>: " <|vision_start|> <|image_pad|>×N <|vision_end|>`,
//! `"<Video k>: "` with per-block `"<t.t seconds>"` text, `"<Audio j>: "`
//! labels) that only a vision tower can consume — and this row is
//! text-only, so it renders nothing.
//!
//! The catalog wants a template column all the same, so this row borrows
//! `qwen_3`'s ChatML constructor as `wan_2` and `mini_dit` do; a guest
//! encoding a t2va prompt must ask the tokenizer for the raw ids, not for
//! a turn. When the vision tower lands, this becomes the presentation
//! builder.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
