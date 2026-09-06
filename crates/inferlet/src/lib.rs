//! Inferlet SDK for Pie: core types and traits for building inferlets that
//! run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

/// Add context to a failure on its way up. `pie:inferlet` declares
/// `type error = string`, so every fallible call already fails with a
/// message; this adds the standard way to say where it came from.
///
/// ```ignore
/// ws.reserve(pages).context("reserve KV")?;
/// tok_out
///     .take()
///     .to_host::<i32>()
///     .await
///     .with_context(|| format!("tok_out.take @{}", generated.len()))?;
/// ```
pub trait Context<T> {
    /// Prefix the error with `what`, as `"{what}: {error}"`.
    fn context(self, what: &str) -> Result<T>;

    /// Same, but the prefix is only built if there is an error to prefix.
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T>;
}

impl<T, E: std::fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: &str) -> Result<T> {
        self.map_err(|error| format!("{what}: {error}"))
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.map_err(|error| format!("{}: {error}", what()))
    }
}

/// The same, for an absent value: `Option` has no error to prefix, so `what`
/// becomes the whole message.
impl<T> Context<T> for Option<T> {
    fn context(self, what: &str) -> Result<T> {
        self.ok_or_else(|| what.to_string())
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.ok_or_else(|| what().to_string())
    }
}

// Re-export serde and serde_json so the macro-generated JSON bridge can use them
pub use serde;
pub use serde_json;

// Re-export the attribute macro
pub use inferlet_macros::main;

wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    generate_all,
});

pub use wit_bindgen;

// Re-export types that don't need async wrappers directly
pub use pie::inferlet::types;

/// The runtime working-set resources (KV page-slot array + recurrent state).
/// The generated WIT resources, unwrapped; [`eta::WorkingSet`] is the
/// pass-facing handle built over them.
pub mod working_set {
    pub use crate::pie::inferlet::working_set::*;
}

pub mod mask;
/// The author-facing ETA bridge: `ForwardPass`/`Pipeline`/
/// `WorkingSet`/`Channel` over the WIT forward resources, driving the `eta-dsl`
/// trace `Builder`. The single home of the ETA authoring surface.
pub mod eta;

pub mod chat;

/// The sampler prelude for the generative families (imagegen design D4):
/// a flow-matching schedule built from `model::schedule()`, the Euler step
/// and guidance rules as epilogue math over `velocity()`, a seeded device
/// noise draw, a positions-grid builder, and `encode_text`.
pub mod latent;

/// The runtime serves exactly one model; these are global functions over
/// that single bound model. There is no `Model`/`Tokenizer` handle to pass
/// around — call `model::encode`, `model::name`, etc. directly.
pub mod model {
    pub use crate::pie::inferlet::model::{
        AxisRole, BlockDrafter, CanvasShape, ForwardKind, LaneStream, LatentSpace, PortFact,
        PortKind, PositionConvention, ReadingFact, ReadoutKind, ScheduleFact, ScheduleKind,
        architecture, arena_block_size,
        canvas, channel_capacity, default_system_speculation, draft_block, frame_size,
        kv_page_size, latent, max_embed_length, max_latent_rows, mtp_depth, name,
        output_vocab_size, pass_kind, prefill_chunk_hint, readings, rs_buffer_page_size,
        rs_fold_granularity, rs_state_size, run_ahead_window, schedule, submit_deadline_us,
    };

    /// The declared reading named `name`, if the bound model has one.
    pub fn reading(name: &str) -> Option<ReadingFact> {
        readings().into_iter().find(|reading| reading.name == name)
    }
    // Tokenizer functions live in the `tokenizer` interface; re-exported here
    // so `model::encode`/`model::decode`/… read off `model` in inferlet source.
    pub use crate::pie::inferlet::tokenizer::{
        Token, decode, encode, special_tokens, split_regex, token_bytes, tokens_with_prefix,
        vocabs,
    };
}

pub mod runtime {
    pub use crate::pie::inferlet::system::*;
}

/// Suspend the current inferlet for `duration` without blocking the host
/// event loop. Backed by the runtime's async timer (host-provided under
/// component-model-async — wasi:clocks 0.2 pollables have no guest-side
/// future bridge). Use for streaming pacing, retry backoff, etc.
///
/// ```ignore
/// inferlet::sleep(std::time::Duration::from_millis(50)).await;
/// ```
pub async fn sleep(duration: std::time::Duration) {
    let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
    crate::wasi::clocks::monotonic_clock::wait_for(nanos).await;
}

/// Current `wasi:clocks/monotonic-clock` mark in nanoseconds.
pub fn monotonic_now_ns() -> u64 {
    crate::wasi::clocks::monotonic_clock::now()
}

/// User <-> process communication. `send`/`send_file` carry text and bytes the
/// guest already holds; `send_frames` and `send_pcm` carry a HANDLE, so the
/// encoded file is built host-side and streamed to the client without ever
/// being a value in this module's address space.
pub mod session {
    pub use crate::pie::inferlet::session::*;
}

/// Pixel and sample OUTPUT — the inverse of [`media`].
///
/// A [`Frames`](frames::Frames) handle holds decoded pixels host-side
/// (`count` frames of `width` x `height` RGB8) and a [`Pcm`](frames::Pcm)
/// handle holds interleaved f32 samples. The guest never sees either payload;
/// it asks for facts, and picks one of two exits:
///
/// ```ignore
/// let clip = frames::Frames::from_rgb8(&rgb, 512, 512, 16, 24.0)?;
/// // The happy path: encoded host-side, streamed out, never in linear memory.
/// session::send_frames(&clip, frames::ImageFormat::Mp4H264, "out.mp4")?;
/// // The other door, when the guest itself needs the bytes:
/// let png = still.encode(frames::ImageFormat::Png)?;
/// ```
///
/// `png`/`jpeg`/`webp` are stills and refuse a multi-frame handle by name;
/// `y4m` is the uncompressed clip; `mp4-h264` is NVENC plus the runtime's own
/// ISO-BMFF muxer, and refuses by name on a machine without an encoder.
pub mod frames {
    pub use crate::pie::inferlet::frames::{AudioFormat, Frames, ImageFormat, Pcm};
}

/// Grammar compilation + incremental matching (the WIT `grammar` interface).
/// [`Grammar`](grammar::Grammar) compiles a JSON Schema / regex / EBNF source
/// once for the bound model's vocabulary; [`Matcher`](grammar::Matcher) walks
/// one generation against it, exposing the packed allowed-token bitmask that
/// [`mask`] interprets.
pub mod grammar {
    pub use crate::pie::inferlet::grammar::*;
}

/// Multimodal input. The inferlet hands the host raw encoded bytes —
/// [`Image::from_bytes`](media::Image) (PNG/JPEG), [`Video::from_bytes`](media::Video)
/// (animated GIF), [`Audio::from_bytes`](media::Audio) (WAV) — and the host
/// decodes + preprocesses per the bound model.
///
/// A span enters the sequence as the token run its handle answers, and the
/// handle crosses again beside the tokens to carry the payload:
///
/// ```ignore
/// let img = media::Image::from_bytes(&bytes)?;
/// let mut toks = tokenizer.encode("Describe: ");
/// toks.extend(img.tokens());          // <|vision_start|> + pad×N + <|vision_end|>
/// toks.extend(tokenizer.encode(" briefly."));
/// pass.embed(&tokens_ch, &indptr_ch)?;
/// pass.media(&[media::Span::Image(&img)])?;
/// ```
///
/// The host matches placeholder runs to attached spans in order, refusing
/// every disagreement by name. Two different images can produce identical
/// token lists, so any cache keyed on tokens must fold in the span's digest.
pub mod media {
    pub use crate::pie::inferlet::media::{Audio, Image, Video};
    /// One attached span, by the resource you hold — `forward-pass.media`'s
    /// argument. Re-exported here rather than from `forward` because it is
    /// about media, and because a guest that names it is holding one of the
    /// resources above.
    pub use crate::pie::inferlet::forward::MediaSpan as Span;
}
