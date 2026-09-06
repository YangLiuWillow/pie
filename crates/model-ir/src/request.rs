//! One request's shape facts, as the runtime states them per fire; a model's
//! `Classify::of` reads what it declared bits for.

/// Which stream of a request a lane carries. A lane is `(request, stream)`:
/// a request that joins several modalities in one attention submits one
/// lane per stream, each with its own fact word, and the lanes attend
/// together as one group (`GeomKind::GroupOfLane`). A text-only request has
/// one lane, on [`Text`](Stream::Text) — the default, so every existing
/// caller states nothing.
///
/// The stream is a fact like any other: a family packs it into its word by
/// [`code`](Stream::code) (`Predicate::stream(base, stream)` on the DSL side
/// spells the matching guard), and its projections are ordinary guarded
/// arms selected per lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Stream {
    /// Text tokens — an LLM's rows, a DiT's caption rows.
    #[default]
    Text,
    /// Image latent rows.
    Image,
    /// Video latent rows.
    Video,
    /// Audio latent rows.
    Audio,
    /// A context lane: an encoder's output (`RuntimeInput::Context`),
    /// constant across denoise steps, read as keys/values only.
    Context,
    /// Reference latent rows (an edit's source image): rows that attend
    /// themselves only under `RaggedMask::ReferenceSelfOnly`.
    Reference,
}

impl Stream {
    /// Every stream, in [`code`](Stream::code) order.
    pub const ALL: [Stream; 6] = [
        Stream::Text,
        Stream::Image,
        Stream::Video,
        Stream::Audio,
        Stream::Context,
        Stream::Reference,
    ];

    /// The stream's small integer, `0..6` in [`ALL`](Stream::ALL)'s order —
    /// what a family adds to its base bit to spell the stream as a fact,
    /// and what `GeomKind::ReferenceTag`-style tables carry.
    #[must_use]
    pub fn code(self) -> u8 {
        self as u8
    }

    /// The stream with this [`code`](Stream::code), if any.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Stream> {
        Stream::ALL.get(usize::from(code)).copied()
    }

    /// The one-hot fact word of this stream at `base`: bit `base + code`.
    /// The spelling a family's `Classify::word` uses when its stream bits
    /// start at `base`.
    #[must_use]
    pub fn word(self, base: u8) -> u64 {
        1u64 << (base + self.code())
    }

    /// The name a refusal or a ledger line spells this stream with.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Stream::Text => "text",
            Stream::Image => "image",
            Stream::Video => "video",
            Stream::Audio => "audio",
            Stream::Context => "context",
            Stream::Reference => "reference",
        }
    }
}

/// One request's shape facts, as the runtime states them per fire: row
/// count, whether it carries a custom mask, an adapter route, a draft head,
/// score capture, and — for a lane of a multi-stream request — which
/// stream it is and which declared reading its pass runs. A model's
/// `Classify::of` reads what it declared bits for and ignores the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
    adapter: bool,
    drafts: bool,
    captures_scores: bool,
    media: bool,
    block_draft: bool,
    denoise: bool,
    stream: Stream,
    reading: u8,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
            adapter: false,
            drafts: false,
            captures_scores: false,
            media: false,
            block_draft: false,
            denoise: false,
            stream: Stream::Text,
            reading: 0,
        }
    }

    /// The same request, read as one lane of `stream`. A request that
    /// joins several streams submits one lane per stream, all sharing one
    /// attention group; each lane is classified on its own.
    #[must_use]
    pub fn on_stream(mut self, stream: Stream) -> Request {
        self.stream = stream;
        self
    }

    /// The same request, stamped with one of the family's declared readings
    /// — which arm of the plan its lanes run (`"text"`, `"denoise"`,
    /// `"denoise.low"`, `"vae.decode"`, ...), as the index the family gave
    /// that reading. `0` is the family's default arm, so a text with one
    /// reading never reads this.
    #[must_use]
    pub fn in_reading(mut self, reading: u8) -> Request {
        self.reading = reading;
        self
    }

    /// The same request, read as a block-diffusion denoiser's canvas: its
    /// rows attend bidirectionally and its input is the denoiser's (the
    /// self-conditioned, post-normed embedding), not the encoder's. A fact
    /// only a diffusion text declares a bit for.
    #[must_use]
    pub fn denoising(mut self, denoise: bool) -> Request {
        self.denoise = denoise;
        self
    }

    /// The same request, routing to an adapter bank.
    #[must_use]
    pub fn adapted(mut self, adapter: bool) -> Request {
        self.adapter = adapter;
        self
    }

    /// The same request, with the model's draft head run over its rows.
    #[must_use]
    pub fn drafting(mut self, drafts: bool) -> Request {
        self.drafts = drafts;
        self
    }

    /// The same request, carrying a BLOCK DRAFTER's proposal rows rather
    /// than rows of the sequence itself.
    ///
    /// A block drafter (`qwen_3`'s [`Recipe::DFlash`]) proposes many tokens
    /// in one pass over a block whose first row is the correction the target
    /// just made and whose rest is the mask token. Those rows are not the
    /// model's own — the trunk must not run over them — so this is what a
    /// plan guards its trunk against, and it is the guest's to set: only the
    /// inferlet knows the accepted prefix that anchors the block, so only it
    /// can say which fire is a draft.
    ///
    /// Distinct from [`drafting`](Request::drafting), which asks a plan to
    /// run its draft head over rows the trunk ALSO processes.
    #[must_use]
    pub fn drafting_a_block(mut self, block_draft: bool) -> Request {
        self.block_draft = block_draft;
        self
    }

    /// The same request, with its attention's per-key mass kept.
    #[must_use]
    pub fn capturing_scores(mut self, captures_scores: bool) -> Request {
        self.captures_scores = captures_scores;
        self
    }

    /// The same request, carrying images. Without this bit a text-only fire
    /// of a vision load resolves `RuntimeInput::PatchRoutes` and panics.
    #[must_use]
    pub fn with_media(mut self, media: bool) -> Request {
        self.media = media;
        self
    }

    #[must_use]
    pub fn query_len(&self) -> u32 {
        self.query_len
    }

    #[must_use]
    pub fn has_custom_mask(&self) -> bool {
        self.custom_mask
    }

    #[must_use]
    pub fn denoise(&self) -> bool {
        self.denoise
    }

    #[must_use]
    pub fn has_adapter(&self) -> bool {
        self.adapter
    }

    #[must_use]
    pub fn drafts(&self) -> bool {
        self.drafts
    }

    #[must_use]
    pub fn drafts_a_block(&self) -> bool {
        self.block_draft
    }

    #[must_use]
    pub fn captures_scores(&self) -> bool {
        self.captures_scores
    }

    #[must_use]
    pub fn has_media(&self) -> bool {
        self.media
    }

    /// Which stream this lane carries; [`Stream::Text`] unless stated.
    #[must_use]
    pub fn stream(&self) -> Stream {
        self.stream
    }

    /// Which declared reading this lane's pass runs; `0` unless stated.
    #[must_use]
    pub fn reading(&self) -> u8 {
        self.reading
    }
}

/// How a model packs a request into the fact word a lane carries.
pub type ClassifyFn = fn(&Request) -> u64;

#[cfg(test)]
mod tests {
    use super::{Request, Stream};

    /// A request states nothing and reads as a text lane in reading 0, so
    /// every caller from before streams existed classifies as it did.
    #[test]
    fn a_request_defaults_to_the_text_stream_and_the_default_reading() {
        let r = Request::new(4, false);
        assert_eq!(r.stream(), Stream::Text);
        assert_eq!(r.reading(), 0);
        let r = r.on_stream(Stream::Audio).in_reading(2);
        assert_eq!(r.stream(), Stream::Audio);
        assert_eq!(r.reading(), 2);
        for stream in Stream::ALL {
            assert_eq!(Stream::from_code(stream.code()), Some(stream));
            assert_eq!(stream.word(8), 1 << (8 + stream.code()));
        }
        assert_eq!(Stream::from_code(6), None);
    }
}
