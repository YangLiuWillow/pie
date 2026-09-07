//! Serialized tokenizer format (`pie.tokenizer/1`): vocab bytes/offsets, merge
//! quads, byte-fallback table, and a JSON descriptor. `from_canonical` refuses
//! any other version.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, DummyPrefix, Pipeline, Splitter, Tokenizer};

pub const VERSION: &str = "pie.tokenizer/1";

/// Sentinel meaning "no token for this byte"; `u32::MAX` cannot collide with
/// a real token id.
pub const NO_TOKEN: u32 = u32::MAX;

/// Object names, relative to the artifact's metadata namespace.
pub const VOCAB_BYTES: &str = "tokenizer/vocab_bytes";
pub const VOCAB_OFFSETS: &str = "tokenizer/vocab_offsets";
pub const MERGE_TABLE: &str = "tokenizer/merge_table";
pub const BYTE_FALLBACK: &str = "tokenizer/byte_fallback";
pub const DESCRIPTOR: &str = "tokenizer/descriptor";
/// **SENTENCEPIECE UNIGRAM ONLY**: one `f32` a token id, little endian, the
/// score its Viterbi walk maximises. A BPE tokenizer writes no such object,
/// and its ABSENCE is what says "this is not a Unigram" — which is why it is
/// [`OPTIONAL_OBJECTS`] and not a sixth entry of [`OBJECTS`]. Every artifact
/// written before Unigram was read lacks it, and must keep loading.
pub const UNIGRAM_SCORES: &str = "tokenizer/unigram_scores";

/// Every object a serialized tokenizer MUST carry, in the order they are
/// written. A reader that finds one missing has no tokenizer at all.
pub const OBJECTS: [&str; 5] = [
    BYTE_FALLBACK,
    DESCRIPTOR,
    MERGE_TABLE,
    VOCAB_BYTES,
    VOCAB_OFFSETS,
];

/// Objects only some tokenizers carry. Missing is not an error; missing is a
/// fact about which kind of tokenizer this is.
pub const OPTIONAL_OBJECTS: [&str; 1] = [UNIGRAM_SCORES];

/// A compiled tokenizer, serialized. Field order matches [`OBJECTS`]
/// (ascending by name), required by canonical `.zt` form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTokenizer {
    pub byte_fallback: Vec<u8>,
    pub descriptor: Vec<u8>,
    pub merge_table: Vec<u8>,
    /// Present for a Unigram and absent for everything else; see
    /// [`UNIGRAM_SCORES`]. Sorts between `merge_table` and `vocab_bytes`,
    /// which is where [`CanonicalTokenizer::objects`] puts it.
    pub unigram_scores: Option<Vec<u8>>,
    pub vocab_bytes: Vec<u8>,
    pub vocab_offsets: Vec<u8>,
}

impl CanonicalTokenizer {
    /// The objects to write, paired with their names, in ascending name order.
    /// A `Vec`, not an array, because the Unigram scores are there only for a
    /// Unigram — and canonical `.zt` form requires ascending names, which is
    /// why `unigram_scores` sits between `merge_table` and `vocab_bytes`
    /// rather than at the end.
    pub fn objects(&self) -> Vec<(&'static str, &[u8])> {
        let mut out: Vec<(&'static str, &[u8])> = vec![
            (BYTE_FALLBACK, &self.byte_fallback),
            (DESCRIPTOR, &self.descriptor),
            (MERGE_TABLE, &self.merge_table),
        ];
        if let Some(scores) = &self.unigram_scores {
            out.push((UNIGRAM_SCORES, scores));
        }
        out.push((VOCAB_BYTES, &self.vocab_bytes));
        out.push((VOCAB_OFFSETS, &self.vocab_offsets));
        out
    }

    /// Collects the objects back from whatever holds them. A missing REQUIRED
    /// object is a hard error rather than an empty-default guess; a missing
    /// optional one is a fact, and the fact it states is "not a Unigram" —
    /// which is what lets every artifact written before Unigram was read keep
    /// loading unchanged.
    pub fn from_objects(mut read: impl FnMut(&str) -> Option<Vec<u8>>) -> Result<Self> {
        let mut fetch = |name: &str| -> Result<Vec<u8>> {
            read(name).with_context(|| format!("the artifact has no '{name}' object"))
        };
        let byte_fallback = fetch(BYTE_FALLBACK)?;
        let descriptor = fetch(DESCRIPTOR)?;
        let merge_table = fetch(MERGE_TABLE)?;
        let vocab_bytes = fetch(VOCAB_BYTES)?;
        let vocab_offsets = fetch(VOCAB_OFFSETS)?;
        Ok(Self {
            byte_fallback,
            descriptor,
            merge_table,
            unigram_scores: read(UNIGRAM_SCORES),
            vocab_bytes,
            vocab_offsets,
        })
    }

    pub fn byte_size(&self) -> usize {
        self.objects().iter().map(|(_, bytes)| bytes.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct Descriptor {
    version: String,
    pipeline: PipelineDescriptor,
    added_tokens: Vec<AddedTokenDescriptor>,
}

/// Untagged: a bare string decodes as `Isolated` (the historical form,
/// still the common case); an object decodes as `Explicit`.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(untagged)]
enum SplitterDescriptor {
    /// `Split { behavior: Isolated }`.
    Isolated(String),
    /// `Split { behavior: Removed, invert: true }`: matches become pieces,
    /// text between them is dropped.
    Explicit { pattern: String, keep_gaps: bool },
}

impl SplitterDescriptor {
    fn pattern(&self) -> &str {
        match self {
            Self::Isolated(p) => p,
            Self::Explicit { pattern, .. } => pattern,
        }
    }

    fn keep_gaps(&self) -> bool {
        match self {
            Self::Isolated(_) => true,
            Self::Explicit { keep_gaps, .. } => *keep_gaps,
        }
    }
}

/// Sentencepiece dummy-prefix mode. Defaults to `None` (what every artifact
/// written before this field existed was).
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum DummyPrefixDescriptor {
    #[default]
    None,
    EverySegment,
    FirstSegment,
}

/// Tagged by `kind`; splitter order is semantic (applied as a sequence).
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PipelineDescriptor {
    ByteLevelRegex {
        nfc: bool,
        splitters: Vec<SplitterDescriptor>,
        prefer_whole_token: bool,
    },
    ByteFallbackReplace {
        normalizer_from: String,
        normalizer_to: String,
        unk_token_id: Option<u32>,
        #[serde(default)]
        dummy_prefix: DummyPrefixDescriptor,
        #[serde(default)]
        strip_decoder_marker: bool,
    },
    Unigram {
        replacement: String,
        prepend_always: bool,
        eos_id: Option<u32>,
        unk_id: u32,
    },
    RawChar,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct AddedTokenDescriptor {
    id: u32,
    content: String,
    special: bool,
    /// Absent from old artifacts; defaults to `false`, their actual behavior.
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// Serializes this tokenizer as `pie.tokenizer/1`. Fails rather than
    /// writing something lossy (a vocabulary hole, an undecodable encode map).
    pub fn to_canonical(&self) -> Result<CanonicalTokenizer> {
        ensure!(
            self.bpe.encode_map_is_derivable(),
            "this tokenizer's encode map cannot be recovered from its vocabulary, so \
             serializing it would change how it tokenizes; this happens when an added \
             token overwrote an id whose original bytes are still encodable"
        );

        let decode = self.bpe.decode_table();
        let mut vocab_bytes = Vec::new();
        let mut vocab_offsets = Vec::with_capacity((decode.len() + 1) * 4);
        vocab_offsets.extend_from_slice(&0u32.to_le_bytes());
        for (id, bytes) in decode.iter().enumerate() {
            let bytes = bytes.as_deref().with_context(|| {
                format!("token id {id} carries no bytes; a vocabulary with a hole in it has no canonical form")
            })?;
            vocab_bytes.extend_from_slice(bytes);
            let end = u32::try_from(vocab_bytes.len())
                .context("the concatenated vocabulary exceeds 4 GiB")?;
            vocab_offsets.extend_from_slice(&end.to_le_bytes());
        }

        let quads = self.bpe.merge_quads();
        let mut merge_table = Vec::with_capacity(quads.len() * 16);
        for (left, right, rank, merged) in quads {
            for value in [left, right, rank, merged] {
                merge_table.extend_from_slice(&value.to_le_bytes());
            }
        }

        let mut byte_fallback = Vec::with_capacity(256 * 4);
        for entry in self.bpe.byte_fallback_table() {
            byte_fallback.extend_from_slice(&entry.unwrap_or(NO_TOKEN).to_le_bytes());
        }

        let descriptor = Descriptor {
            version: VERSION.to_string(),
            pipeline: describe_pipeline(&self.pipeline),
            added_tokens: self.added_token_descriptors(),
        };
        let descriptor = serde_json::to_vec(&descriptor).context("encoding the descriptor")?;

        // A Unigram's scores, one f32 a token id, little endian. `None` for
        // everything else, which is what the reader keys on.
        let unigram_scores = match &self.pipeline {
            Pipeline::Unigram { scores, .. } => {
                let by_id = scores.scores_by_id(decode.len());
                let mut bytes = Vec::with_capacity(by_id.len() * 4);
                for score in by_id {
                    bytes.extend_from_slice(&score.to_le_bytes());
                }
                Some(bytes)
            }
            _ => None,
        };

        Ok(CanonicalTokenizer {
            byte_fallback,
            descriptor,
            merge_table,
            unigram_scores,
            vocab_bytes,
            vocab_offsets,
        })
    }

    /// Recovers added tokens in registration order.
    fn added_token_descriptors(&self) -> Vec<AddedTokenDescriptor> {
        self.added_tokens
            .iter()
            .map(|token| AddedTokenDescriptor {
                id: token.id,
                content: self
                    .bpe
                    .id_to_bytes(token.id)
                    .map(String::from_utf8_lossy)
                    .unwrap_or_default()
                    .into_owned(),
                special: self.special_token_ids.binary_search(&token.id).is_ok(),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect()
    }
}

fn describe_pipeline(pipeline: &Pipeline) -> PipelineDescriptor {
    match pipeline {
        // The scores themselves ride their OWN object (`UNIGRAM_SCORES`), one
        // f32 a token id: a descriptor is a description, and 256 300 floats
        // in one are not a description.
        Pipeline::Unigram {
            replacement,
            prepend_always,
            eos_id,
            scores,
        } => PipelineDescriptor::Unigram {
            replacement: replacement.clone(),
            prepend_always: *prepend_always,
            eos_id: *eos_id,
            unk_id: scores.unk_id(),
        },
        Pipeline::ByteLevelRegex {
            nfc,
            splitters,
            bpe_mode,
        } => PipelineDescriptor::ByteLevelRegex {
            nfc: *nfc,
            splitters: splitters
                .iter()
                .map(|s| {
                    if s.keep_gaps {
                        SplitterDescriptor::Isolated(s.regex.as_str().to_string())
                    } else {
                        SplitterDescriptor::Explicit {
                            pattern: s.regex.as_str().to_string(),
                            keep_gaps: false,
                        }
                    }
                })
                .collect(),
            prefer_whole_token: *bpe_mode == BpeMode::PreferWholeToken,
        },
        Pipeline::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix,
            strip_decoder_marker,
        } => PipelineDescriptor::ByteFallbackReplace {
            normalizer_from: normalizer_from.clone(),
            normalizer_to: normalizer_to.clone(),
            unk_token_id: *unk_token_id,
            dummy_prefix: match dummy_prefix {
                DummyPrefix::None => DummyPrefixDescriptor::None,
                DummyPrefix::EverySegment => DummyPrefixDescriptor::EverySegment,
                DummyPrefix::FirstSegment => DummyPrefixDescriptor::FirstSegment,
            },
            strip_decoder_marker: *strip_decoder_marker,
        },
        Pipeline::RawChar => PipelineDescriptor::RawChar,
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// Rebuilds a tokenizer from its `pie.tokenizer/1` objects; no format
    /// sniffing, no merge synthesis.
    pub fn from_canonical(objects: &CanonicalTokenizer) -> Result<Self> {
        let descriptor: Descriptor =
            serde_json::from_slice(&objects.descriptor).context("decoding the descriptor")?;
        ensure!(
            descriptor.version == VERSION,
            "this artifact's tokenizer is {:?}, and this build reads {VERSION}; \
             regenerate it with `pie model import --force`",
            descriptor.version
        );

        let offsets = read_u32s(&objects.vocab_offsets, "vocab_offsets")?;
        ensure!(
            !offsets.is_empty(),
            "vocab_offsets is empty; it holds one more entry than there are tokens"
        );
        ensure!(
            offsets[0] == 0,
            "vocab_offsets starts at {} rather than 0",
            offsets[0]
        );
        let mut vocab = Vec::with_capacity(offsets.len() - 1);
        for (id, window) in offsets.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            ensure!(
                start <= end && end <= objects.vocab_bytes.len(),
                "token id {id} spans [{start}, {end}) outside a {}-byte vocabulary",
                objects.vocab_bytes.len()
            );
            vocab.push(objects.vocab_bytes[start..end].to_vec());
        }
        ensure!(
            offsets[offsets.len() - 1] as usize == objects.vocab_bytes.len(),
            "vocab_offsets accounts for {} of {} vocabulary bytes",
            offsets[offsets.len() - 1],
            objects.vocab_bytes.len()
        );

        let flat = read_u32s(&objects.merge_table, "merge_table")?;
        ensure!(
            flat.len() % 4 == 0,
            "merge_table holds {} values, which is not a whole number of \
             (left, right, rank, merged) quads",
            flat.len()
        );
        let merges: Vec<(u32, u32, u32, u32)> = flat
            .chunks_exact(4)
            .map(|quad| (quad[0], quad[1], quad[2], quad[3]))
            .collect();

        let fallback = read_u32s(&objects.byte_fallback, "byte_fallback")?;
        ensure!(
            fallback.len() == 256,
            "byte_fallback holds {} entries rather than 256",
            fallback.len()
        );
        let mut byte_fallback_ids = [None; 256];
        for (byte, &id) in fallback.iter().enumerate() {
            byte_fallback_ids[byte] = (id != NO_TOKEN).then_some(id);
        }

        // The pipeline is rebuilt BEFORE the table takes the vocabulary: a
        // Unigram's walk needs the piece strings, and the table consumes
        // them.
        let pipeline = rebuild_pipeline(descriptor.pipeline, &vocab, &objects.unigram_scores)?;
        let bpe = BpeTable::from_canonical(vocab, &merges, byte_fallback_ids)?;
        let added_tokens = descriptor
            .added_tokens
            .into_iter()
            .map(|token| AddedToken {
                id: token.id,
                content: token.content,
                special: token.special,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect();
        Tokenizer::new(bpe, pipeline, added_tokens)
    }
}

/// `vocab` and `scores` are the two objects only a Unigram reads: its walk
/// needs the piece STRINGS beside their scores, and both are already decoded
/// here for the symbol table.
fn rebuild_pipeline(
    descriptor: PipelineDescriptor,
    vocab: &[Vec<u8>],
    scores: &Option<Vec<u8>>,
) -> Result<Pipeline> {
    Ok(match descriptor {
        PipelineDescriptor::Unigram {
            replacement,
            prepend_always,
            eos_id,
            unk_id,
        } => {
            let raw = scores.as_deref().context(
                "this artifact's tokenizer is a Unigram and carries no                  `tokenizer/unigram_scores` object; re-import it with                  `pie model import --force`",
            )?;
            ensure!(
                raw.len() == vocab.len() * 4,
                "the score plane holds {} bytes and the vocabulary is {} pieces",
                raw.len(),
                vocab.len()
            );
            let pieces: Vec<(String, f32)> = vocab
                .iter()
                .zip(raw.chunks_exact(4))
                .map(|(piece, word)| {
                    // A piece that is not UTF-8 cannot be a Unigram piece —
                    // this vocabulary is strings, not bytes.
                    let piece = std::str::from_utf8(piece)
                        .context("a Unigram piece is not UTF-8")?
                        .to_string();
                    let score = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                    Ok((piece, score))
                })
                .collect::<Result<Vec<_>>>()?;
            Pipeline::Unigram {
                scores: crate::unigram::UnigramScores::from_scores(&pieces, unk_id)?,
                replacement,
                prepend_always,
                eos_id,
            }
        }
        PipelineDescriptor::ByteLevelRegex {
            nfc,
            splitters,
            prefer_whole_token,
        } => {
            let splitters = splitters
                .iter()
                .map(|d| {
                    let pattern = d.pattern();
                    Ok(Splitter {
                        regex: fancy_regex::Regex::new(pattern)
                            .with_context(|| format!("recompiling splitter {pattern:?}"))?,
                        keep_gaps: d.keep_gaps(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Pipeline::ByteLevelRegex {
                nfc,
                splitters,
                bpe_mode: if prefer_whole_token {
                    BpeMode::PreferWholeToken
                } else {
                    BpeMode::Merge
                },
            }
        }
        PipelineDescriptor::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix,
            strip_decoder_marker,
        } => Pipeline::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix: match dummy_prefix {
                DummyPrefixDescriptor::None => DummyPrefix::None,
                DummyPrefixDescriptor::EverySegment => DummyPrefix::EverySegment,
                DummyPrefixDescriptor::FirstSegment => DummyPrefix::FirstSegment,
            },
            strip_decoder_marker,
        },
        PipelineDescriptor::RawChar => Pipeline::RawChar,
    })
}

fn read_u32s(bytes: &[u8], what: &str) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("{what} is {} bytes, not a whole number of u32", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

