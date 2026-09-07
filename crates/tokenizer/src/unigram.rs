//! **SENTENCEPIECE UNIGRAM**: the piece→score table and the Viterbi walk that
//! reads a string off it.
//!
//! BPE builds a segmentation by merging upward from bytes; Unigram picks the
//! segmentation of highest total log-probability out of every segmentation the
//! vocabulary admits. There is no merge table and no rank — only a score per
//! piece — so nothing in [`crate::bpe`] applies to the search, though its
//! table is still what maps a piece to its id and back (which is why the
//! decode, grammar and special-token paths need no Unigram of their own).
//!
//! The walk is the textbook one and is exact, not greedy: `best[i]` is the
//! score of the best segmentation of the first `i` bytes, and the answer is
//! read backwards off the piece that achieved each `best`. A position no piece
//! can reach falls to `unk`, one CHARACTER at a time — SentencePiece's own
//! behaviour, and the reason `unk_id` is required rather than optional.
//!
//! Byte fallback is deliberately absent: umT5's `tokenizer.json` states
//! `byte_fallback: false`, so a character outside the vocabulary is `unk` and
//! not a run of `<0xNN>` pieces. A model that wants the other behaviour is a
//! model this does not serve yet, and it will say so at load rather than
//! silently spelling a different word.

use rustc_hash::FxHashMap;

/// The scores, keyed the way the walk reads them.
///
/// Held beside [`crate::bpe::BpeTable`] rather than inside it: the table is a
/// symbol map that BPE and Unigram share, and the score is the one thing only
/// Unigram has.
#[derive(Debug)]
pub(crate) struct UnigramScores {
    /// `piece → (id, score)`. Owned rather than borrowed from the table
    /// because the walk probes by `&str` slice of the input, and the table is
    /// keyed by bytes.
    pieces: FxHashMap<Box<str>, (u32, f32)>,
    /// The longest piece in bytes: the walk never probes past it.
    max_len: usize,
    /// What an unreachable character becomes. Required, not optional — a
    /// Unigram vocabulary that states none cannot spell every string.
    unk_id: u32,
}

impl UnigramScores {
    /// Build from the `[[piece, score], …]` list `tokenizer.json` states, in
    /// its own order: a piece's index IS its id.
    pub(crate) fn new(pieces: &[(String, f32)], unk_id: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(!pieces.is_empty(), "Unigram vocabulary is empty");
        anyhow::ensure!(
            (unk_id as usize) < pieces.len(),
            "unk_id {unk_id} is outside a {}-piece vocabulary",
            pieces.len()
        );
        let mut max_len = 0usize;
        let mut map: FxHashMap<Box<str>, (u32, f32)> =
            FxHashMap::with_capacity_and_hasher(pieces.len(), Default::default());
        for (index, (piece, score)) in pieces.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            max_len = max_len.max(piece.len());
            let id = u32::try_from(index).map_err(|_| {
                anyhow::anyhow!("Unigram vocabulary is longer than a u32 counts")
            })?;
            // First writer wins, as SentencePiece's own loader does: a
            // duplicate piece later in the list is unreachable there too.
            map.entry(piece.as_str().into()).or_insert((id, *score));
        }
        Ok(Self {
            pieces: map,
            max_len,
            unk_id,
        })
    }

    /// The scores by id, for the canonical form: `[f32; vocab]` where index
    /// IS the token id. A piece the vocabulary never named (an id no entry
    /// reached) scores `0.0`, which the walk never consults because no piece
    /// spells it.
    pub(crate) fn scores_by_id(&self, vocab: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; vocab];
        for (id, score) in self.pieces.values() {
            if let Some(cell) = out.get_mut(*id as usize) {
                *cell = *score;
            }
        }
        out
    }

    /// What an unreachable character becomes.
    pub(crate) fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Rebuild from the canonical form: the pieces are the vocabulary's own
    /// bytes, in id order, paired with `scores[id]`.
    pub(crate) fn from_scores(pieces: &[(String, f32)], unk_id: u32) -> anyhow::Result<Self> {
        Self::new(pieces, unk_id)
    }

    /// The best segmentation of `text`, appended to `ids`.
    ///
    /// Exact: every position keeps the best score reaching it, so a long piece
    /// that scores worse than two short ones loses. `unk` is charged a score
    /// of `worst - 10` (SentencePiece's `kUnkPenalty` in spirit: strictly
    /// worse than any real piece, so the walk takes it only when nothing else
    /// reaches), and consumes exactly one character.
    pub(crate) fn encode(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let bytes = text.len();
        // `best[i]`: the score of the best path to byte `i`. `NEG_INFINITY`
        // is "unreachable", which only byte 0 is not.
        let mut best = vec![f32::NEG_INFINITY; bytes + 1];
        // `from[i] = (start, id)`: the piece that achieved `best[i]`.
        let mut from = vec![(usize::MAX, 0u32); bytes + 1];
        best[0] = 0.0;

        let unk_penalty = self
            .pieces
            .values()
            .map(|(_, score)| *score)
            .fold(f32::INFINITY, f32::min)
            - 10.0;

        for at in 0..bytes {
            if best[at] == f32::NEG_INFINITY || !text.is_char_boundary(at) {
                continue;
            }
            let rest = &text[at..];
            let mut reached = false;
            // Probe every prefix of `rest` up to the longest piece. Char
            // boundaries only — a piece is a string, never a partial code
            // point.
            let ceiling = self.max_len.min(rest.len());
            for end in 1..=ceiling {
                if !rest.is_char_boundary(end) {
                    continue;
                }
                let Some((id, score)) = self.pieces.get(&rest[..end]) else {
                    continue;
                };
                reached = true;
                let candidate = best[at] + score;
                if candidate > best[at + end] {
                    best[at + end] = candidate;
                    from[at + end] = (at, *id);
                }
            }
            // Nothing reaches from here, or nothing that gets further than one
            // character: `unk` for exactly one character, so the walk can
            // always continue.
            let step = rest.chars().next().map_or(0, char::len_utf8);
            if step > 0 {
                let candidate = best[at] + unk_penalty;
                if candidate > best[at + step] {
                    best[at + step] = candidate;
                    from[at + step] = (at, self.unk_id);
                }
            }
            let _ = reached;
        }

        // Read it backwards, then reverse: the path is a chain of `from`s.
        let mut walk = Vec::new();
        let mut at = bytes;
        while at > 0 {
            let (start, id) = from[at];
            if start == usize::MAX {
                // Unreachable, which the `unk` step above should have made
                // impossible. Rather than loop, spell the rest as `unk` per
                // character and stop.
                walk.clear();
                walk.extend(text.chars().map(|_| self.unk_id));
                break;
            }
            walk.push(id);
            at = start;
        }
        walk.reverse();
        ids.extend(walk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three pieces and a Viterbi that must not be greedy: `ab` scores worse
    /// than `a` + `b`, so the walk takes the pair even though the long piece
    /// matches first and matches further.
    #[test]
    fn the_walk_is_exact_and_not_greedy() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("ab".to_string(), -9.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("ab", &mut ids);
        assert_eq!(ids, vec![1, 2], "-1 + -1 beats -9");
    }

    /// And when the long piece IS better it wins, which is the other half of
    /// the same claim.
    #[test]
    fn the_walk_takes_the_long_piece_when_it_scores_better() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -5.0),
            ("b".to_string(), -5.0),
            ("ab".to_string(), -1.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("ab", &mut ids);
        assert_eq!(ids, vec![3]);
    }

    /// A character no piece spells is `unk`, one character at a time, and the
    /// walk carries on past it — a string is always spellable.
    #[test]
    fn an_unreachable_character_is_one_unk_and_the_walk_continues() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("azb", &mut ids);
        assert_eq!(ids, vec![1, 0, 2], "`z` is one unk between two pieces");
    }

    /// A multi-byte character is one `unk`, not one per byte: the step is a
    /// char, and a piece is never a partial code point.
    #[test]
    fn a_multibyte_character_is_one_unk() {
        let pieces = vec![("<unk>".to_string(), -100.0), ("a".to_string(), -1.0)];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("a\u{4e16}a", &mut ids);
        assert_eq!(ids, vec![1, 0, 1]);
    }

    /// An `unk_id` outside the vocabulary is refused at build, not at the
    /// first string that needs it.
    #[test]
    fn an_unk_outside_the_vocabulary_is_refused_at_build() {
        let pieces = vec![("a".to_string(), -1.0)];
        let why = UnigramScores::new(&pieces, 7).expect_err("no such piece");
        assert!(why.to_string().contains("unk_id 7"), "{why}");
    }
}
