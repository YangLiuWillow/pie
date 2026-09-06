//! The D2 packing tables (`crates/model-ir/IMAGEGEN_CONTRACT.md` §1): which
//! attention group each lane of a fire joins, and — per row [`Selection`] a
//! plan reads — the packed order of the selected lanes' rows, the group and
//! lane CSRs over that order, and the reference tags.
//!
//! Pure host arithmetic over a [`Composition`]'s placed lanes and the
//! `(stream, group)` facts the runtime states per lane; nothing here knows a
//! device. Every shell builds its tables here so the packed order is one
//! definition.
//!
//! # The packed order, and where the packed rectangle lives
//!
//! A selection's lanes are sorted by `(group, stream code, fire lane)`, each
//! lane's rows contiguous. Packed row `j` of selection `S` lives at fire row
//! `origin_S + j` of a packed rectangle, where `origin_S` is the first fire
//! row of `S`'s lanes: `layout.pack_rows` is launched over the window of the
//! class(es) the selection names and writes the rows of that window, so the
//! packed rectangle stands exactly where the selected lanes' rows stand. For
//! `Selection::ALL` — the joint attention every text under test reads — the
//! origin is `0` and the rectangle is indexed from row zero, as the contract
//! says. A selection whose lanes' rows are not one contiguous run of the
//! fire (its classes seriated apart) is refused ([`Fault::ScatteredSelection`])
//! rather than packed into rows another class owns.
//!
//! Row indices in every table are fire-absolute: `permutation[origin + j]`
//! is the fire row packed row `j` was read from, the CSR bounds are rows of
//! the packed rectangle (`origin + …`), and `reference_tag[origin + j]` is
//! the fire lane of packed row `j` when its lane is on `Stream::Reference`.
//!
//! # Groups are fire-global
//!
//! [`group_of_lane`] numbers groups densely from `0` in order of first
//! appearance in fire lane order. The group CSR of every selection is indexed
//! by that fire-global id — a group none of the selection's lanes belong to
//! is an EMPTY segment, not a skipped one — so the query side and the key
//! side of a cross-attention (`q` from the image lanes, `k`/`v` from the
//! context lanes) pair segment `g` with segment `g` by construction, even
//! when one side has no lane in some group.

use model_ir::{Selection, Stream};

use super::Fault;
use super::compose::LaneRow;

/// What the runtime states about one submitted lane beyond its word: which
/// stream its rows are, and which attention group it joins (`None` is a
/// group of its own).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneFacts {
    /// `model_ir::Stream::code()`.
    pub stream: u8,
    /// The caller's group id, shared by the lanes of one request.
    pub group: Option<u32>,
}

/// One selection's tables, built by [`pack`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    /// The selection these tables are keyed by.
    pub select: Selection,
    /// The first fire row of the packed rectangle — the selected lanes'
    /// first row.
    pub origin: u32,
    /// How many rows the selection has.
    pub rows: u32,
    /// How many lanes the selection has.
    pub lanes: u32,
    /// `[groups + 1]`, indexed by fire-global group: group `g`'s packed rows
    /// are `[indptr[g], indptr[g + 1])` of the packed rectangle (absolute
    /// rows, `origin`-based); an absent group is an empty segment.
    pub group_indptr: Vec<i32>,
    /// `[selected lanes + 1]`: packed lane `i`'s rows, likewise absolute.
    pub lane_indptr: Vec<i32>,
    /// `[fire rows]`: per fire row of the packed rectangle, the fire lane of
    /// the row when its lane is a reference, else `-1`; `-1` outside the
    /// rectangle.
    pub reference_tag: Vec<i32>,
    /// `[fire rows]`: `perm[origin + j]` is the fire row packed row `j` came
    /// from; `-1` outside the rectangle.
    pub permutation: Vec<i32>,
    /// `[groups]`: per fire-global group, the row of the group (counted from
    /// its own first packed row) where its reference rows begin — the
    /// group's length when it has none. Reference is the highest stream code,
    /// so its lanes pack last within a group; this is the tail the CUDA
    /// ragged kernel's `ReferenceSelfOnly` reads.
    pub reference_start: Vec<i32>,
    /// `[groups]`: how many reference lanes each group has in this
    /// selection. A kernel with one tail per group serves at most one.
    pub references: Vec<u32>,
}

/// Which fire-global group each fire lane joins: `[fire lanes]`, dense from
/// `0` in order of first appearance, a stated group shared by every lane
/// that states it and an unstated one a group of its own. `facts[source]`
/// is the submitted lane's facts; `lanes` is the composition's fire order.
#[must_use]
pub fn group_of_lane(lanes: &[LaneRow], facts: &[LaneFacts]) -> Vec<i32> {
    let mut stated: Vec<(u32, i32)> = Vec::new();
    let mut next = 0i32;
    lanes
        .iter()
        .map(|row| {
            let group = facts.get(row.source as usize).and_then(|facts| facts.group);
            match group {
                Some(id) => match stated.iter().find(|(seen, _)| *seen == id) {
                    Some((_, dense)) => *dense,
                    None => {
                        let dense = next;
                        next += 1;
                        stated.push((id, dense));
                        dense
                    }
                },
                None => {
                    let dense = next;
                    next += 1;
                    dense
                }
            }
        })
        .collect()
}

/// How many groups a `group_of_lane` table names.
#[must_use]
pub fn groups_of(group_of_lane: &[i32]) -> u32 {
    group_of_lane
        .iter()
        .map(|&group| u32::try_from(group + 1).unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// Build one selection's tables. `lanes` is the composition's fire order,
/// `facts[source]` the submitted lane's facts, `group_of_lane` the table
/// [`group_of_lane`] built over the same lanes, and `fire_rows` the fire's
/// token rows (every `[Tokens]` table is that long).
///
/// # Errors
///
/// [`Fault::ScatteredSelection`] when the selected lanes' rows are not one
/// contiguous run of the fire.
pub fn pack(
    select: Selection,
    lanes: &[LaneRow],
    facts: &[LaneFacts],
    group_of_lane: &[i32],
    fire_rows: u32,
) -> Result<Packed, Fault> {
    let groups = groups_of(group_of_lane) as usize;
    // The selected lanes, as (group, stream, fire lane) keys.
    let mut chosen: Vec<(i32, u8, usize)> = lanes
        .iter()
        .enumerate()
        .filter(|(_, row)| select.holds(row.word))
        .map(|(at, row)| {
            let stream = facts
                .get(row.source as usize)
                .map_or(0, |facts| facts.stream);
            (group_of_lane.get(at).copied().unwrap_or(0), stream, at)
        })
        .collect();
    chosen.sort_unstable();

    // Contiguity: sorted by fire row, each lane begins where the last ended.
    let mut by_row: Vec<(u32, u32)> = chosen
        .iter()
        .map(|&(_, _, at)| (lanes[at].row_offset, lanes[at].rows))
        .collect();
    by_row.sort_unstable();
    let origin = by_row.first().map_or(0, |(offset, _)| *offset);
    let mut end = origin;
    for (offset, rows) in &by_row {
        if *offset != end {
            return Err(Fault::ScatteredSelection {
                mask: select.mask,
                value: select.value,
                at: *offset,
                expected: end,
            });
        }
        end += rows;
    }
    let rows = end - origin;

    let mut group_indptr = vec![0i32; groups + 1];
    let mut lane_indptr = Vec::with_capacity(chosen.len() + 1);
    let mut reference_tag = vec![-1i32; fire_rows as usize];
    let mut permutation = vec![-1i32; fire_rows as usize];
    let mut reference_start = vec![0i32; groups];
    let mut references = vec![0u32; groups];
    let mut seen_reference = vec![false; groups];

    let mut packed = origin;
    lane_indptr.push(packed as i32);
    let mut group_first = vec![origin; groups];
    let mut cursor_group: Option<i32> = None;
    for &(group, stream, at) in &chosen {
        let row = &lanes[at];
        let g = group as usize;
        if cursor_group != Some(group) {
            cursor_group = Some(group);
            group_first[g] = packed;
        }
        let reference = stream == Stream::Reference.code();
        if reference {
            references[g] += 1;
            if !seen_reference[g] {
                seen_reference[g] = true;
                reference_start[g] = (packed - group_first[g]) as i32;
            }
        }
        for j in 0..row.rows {
            let fire_row = row.row_offset + j;
            let here = (packed + j) as usize;
            permutation[here] = fire_row as i32;
            reference_tag[here] = if reference { at as i32 } else { -1 };
        }
        packed += row.rows;
        lane_indptr.push(packed as i32);
    }
    // The group CSR: each group's rows, in fire-global order; an absent
    // group repeats the bound before it.
    let mut bound = origin as i32;
    let mut run = 0usize;
    for g in 0..groups {
        group_indptr[g] = bound;
        let mut count = 0u32;
        while run < chosen.len() && chosen[run].0 as usize == g {
            count += lanes[chosen[run].2].rows;
            run += 1;
        }
        if !seen_reference[g] {
            reference_start[g] = count as i32;
        }
        bound += count as i32;
    }
    group_indptr[groups] = bound;
    debug_assert_eq!(bound, (origin + rows) as i32);

    Ok(Packed {
        select,
        origin,
        rows,
        lanes: chosen.len() as u32,
        group_indptr,
        lane_indptr,
        reference_tag,
        permutation,
        reference_start,
        references,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(source: u32, word: u64, row_offset: u32, rows: u32) -> LaneRow {
        LaneRow {
            source,
            word,
            class: 0,
            row_offset,
            rows,
            ..LaneRow::default()
        }
    }

    const TEXT: u64 = 1;
    const IMAGE: u64 = 2;
    const REFERENCE: u64 = 32;

    /// Two requests, text and image each: fire order is text lanes then image
    /// lanes (class-major); the joint packing interleaves by group.
    #[test]
    fn a_joint_selection_packs_by_group_then_stream() {
        // Submitted: A.text(3), A.image(4), B.text(2), B.image(5).
        // Fire order: A.text, B.text, A.image, B.image.
        let lanes = [
            row(0, TEXT, 0, 3),
            row(2, TEXT, 3, 2),
            row(1, IMAGE, 5, 4),
            row(3, IMAGE, 9, 5),
        ];
        let facts = [
            LaneFacts {
                stream: 0,
                group: Some(7),
            },
            LaneFacts {
                stream: 1,
                group: Some(7),
            },
            LaneFacts {
                stream: 0,
                group: Some(9),
            },
            LaneFacts {
                stream: 1,
                group: Some(9),
            },
        ];
        let groups = group_of_lane(&lanes, &facts);
        assert_eq!(groups, vec![0, 1, 0, 1]);
        let packed = pack(Selection::ALL, &lanes, &facts, &groups, 14).unwrap();
        assert_eq!(packed.origin, 0);
        assert_eq!(packed.rows, 14);
        assert_eq!(packed.group_indptr, vec![0, 7, 14]);
        assert_eq!(packed.lane_indptr, vec![0, 3, 7, 9, 14]);
        let expected: Vec<i32> = (0..3).chain(5..9).chain(3..5).chain(9..14).collect();
        assert_eq!(packed.permutation, expected);
        assert!(packed.reference_tag.iter().all(|&tag| tag == -1));
        assert_eq!(packed.reference_start, vec![7, 7]);
    }

    /// A selection of one class packs at that class's window and leaves the
    /// rest of the fire's tables `-1`; an absent group is an empty segment.
    #[test]
    fn a_class_selection_packs_at_its_own_window() {
        let lanes = [row(0, TEXT, 0, 3), row(2, TEXT, 3, 2), row(1, IMAGE, 5, 4)];
        let facts = [
            LaneFacts {
                stream: 0,
                group: Some(1),
            },
            LaneFacts {
                stream: 1,
                group: Some(1),
            },
            LaneFacts {
                stream: 0,
                group: None,
            },
        ];
        let groups = group_of_lane(&lanes, &facts);
        assert_eq!(groups, vec![0, 1, 0]);
        let image = Selection { mask: 2, value: 2 };
        let packed = pack(image, &lanes, &facts, &groups, 9).unwrap();
        assert_eq!(packed.origin, 5);
        assert_eq!(packed.rows, 4);
        assert_eq!(packed.group_indptr, vec![5, 9, 9]);
        assert_eq!(packed.lane_indptr, vec![5, 9]);
        assert_eq!(packed.permutation, vec![-1, -1, -1, -1, -1, 5, 6, 7, 8]);
    }

    /// Reference lanes pack last in their group and tag their rows.
    #[test]
    fn a_reference_lane_packs_last_and_is_tagged() {
        let lanes = [
            row(0, TEXT, 0, 2),
            row(1, IMAGE, 2, 3),
            row(2, REFERENCE, 5, 2),
        ];
        let facts = [
            LaneFacts {
                stream: 0,
                group: Some(0),
            },
            LaneFacts {
                stream: 1,
                group: Some(0),
            },
            LaneFacts {
                stream: 5,
                group: Some(0),
            },
        ];
        let groups = group_of_lane(&lanes, &facts);
        let packed = pack(Selection::ALL, &lanes, &facts, &groups, 7).unwrap();
        assert_eq!(packed.group_indptr, vec![0, 7]);
        assert_eq!(packed.reference_tag, vec![-1, -1, -1, -1, -1, 2, 2]);
        assert_eq!(packed.reference_start, vec![5]);
        assert_eq!(packed.references, vec![1]);
    }

    /// A selection whose lanes stand apart in the fire is refused by name.
    #[test]
    fn a_scattered_selection_is_refused() {
        let lanes = [row(0, TEXT, 0, 2), row(1, IMAGE, 2, 3), row(2, TEXT, 5, 2)];
        let facts = [LaneFacts::default(); 3];
        let groups = group_of_lane(&lanes, &facts);
        let text = Selection { mask: 1, value: 1 };
        let fault = pack(text, &lanes, &facts, &groups, 7).unwrap_err();
        assert!(matches!(fault, Fault::ScatteredSelection { .. }), "{fault}");
    }
}
