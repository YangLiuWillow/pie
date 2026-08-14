//! adopt_kv — graft another context's KV into this one by copying rows.
//!
//! The faithful join of a parallel-decoded branch (NPR-style fork/merge)
//! needs the sibling's KV to appear in the merged context at overlapped
//! position ids with hole masks exposing only the shared prefix plus the
//! sibling's own tokens. Recomputing that KV via an explicit-mask refill is
//! numerically validated but costs a full prefill of the sibling's tokens.
//! Because K/V vectors depend only on token content, position ids, and the
//! visible attention set — all identical between the sibling's own decode
//! and the refill — the refill can be replaced by a device-side copy of the
//! sibling's KV rows.
//!
//! `adopt_kv(dst, src, src_token_start, num_tokens)` copies the KV rows for
//! src token-slots `[src_token_start, src_token_start + num_tokens)` into
//! dst's next free working slots, and appends the matching token metadata:
//!
//! - tokens and position ids come verbatim from src's own lineage — the
//!   guest cannot claim content that differs from what the pages hold, so
//!   the content-addressed page hashes computed at commit stay truthful and
//!   CAS dedup stays sound;
//! - per-token attention masks are synthesized host-side as the canonical
//!   hole rows `[0, src_token_start) ∪ [dst_start, dst_start + i]` — exactly
//!   the rows an explicit-mask refill would carry.
//!
//! Soundness constraints enforced here:
//!
//! 1. **Prefix agreement**: src and dst must have identical
//!    (token, position, mask) histories over `[0, src_token_start)`. Fork
//!    siblings satisfy this by construction. Together with (2) this makes
//!    the attended token set of every adopted token identical between src
//!    (causal, by slot) and dst (canonical hole mask), which is what makes
//!    the copied KV bits correct in their new home.
//! 2. **Causal source range**: every adopted src token must have attended
//!    exactly all earlier src slots — a default (empty) mask or an explicit
//!    slot-causal `all_true(slot+1)` row (what position-decoupled decoding
//!    emits) — and no adapter. Nested hole-mask regions inside the range
//!    would need general mask translation; not supported.
//! 3. Both contexts on the same driver and on GPU; no rs_cache models
//!    (recurrent state cannot be stitched by copying KV rows).
//!
//! The device copy is **awaited**: both contexts are pinned, the copy is
//! submitted to the driver, and the token metadata is appended only after
//! the driver confirms success (`Message::AdoptKvComplete`). A refused or
//! failed copy therefore surfaces as a hard error to the guest instead of
//! silently leaving stale bytes that a forward would read as plausible KV.
//! Pinning both contexts for the duration keeps eviction and any other
//! mutation off the pages the copy reads and writes.
//!
//! The resulting dst layout is byte-identical in metadata to what the
//! refill produces, so commit hashing, suspend/restore replay (which
//! recomputes via ordinary fills — every mask references only earlier
//! slots), and eviction all work unchanged. Adopted tokens are not billed
//! against the token budget: no forward compute was spent on them.

use anyhow::Result;
use tokio::sync::oneshot;

use pie_driver_abi::Brle;

use super::{Context, ContextId, ContextManager, Message, PINNED_COUNTS, Record, SERVICES, State};
use crate::driver::{self, KvRowCopySegment};

/// One entry of a context's flattened token history, in slot order.
struct HistEntry<'a> {
    token: u32,
    position: u32,
    mask: &'a Brle,
    has_adapter: bool,
}

/// Flatten a context's token history in slot order: committed lineage
/// records first (their token count equals `committed_len * page_size`),
/// then working-page tokens.
fn history(ctx: &Context) -> Vec<HistEntry<'_>> {
    let mut out = Vec::new();
    for record in &ctx.lineage {
        let Record::Fill {
            tokens,
            positions,
            mask,
            adapter,
            ..
        } = record;
        for i in 0..tokens.len() {
            out.push(HistEntry {
                token: tokens[i],
                position: positions[i],
                mask: &mask[i],
                has_adapter: adapter.is_some(),
            });
        }
    }
    for info in &ctx.working_page_tokens {
        out.push(HistEntry {
            token: info.token,
            position: info.position,
            mask: &info.mask,
            has_adapter: info.adapter.is_some(),
        });
    }
    out
}

fn is_default_mask(mask: &Brle) -> bool {
    mask.buffer.is_empty() && mask.total_size == 0
}

/// True when the mask attends exactly `[0, slot]` — either the default
/// (empty) mask or an explicit slot-causal row `all_true(slot + 1)`. The
/// latter is what position-decoupled decoding emits (slot-causal rows with
/// explicit positions); both attend the same token set, which is all the
/// adopt soundness argument needs.
fn is_causal_equivalent(mask: &Brle, slot: usize) -> bool {
    if is_default_mask(mask) {
        return true;
    }
    mask.buffer.len() == 2 && mask.buffer[0] == 0 && mask.buffer[1] as usize == slot + 1
}

/// Canonical hole-mask row for adopted token `i`: sees the shared prefix
/// `[0, prefix_slots)` plus its own range `[dst_start, dst_start + i]`.
fn canonical_mask(prefix_slots: u32, dst_start: u32, i: u32) -> Brle {
    if prefix_slots == dst_start {
        Brle::from_vec(vec![0, dst_start + i + 1])
    } else {
        Brle::from_vec(vec![0, prefix_slots, dst_start - prefix_slots, i + 1])
    }
}

/// Split the copy of `num_tokens` rows from src slot `src_start` to dst slot
/// `dst_start` into segments that each stay within one src page and one dst
/// page. `src_pages` / `dst_pages` are the contexts' concatenated
/// (committed ++ working) physical page lists.
fn build_segments(
    page_size: usize,
    src_pages: &[u32],
    dst_pages: &[u32],
    src_start: usize,
    dst_start: usize,
    num_tokens: usize,
) -> Result<Vec<KvRowCopySegment>> {
    let mut segments = Vec::new();
    let mut done = 0usize;
    while done < num_tokens {
        let s = src_start + done;
        let d = dst_start + done;
        let src_row = s % page_size;
        let dst_row = d % page_size;
        let n = (num_tokens - done)
            .min(page_size - src_row)
            .min(page_size - dst_row);
        let src_page = *src_pages
            .get(s / page_size)
            .ok_or_else(|| anyhow::anyhow!("adopt_kv: src slot {s} has no resident page"))?;
        let dst_page = *dst_pages
            .get(d / page_size)
            .ok_or_else(|| anyhow::anyhow!("adopt_kv: dst slot {d} has no reserved page"))?;
        segments.push(KvRowCopySegment {
            src_page,
            src_row: src_row as u32,
            dst_page,
            dst_row: dst_row as u32,
            row_count: n as u32,
        });
        done += n;
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_mask_matches_refill_rows() {
        // Distinct prefix and dst_start: [0, prefix) true, gap false, own true.
        let m = canonical_mask(7, 12, 2);
        assert_eq!(m.buffer, vec![0, 7, 5, 3]);
        assert_eq!(m.total_size, 15);
        // Contiguous case (dst_start == prefix): plain causal row.
        let m = canonical_mask(7, 7, 0);
        assert_eq!(m.buffer, vec![0, 8]);
        assert_eq!(m.total_size, 8);
    }

    #[test]
    fn causal_equivalence_accepts_default_and_slot_causal_only() {
        assert!(is_causal_equivalent(&Brle::new(0), 41));
        assert!(is_causal_equivalent(&Brle::all_true(42), 41));
        // Wrong length: attends one slot short.
        assert!(!is_causal_equivalent(&Brle::all_true(41), 41));
        // Hole mask: not causal.
        assert!(!is_causal_equivalent(
            &Brle::from_vec(vec![0, 10, 5, 27]),
            41
        ));
    }

    #[test]
    fn segments_split_on_both_page_grids() {
        // page_size 4; copy 6 tokens from src slot 2 to dst slot 5.
        // Src rows: page0[2..4), page1[0..4), page2[0..2)
        // Dst rows: page1[1..4), page2[0..4)  (dst pages 100..)
        let src_pages = [10, 11, 12];
        let dst_pages = [100, 101, 102];
        let segs = build_segments(4, &src_pages, &dst_pages, 2, 5, 6).unwrap();
        let flat: Vec<_> = segs
            .iter()
            .map(|s| (s.src_page, s.src_row, s.dst_page, s.dst_row, s.row_count))
            .collect();
        assert_eq!(
            flat,
            vec![
                (10, 2, 101, 1, 2), // src pg0 rows 2-3 → dst pg1 rows 1-2
                (11, 0, 101, 3, 1), // src pg1 row 0   → dst pg1 row 3
                (11, 1, 102, 0, 3), // src pg1 rows 1-3 → dst pg2 rows 0-2
            ]
        );
        // Total rows conserved.
        assert_eq!(segs.iter().map(|s| s.row_count).sum::<u32>(), 6);
    }

    #[test]
    fn segments_error_when_pages_missing() {
        assert!(build_segments(4, &[10], &[100, 101], 2, 5, 6).is_err());
        assert!(build_segments(4, &[10, 11, 12], &[100], 2, 5, 6).is_err());
    }
}

impl ContextManager {
    /// Contention-aware adopt: reserves the dst working pages the adopted
    /// tokens need, then verifies, copies KV rows (awaited, both contexts
    /// pinned), and appends metadata on driver-confirmed success. Responds
    /// with the dst slot index where the adopted tokens begin.
    pub(crate) fn adopt_kv(
        &mut self,
        dst_id: ContextId,
        src_id: ContextId,
        src_token_start: usize,
        num_tokens: usize,
        response: oneshot::Sender<Result<u32>>,
    ) {
        let page_size = self.page_size;

        // Cheap pre-checks + page-need estimate (re-validated in the
        // closure — when_allocated may defer it arbitrarily long).
        let (driver_idx, needed) = {
            if dst_id == src_id {
                let _ = response.send(Err(anyhow::anyhow!(
                    "adopt_kv: src and dst are the same context"
                )));
                return;
            }
            let Some(dst) = self.contexts.get(&dst_id) else {
                let _ = response.send(Err(anyhow::anyhow!("adopt_kv: dst context not found")));
                return;
            };
            let driver_idx = dst.driver.unwrap_or(0) as usize;
            let w = dst.working_page_tokens.len();
            let target_pages = (w + num_tokens).div_ceil(page_size);
            (driver_idx, target_pages.saturating_sub(dst.working_pages.len()))
        };

        self.when_allocated(dst_id, driver_idx, needed, move |mgr, pages| {
            mgr.adopt_kv_start(
                dst_id,
                src_id,
                src_token_start,
                num_tokens,
                driver_idx,
                pages,
                response,
            );
        });
    }

    /// Phase A (actor, synchronous): validate everything, attach the
    /// reserved pages, pin both contexts, and hand the copy to a background
    /// task. The task awaits the driver's status and reports back via
    /// `Message::AdoptKvComplete`.
    #[allow(clippy::too_many_arguments)]
    fn adopt_kv_start(
        &mut self,
        dst_id: ContextId,
        src_id: ContextId,
        src_token_start: usize,
        num_tokens: usize,
        driver_idx: usize,
        pages: Vec<super::pagestore::PhysicalPageId>,
        response: oneshot::Sender<Result<u32>>,
    ) {
        match self.adopt_kv_validate(
            dst_id,
            src_id,
            src_token_start,
            num_tokens,
            driver_idx,
            pages,
        ) {
            Ok(Some((segments, src_range, dst_start_slot))) => {
                // Pin both contexts: nothing may evict or mutate the pages
                // the copy reads/writes until the driver confirms.
                for id in [dst_id, src_id] {
                    if let Some(ctx) = self.contexts.get_mut(&id) {
                        ctx.state = State::Pinned;
                    }
                }
                if driver_idx < PINNED_COUNTS.len() {
                    PINNED_COUNTS[driver_idx]
                        .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                }

                let model_idx = self.model_idx;
                tokio::spawn(async move {
                    let copy_result = driver::copy_kv_rows_d2d(
                        driver_idx as crate::driver::DriverId,
                        &segments,
                    )
                    .await
                    .map_err(|e| e.to_string());
                    let _ = SERVICES.send(
                        model_idx,
                        Message::AdoptKvComplete {
                            dst_id,
                            src_id,
                            copy_result,
                            src_range,
                            prefix_slots: src_token_start as u32,
                            dst_start: dst_start_slot as u32,
                            response,
                        },
                    );
                });
            }
            // Zero-token adopt: nothing to copy, respond immediately.
            Ok(None) => {
                let dst_start = self
                    .contexts
                    .get(&dst_id)
                    .map(|d| (d.committed_len() * self.page_size + d.working_page_tokens.len()) as u32)
                    .unwrap_or(0);
                let _ = response.send(Ok(dst_start));
            }
            Err(e) => {
                let _ = response.send(Err(e));
            }
        }
    }

    /// Validation + page attachment. Returns `Ok(None)` for a zero-token
    /// adopt, otherwise the copy segments, the adopted (token, position)
    /// pairs, and the dst slot where they will land.
    #[allow(clippy::type_complexity)]
    fn adopt_kv_validate(
        &mut self,
        dst_id: ContextId,
        src_id: ContextId,
        src_token_start: usize,
        num_tokens: usize,
        driver_idx: usize,
        pages: Vec<super::pagestore::PhysicalPageId>,
    ) -> Result<Option<(Vec<KvRowCopySegment>, Vec<(u32, u32)>, usize)>> {
        let page_size = self.page_size;
        // On any failure the freshly-allocated pages must go back to the pool.
        let fail = |mgr: &mut Self, pages: Vec<u32>, err: anyhow::Error| -> Result<_> {
            if !pages.is_empty() {
                mgr.gpu_stores[driver_idx].free(&pages);
                mgr.drain_queues();
            }
            Err(err)
        };

        if self.driver_uses_rs_cache(driver_idx) {
            return fail(
                self,
                pages,
                anyhow::anyhow!("adopt_kv: not supported on rs_cache (linear-attention) models"),
            );
        }

        // Validate both contexts and gather everything needed from src.
        let (src_range, src_page_list, dst_start_slot) = {
            let Some(dst) = self.contexts.get(&dst_id) else {
                return fail(self, pages, anyhow::anyhow!("adopt_kv: dst context not found"));
            };
            let Some(src) = self.contexts.get(&src_id) else {
                return fail(self, pages, anyhow::anyhow!("adopt_kv: src context not found"));
            };
            if dst.is_off_gpu() || src.is_off_gpu() {
                return fail(self, pages, anyhow::anyhow!("adopt_kv: context is off GPU"));
            }
            if src.is_pinned() {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!("adopt_kv: src has a forward pass in flight"),
                );
            }
            let src_driver = src.driver.unwrap_or(0) as usize;
            if src_driver != driver_idx {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!(
                        "adopt_kv: src is on driver {src_driver}, dst on {driver_idx}"
                    ),
                );
            }

            let src_hist = history(src);
            let dst_hist = history(dst);
            if src_token_start + num_tokens > src_hist.len() {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!(
                        "adopt_kv: src range [{src_token_start}, {}) exceeds src history of {} tokens",
                        src_token_start + num_tokens,
                        src_hist.len()
                    ),
                );
            }
            if src_token_start > dst_hist.len() {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!(
                        "adopt_kv: shared prefix of {src_token_start} exceeds dst history of {} tokens",
                        dst_hist.len()
                    ),
                );
            }

            // (1) Prefix agreement.
            let prefix_ok = src_hist[..src_token_start]
                .iter()
                .zip(&dst_hist[..src_token_start])
                .all(|(a, b)| a.token == b.token && a.position == b.position && a.mask == b.mask);
            if !prefix_ok {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!(
                        "adopt_kv: src and dst histories differ within the claimed shared prefix"
                    ),
                );
            }

            // (2) Causal, adapter-free source range.
            for (i, e) in src_hist[src_token_start..src_token_start + num_tokens]
                .iter()
                .enumerate()
            {
                if !is_causal_equivalent(e.mask, src_token_start + i) {
                    return fail(
                        self,
                        pages,
                        anyhow::anyhow!(
                            "adopt_kv: src token {} carries a non-causal mask (range must attend all earlier slots)",
                            src_token_start + i
                        ),
                    );
                }
                if e.has_adapter {
                    return fail(
                        self,
                        pages,
                        anyhow::anyhow!(
                            "adopt_kv: src token {} was computed under an adapter",
                            src_token_start + i
                        ),
                    );
                }
            }

            let src_range: Vec<(u32, u32)> = src_hist
                [src_token_start..src_token_start + num_tokens]
                .iter()
                .map(|e| (e.token, e.position))
                .collect();

            // Src physical pages: committed (via CAS trie) ++ working.
            let mut src_page_list = if src.committed_hashes.is_empty() {
                Vec::new()
            } else {
                self.gpu_stores[driver_idx].physical_ids(&src.committed_hashes)
            };
            if src_page_list.len() != src.committed_len() {
                return fail(
                    self,
                    pages,
                    anyhow::anyhow!("adopt_kv: src committed pages not fully resident"),
                );
            }
            src_page_list.extend_from_slice(&src.working_pages);

            let dst_start_slot = dst.committed_len() * page_size + dst.working_page_tokens.len();
            (src_range, src_page_list, dst_start_slot)
        };

        if num_tokens == 0 {
            if !pages.is_empty() {
                self.gpu_stores[driver_idx].free(&pages);
                self.drain_queues();
            }
            return Ok(None);
        }

        // Attach the reserved pages and build the dst page list. From here
        // on the pages belong to dst (freed with it on any later failure).
        let dst_page_list = {
            let dst = self
                .contexts
                .get_mut(&dst_id)
                .ok_or_else(|| anyhow::anyhow!("adopt_kv: dst context lost"))?;
            dst.working_pages.extend_from_slice(&pages);
            let capacity = (dst.committed_len() + dst.working_pages.len()) * page_size;
            if dst_start_slot + num_tokens > capacity {
                anyhow::bail!(
                    "adopt_kv: dst page capacity {} short of {} slots",
                    capacity,
                    dst_start_slot + num_tokens
                );
            }
            let mut list = if dst.committed_hashes.is_empty() {
                Vec::new()
            } else {
                self.gpu_stores[driver_idx].physical_ids(&dst.committed_hashes)
            };
            list.extend_from_slice(&dst.working_pages);
            list
        };

        let segments = build_segments(
            page_size,
            &src_page_list,
            &dst_page_list,
            src_token_start,
            dst_start_slot,
            num_tokens,
        )?;
        Ok(Some((segments, src_range, dst_start_slot)))
    }

    /// Phase B (actor, on `Message::AdoptKvComplete`): unpin both contexts
    /// and, if the driver confirmed the copy, append the adopted tokens'
    /// metadata. Not billed: no forward compute was spent on them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adopt_kv_complete(
        &mut self,
        dst_id: ContextId,
        src_id: ContextId,
        copy_result: Result<(), String>,
        src_range: Vec<(u32, u32)>,
        prefix_slots: u32,
        dst_start: u32,
        response: oneshot::Sender<Result<u32>>,
    ) {
        let result = match copy_result {
            Ok(()) => match self.contexts.get_mut(&dst_id) {
                Some(dst) => {
                    let forward_id = dst.next_forward_id;
                    dst.next_forward_id = dst.next_forward_id.wrapping_add(1);
                    for (i, (token, position)) in src_range.into_iter().enumerate() {
                        dst.working_page_tokens.push(super::TokenInfo {
                            token,
                            position,
                            mask: canonical_mask(prefix_slots, dst_start, i as u32),
                            adapter: None,
                            adapter_seed: None,
                            forward_id,
                        });
                    }
                    dst.driver_repaired_spec_tail = 0;
                    Ok(dst_start)
                }
                None => Err(anyhow::anyhow!("adopt_kv: dst context destroyed mid-copy")),
            },
            Err(e) => Err(anyhow::anyhow!("adopt_kv: device copy failed: {e}")),
        };

        // Unpin AFTER the append so a deferred pending_suspend sees the
        // final metadata (suspension keeps working_page_tokens for replay).
        self.unpin(dst_id);
        self.unpin(src_id);
        self.publish_context_counts(dst_id);

        let _ = response.send(result);
    }
}
