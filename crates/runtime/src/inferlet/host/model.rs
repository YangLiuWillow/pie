//! pie:core/model - Model and tokenizer global functions.
//!
//! The runtime serves exactly one model, so these are free functions over the
//! single global [`crate::model::Model`] rather than resource methods.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

/// A catalog stream as the WIT enum spells it.
pub fn lane_stream(stream: models::Stream) -> pie::inferlet::model::LaneStream {
    use pie::inferlet::model::LaneStream;
    match stream {
        models::Stream::Text => LaneStream::Text,
        models::Stream::Image => LaneStream::Image,
        models::Stream::Video => LaneStream::Video,
        models::Stream::Audio => LaneStream::Audio,
        models::Stream::Context => LaneStream::Context,
        models::Stream::Reference => LaneStream::Reference,
    }
}

/// The WIT enum as the catalog spells it.
pub fn catalog_stream(stream: pie::inferlet::model::LaneStream) -> models::Stream {
    use pie::inferlet::model::LaneStream;
    match stream {
        LaneStream::Text => models::Stream::Text,
        LaneStream::Image => models::Stream::Image,
        LaneStream::Video => models::Stream::Video,
        LaneStream::Audio => models::Stream::Audio,
        LaneStream::Context => models::Stream::Context,
        LaneStream::Reference => models::Stream::Reference,
    }
}

/// One catalog axis role as the facts spell it.
fn axis_role(role: models::AxisRole) -> pie::inferlet::model::AxisRole {
    use pie::inferlet::model::AxisRole;
    match role {
        models::AxisRole::Time => AxisRole::Time,
        models::AxisRole::Height => AxisRole::Height,
        models::AxisRole::Width => AxisRole::Width,
        models::AxisRole::Index => AxisRole::Index,
    }
}

/// One catalog reading as `model.readings()` answers it.
fn reading_fact(reading: &models::ReadingFact) -> pie::inferlet::model::ReadingFact {
    use pie::inferlet::model::{PortKind, ReadoutKind};
    pie::inferlet::model::ReadingFact {
        name: reading.name.to_string(),
        index: reading.index,
        has_kv: reading.has_kv,
        takes_tokens: reading.takes_tokens,
        streams: reading.streams.iter().copied().map(lane_stream).collect(),
        ports: reading
            .ports
            .iter()
            .map(|port| pie::inferlet::model::PortFact {
                name: port.name.to_string(),
                kind: match port.kind {
                    models::PortKind::Latents => PortKind::Latents,
                    models::PortKind::LaneVector => PortKind::LaneVector,
                    models::PortKind::Context => PortKind::Context,
                    models::PortKind::AxisPositions => PortKind::AxisPositions,
                    models::PortKind::Voxels => PortKind::Voxels,
                },
                width: port.width,
                // Every float port is fed from an f32 channel: the WIT dtype
                // set has no bf16, and the engine marshals at the feed.
                dtype: pie::inferlet::types::Dtype::F32,
                streams: port.streams.iter().copied().map(lane_stream).collect(),
                rows: port.rows,
            })
            .collect(),
        positions: reading.positions.as_ref().map(|convention| {
            pie::inferlet::model::PositionConvention {
                axes: convention.axes.iter().copied().map(axis_role).collect(),
                text_axis: convention.text_axis,
                text_origin: convention.text_origin,
                image_follows_text: convention.image_follows_text,
            }
        }),
        readout: match reading.readout {
            models::ReadoutKind::Logits => ReadoutKind::Logits,
            models::ReadoutKind::Velocity => ReadoutKind::Velocity,
            models::ReadoutKind::Hidden => ReadoutKind::Hidden,
            models::ReadoutKind::Pixels => ReadoutKind::Pixels,
        },
        readout_width: reading.readout_width,
    }
}

impl pie::inferlet::model::Host for ProcessCtx {
    async fn name(&mut self) -> Result<String> {
        Ok(model::model().name().to_string())
    }

    async fn architecture(&mut self) -> Result<String> {
        Ok(model::model().arch_name().to_string())
    }

    async fn default_system_speculation(&mut self) -> Result<bool> {
        Ok(false)
    }

    async fn mtp_depth(&mut self) -> Result<u32> {
        Ok(model::model().eta_caps().mtp_depth)
    }

    async fn draft_block(&mut self) -> Result<Option<pie::inferlet::model::BlockDrafter>> {
        let caps = model::model().eta_caps();
        Ok(
            (caps.draft_block > 0).then(|| pie::inferlet::model::BlockDrafter {
                rows: caps.draft_block,
                mask_token: caps.draft_mask_token,
                bidirectional: caps.draft_bidirectional,
                proposals_from: caps.draft_proposals_from,
            }),
        )
    }

    /// Which forward-pass interface the bound model requires, keyed on state
    /// semantics: recurrent state is present iff the engine handshake
    /// reports a non-zero folded-state size; paged KV is present iff the
    /// model has a KV page size.
    async fn pass_kind(&mut self) -> Result<pie::inferlet::model::ForwardKind> {
        use pie::inferlet::model::ForwardKind;
        let model = model::model();
        // A diffusion row states its canvas on the catalog; the kind is
        // that statement, not a reading of its page sizes (which are an
        // attention model's).
        if model.diffusion().is_some() {
            return Ok(ForwardKind::Diffusion);
        }
        let has_rs = model.rs_caps().state_size > 0;
        let has_kv = model.kv_page_size() > 0;
        Ok(match (has_kv, has_rs) {
            (_, false) => ForwardKind::Attention,
            (true, true) => ForwardKind::Hybrid,
            (false, true) => ForwardKind::Recurrent,
        })
    }

    /// The canvas a diffusion row denoises; `None` for every other kind.
    async fn canvas(&mut self) -> Result<Option<pie::inferlet::model::CanvasShape>> {
        Ok(model::model()
            .diffusion()
            .map(|d| pie::inferlet::model::CanvasShape {
                length: d.canvas,
                hidden: d.hidden,
                self_cond_taps: d.self_cond_taps,
            }))
    }

    /// The family's readings (design D12), in index order; empty for a
    /// text row.
    async fn readings(&mut self) -> Result<Vec<pie::inferlet::model::ReadingFact>> {
        Ok(model::model().readings().iter().map(reading_fact).collect())
    }

    /// The latent space a denoiser works in; `None` for a text row.
    async fn latent(&mut self) -> Result<Option<pie::inferlet::model::LatentSpace>> {
        Ok(model::model().generative().and_then(|g| g.latent).map(|l| {
            pie::inferlet::model::LatentSpace {
                channels: l.channels,
                patch_t: l.patch_t,
                patch_h: l.patch_h,
                patch_w: l.patch_w,
                spatial_compression: l.spatial_compression,
                temporal_compression: l.temporal_compression,
            }
        }))
    }

    /// The schedule the denoiser was trained under; `None` when nothing
    /// denoises.
    async fn schedule(&mut self) -> Result<Option<pie::inferlet::model::ScheduleFact>> {
        use pie::inferlet::model::ScheduleKind;
        Ok(model::model()
            .generative()
            .and_then(|g| g.schedule.as_ref())
            .map(|s| pie::inferlet::model::ScheduleFact {
                kind: match s.kind {
                    models::ScheduleKind::Flow => ScheduleKind::Flow,
                    models::ScheduleKind::Epsilon => ScheduleKind::Epsilon,
                    models::ScheduleKind::V => ScheduleKind::V,
                },
                shift: s.shift,
                train_steps: s.train_steps,
                boundary: s.boundary,
                pinned_sigmas: s.pinned_sigmas.clone(),
                stream_shifts: s
                    .stream_shifts
                    .iter()
                    .map(|&(stream, shift)| pie::inferlet::model::StreamShift {
                        lane: lane_stream(stream),
                        shift,
                    })
                    .collect(),
            }))
    }

    /// The most latent rows one pass carries; 0 without a float lane.
    async fn max_latent_rows(&mut self) -> Result<u32> {
        Ok(model::model().generative().map_or(0, |g| g.max_rows))
    }

    /// LM-head output dimension (`hf_config.vocab_size`), not the tokenizer
    /// vocab.
    async fn output_vocab_size(&mut self) -> Result<u32> {
        Ok(model::model().vocab_size())
    }

    async fn kv_page_size(&mut self) -> Result<u32> {
        Ok(model::model().kv_page_size())
    }

    /// Waves per frame (k) — the static deployment constant `forward.submit`
    /// sizes its slot list to. Fixed at runtime start, like `kv-page-size`.
    async fn frame_size(&mut self) -> Result<u32> {
        Ok(crate::scheduler::configured_frame_size() as u32)
    }

    /// Bound on how long a pipeline may hold a frame's wait-set without
    /// submitting. See `scheduler::configured_submit_deadline`.
    async fn submit_deadline_us(&mut self) -> Result<u64> {
        Ok(crate::scheduler::configured_submit_deadline().as_micros() as u64)
    }

    /// Host-reader channel capacity, in cells, that sustains the runtime's
    /// run-ahead for one lane. Includes the staging margin; see
    /// `scheduler::channel_capacity`.
    async fn channel_capacity(&mut self) -> Result<u32> {
        Ok(crate::scheduler::channel_capacity() as u32)
    }

    /// The run-ahead window in fires; see `scheduler::run_ahead_window`.
    async fn run_ahead_window(&mut self) -> Result<u32> {
        Ok(crate::scheduler::run_ahead_window() as u32)
    }

    /// Max embed tokens in a single pass (C) — the guest-side prefill chunk
    /// budget, sourced from the bound engine's structural per-launch token
    /// capacity.
    async fn max_embed_length(&mut self) -> Result<u32> {
        Ok(crate::engine::get_spec(0)?.limits.max_forward_tokens as u32)
    }

    /// The prefill chunk the scheduler would like right now: the forward
    /// token budget shared evenly among live processes, in whole KV pages.
    /// See `model.wit`.
    async fn prefill_chunk_hint(&mut self) -> Result<u32> {
        let budget = crate::engine::get_spec(0)?.limits.max_forward_tokens;
        let live = crate::inferlet::process::live_count().max(1);
        let page = (model::model().kv_page_size() as usize).max(1);
        let share = (budget / live) / page * page;
        Ok(share.clamp(page.min(budget.max(1)), budget.max(1)) as u32)
    }

    // working-set / arena capabilities, global over the bound model.

    /// Bytes of one folded recurrent-state object (0 if the model has no RS).
    async fn rs_state_size(&mut self) -> Result<u64> {
        Ok(model::model().rs_caps().state_size)
    }

    /// Tokens per buffered RS page (0 if the model has no RS).
    async fn rs_buffer_page_size(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().buffer_page_size)
    }

    /// Fold granularity in tokens; 1 = unconstrained. An RS fold of `n`
    /// tokens requires `n` to be a positive multiple of this.
    async fn rs_fold_granularity(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().fold_granularity)
    }

    /// KV page size (tokens) of the bound model.
    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::store::registry::get(0, 0).kv_page_size as u64)
    }
}
