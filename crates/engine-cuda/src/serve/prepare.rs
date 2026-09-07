//! Host half of one fire step (`FrameShell::prepare`, plus the `enqueue` and
//! `settle` verbs of the same impl): the admission gate, descriptor-port
//! reads, composition, page geometry, the window table and mask bits — no
//! stream is touched here.

use engine::fire::{Boundary, FoldLen, Masking, RsReset, RsVerb};
use engine::frame::{Demand, Shell as FrameShell};
use model_exec::fire::{FireDescriptor, Lane as FireLane, compose_axes};

use crate::error::{Fault, Result};
use crate::record;
use crate::run::RsMove;
use crate::store::kv::{self, Seat};
use crate::window::Windows;

use super::{
    Enqueued, FireCost, MROPE_COORDS, Media, MergeLand, PATCH_ROUTE_DROP, PortFeedPlan, Prepared,
    RsFire, Settled, Shell, StepView, VoxelFeedPlan,
};

/// `prepare`: host-only (gate, ports, compose, lane loop, geometry, windows,
/// mask). `enqueue`: stream-only (prologue, memsets, staging write, tables,
/// schedule, the walk). `settle`: post-sync (readback, capture, epilogue,
/// `held`).
/// How many resolved window tables a shell keeps (see step 4 of `prepare`).
const WINDOWS_MEMO: usize = 8;

/// One resolved window table and what it was resolved from.
pub(super) struct WindowsMemo {
    tables: [model_exec::fire::WindowTable; 3],
    indptr_host: Vec<i32>,
    bucket: u32,
    copies: bool,
    windows: Windows,
    packed: Vec<i32>,
}

impl FrameShell for Shell {
    type Step<'a> = StepView<'a>;
    type Prepared<'a> = Prepared<'a>;
    type Enqueued<'a> = Enqueued<'a>;
    type Settled = Settled;
    type Error = Fault;

    fn prepare<'a>(
        &mut self,
        step: StepView<'a>,
        prev: Option<&Prepared<'a>>,
    ) -> Result<Prepared<'a>>
    where
        Self: 'a,
    {
        // One step per frame, so there is never a predecessor; `prev` is kept
        // in the signature for a future caller that needs it.
        let _ = prev;
        let StepView {
            lanes,
            attachments,
            media,
            ..
        } = step;
        let arming = self.arming;
        let copies = self.copies;

        // 0. The gate: nothing has launched, so a refusal here is free. Only
        // checks the submission's own shape (lane existence, no doubly-
        // attached instance).
        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            // A program reading the token plane needs a text that plants
            // one; refused here, where nothing has launched, rather than at
            // `Session::fire`'s unbound guard after the forward has run.
            if self.exports.drafts.is_none() && self.programs.needs_mtp_drafts(attached.instance)? {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} reads the `mtp_drafts` intrinsic and this load's model \
                         text plants no `{}` export, so there is no token plane to point it at",
                        attached.instance,
                        crate::exports::DRAFTS_SEAM
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
        }

        // Run-ahead (on wherever the boot document seats more than one frame
        // in flight): a decode-envelope lane can be built from
        // host state alone — the token is injected device-to-device (slice 1)
        // and positions/kv_len are provably equal to what the host computes
        // (`positions_for`'s natural check and `check_extent` enforce it). So
        // its ports need no device read, the reap a read would force is
        // skipped, and the host launches the next frame ahead of this
        // epilogue. Only when EVERY attachment qualifies: a `DeviceGeometry`
        // lane still derives its whole geometry on the device and must reap.
        // (A decode-envelope instance is always single-lane — it never owns
        // pages — so the per-lane injection needs no multi-lane handling.)
        let can_runahead = self.runahead.runs_ahead()
            && attachments.iter().all(|attached| {
                match self.programs.geometry_of(attached.instance) {
                    None | Some(eta_ir::registry::GeometryClass::Host) => true,
                    Some(eta_ir::registry::GeometryClass::DecodeEnvelope) => {
                        // Only a genuinely 1-wide decode lane: one token per
                        // step, so positions (have+0) and kv_len (have+1) are
                        // deterministic and the skipped checks are provably
                        // redundant. A multi-row fire is a speculative verify
                        // whose accepted count is the device's to decide — its
                        // positions/kv_len are NOT host-derivable, so it keeps
                        // the device read (and the reap) until the async
                        // settle-validation path lands.
                        self.programs.token_device_source(attached.instance).is_some()
                            && lanes
                                .get(attached.lane as usize)
                                .is_some_and(|seated| seated.lane.tokens.len() == 1)
                    }
                    Some(eta_ir::registry::GeometryClass::DeviceGeometry) => false,
                }
            });

        // 0b. Descriptor ports, read off the rings the gate just approved.
        // Read here, before the prologue, since a prologue's commit would
        // move the cursors under a later read. A `GeometryClass::Host` lane
        // resolves `None` and reads the submission unchanged.
        if !can_runahead
            && self.owed.is_some()
            && attachments.iter().any(|attached| {
                self.programs
                    .geometry_of(attached.instance)
                    .is_some_and(|class| class != eta_ir::registry::GeometryClass::Host)
            })
        {
            self.reap_guests_at("prepare")?;
        }
        super::btrace::mark("reap");
        let mut resolved: Vec<crate::program::Envelope> = Vec::new();
        // Per submission lane: the device-side token source, when the token
        // can be injected device-to-device instead of round-tripped through
        // the host. Recorded whether the lane resolves its envelope (slice 1)
        // or is built host-side under run-ahead.
        let mut token_src_of: Vec<Option<(u64, u32)>> = vec![None; lanes.len()];
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        for attached in attachments {
            let first = attached.lane as usize;
            let src = self.programs.token_device_source(attached.instance);
            // Run-ahead builds a decode-envelope lane from host state: skip
            // the device read, leave `envelope_of` `None` so the assembly
            // takes the host path (submission placeholder token, overwritten
            // device-side by the injection; positions host-computed).
            if can_runahead
                && self.programs.geometry_of(attached.instance)
                    == Some(eta_ir::registry::GeometryClass::DecodeEnvelope)
            {
                if first < lanes.len() {
                    token_src_of[first] = src;
                }
                continue;
            }
            let Some(envelope) = self.programs.envelope(attached.instance)? else {
                continue;
            };
            let carried = envelope.lanes();
            if first + carried > lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached at lane {first} and its `embed_indptr` \
                         port describes {carried} lane(s), which runs past the {} this \
                         fire carries",
                        attached.instance,
                        lanes.len()
                    ),
                ));
            }
            let held = resolved.len();
            for lane in 0..carried {
                if envelope_of[first + lane].is_some() {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {} is claimed by two attached instances; a lane's \
                             descriptor ports have one author",
                            first + lane
                        ),
                    ));
                }
                envelope_of[first + lane] = Some((held, lane));
            }
            if first < lanes.len() {
                token_src_of[first] = src;
            }
            resolved.push(envelope);
        }

        // 0c. Two device-resolved payloads (page table, mask), built once
        // here and indexed by submission lane. The mask is run-length
        // encoded the same way a host-stated mask is, so both reach the
        // attention arm as the same slab.
        //
        // A device-geometry submission carries no row split (`Lane::tokens`
        // ships empty; the split lives on the `embed_indptr` port), so the
        // row count is read off the port here.
        //
        // A guest holds working-set-relative page indexes, never a pool page
        // id, so a device-resolved page or write reference is translated
        // through `Seated::translation` right here.
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; lanes.len()];
        let mut device_masks: Vec<Option<Masking>> = vec![None; lanes.len()];
        let mut lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();
        for source in 0..lanes.len() {
            let Some((held, at)) = envelope_of[source] else {
                continue;
            };
            let ports = resolved[held].lane(at, source)?;
            let table = lanes[source].translation;
            // An out-of-range relative index is refused rather than
            // translated by identity.
            let translate = |page: u32, port: &str| -> Result<u32> {
                table.get(page as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source}'s `{port}` port names working-set page {page} and the \
                             table this fire was handed maps {} page(s); a guest holds relative \
                             indexes and the pool's ids are the runtime's, so an index past the \
                             table addresses somebody else's cache",
                            table.len()
                        ),
                    )
                })
            };
            device_pages[source] = ports
                .pages()?
                .map(|relative| {
                    relative
                        .iter()
                        .map(|&page| translate(page, "pages"))
                        .collect::<Result<Vec<u32>>>()
                })
                .transpose()?;
            if ports.owns_pages() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            // `w_slot` is a page reference like `pages`; `w_off` is a plain
            // in-page offset, in no space at all.
            device_writes[source] = ports
                .writes(rows)?
                .map(|(slots, offsets)| {
                    Ok::<(Vec<u32>, Vec<u32>), Fault>((
                        slots
                            .iter()
                            .map(|&page| translate(page, "w_slot"))
                            .collect::<Result<Vec<u32>>>()?,
                        offsets.to_vec(),
                    ))
                })
                .transpose()?;
            if let Some((cells, stride)) = ports.mask(rows)? {
                // A device-resolved mask is refused for more than one query
                // row: `mask::stage` intersects each row against `k <= have +
                // q`, which is only honest in the one-row case for a
                // device-geometry lane's guest-defined write order.
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane whose \
                             write descriptor is the guest's has no such order this shell \
                             can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(crate::mask::from_dense(cells, stride));
            }
        }

        // The voxel-axis submission (D8): one entry per lane at most, only
        // against a load that seats the axis, the payload one port row per
        // voxel. Laid out in fire order once the composition has placed it.
        let mut clips_of: Vec<Option<crate::voxels::Clips<'_>>> = vec![None; lanes.len()];
        for shot in step.voxels {
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "the lane index is past the submission",
                });
            }
            if clips_of[at].is_some() {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "a lane's clips are one submission",
                });
            }
            if shot.clips.iter().any(|b| b.contains(&0)) {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "a clip's box has a zero side",
                });
            }
            clips_of[at] = Some(*shot);
        }

        // The patch-axis submission is checked here, before anything
        // launches: past this, `layout.scatter_rows` is an unchecked
        // indexed write.
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes);
        // Position-gather width, from the load, not the submission.
        let embed_taps = self.patch_seat.map_or(0, |seat| seat.embed_taps);
        let embed_weight_taps = self.patch_seat.map_or(0, |seat| {
            if seat.embed_weights {
                seat.embed_taps
            } else {
                0
            }
        });
        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "images were submitted for lane {at} and this fire carries {}",
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {at} was handed two media rows, and a lane's images are one \
                         concatenation with one patch order"
                    ),
                ));
            }
            let patch_rows: u64 = shot.rows.iter().map(|&rows| u64::from(rows)).sum();
            // Payload bytes, geometry rows and plan width must agree.
            let need = patch_rows * row_bytes;
            if need != shot.patches.len() as u64 || patch_rows != shot.routes.len() as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need,
                    have: shot.patches.len() as u64,
                });
            }
            // Two rotation streams: three numbers per patch row; three per
            // token row, which may be empty and reads as scalar `(p, p, p)`.
            if shot.positions.len() as u64 != patch_rows * MROPE_COORDS as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: patch_rows * MROPE_COORDS as u64,
                    have: shot.positions.len() as u64,
                });
            }
            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64
                    != u64::from(lane_rows[at]) * MROPE_COORDS as u64
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: u64::from(lane_rows[at]) * MROPE_COORDS as u64,
                    have: shot.token_positions.len() as u64,
                });
            }
            // Position-gather id/weight streams are checked exactly against
            // the plan's own tap counts (0 taps = no table).
            for (what, have, owed) in [
                (
                    "the position table's gather rows",
                    shot.embed_rows.len() as u64,
                    patch_rows * embed_taps,
                ),
                (
                    "the position table's interpolation weights",
                    shot.embed_weights.len() as u64,
                    patch_rows * embed_weight_taps,
                ),
            ] {
                let _ = what;
                if have != owed {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        need: owed,
                        have,
                    });
                }
            }
            // Routes are checked against this lane's own token rows (rebased
            // later, once composed). `-1` marks "no destination" and is
            // legal only when the plan declares an op that honours it
            // (`self.drops_patch_rows`).
            let rows = lane_rows[at];
            let drop = self.drops_patch_rows;
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows)
            }) {
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        super::btrace::mark("ports");
        // 1. Lane words in; `compose_axes` seriates the patch axis beside
        // the token one.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match (media_of[at], clips_of[at]) {
                (None, None) => FireLane::new(seated.lane.word, rows),
                (Some(shot), _) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().sum(),
                ),
                (None, Some(clips)) => FireLane::with_clips(
                    seated.lane.word,
                    rows,
                    clips.clips.len() as u32,
                    u32::try_from(clips.voxels()).unwrap_or(u32::MAX),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        // **THE ROWS A READER TAKES.** The trunk head runs over these and
        // no others (`layout.gather_rows` compacts them out of the token
        // rectangle), so a prefill's head is the size of a decode's rather
        // than the size of its prompt. Laid out in FIRE row order, which
        // makes the gather a monotone read; each submitted lane's run is
        // recorded so the readback can turn a lane into a row of the
        // gathered logits.
        let mut readout_rows: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut readout_first = vec![0u32; lanes.len()];
        let mut readout_count = vec![0u32; lanes.len()];
        {
            let mut placed: Vec<(u32, usize)> = composition
                .lanes()
                .iter()
                .map(|row| (row.row_offset, row.source as usize))
                .collect();
            placed.sort_unstable();
            for (row_offset, source) in placed {
                let owned = composition
                    .lanes()
                    .iter()
                    .find(|row| row.source as usize == source)
                    .map_or(0, |row| row.rows);
                let stated = lanes.get(source).and_then(|seated| seated.readout);
                let wanted: Vec<u32> = match stated {
                    Some(rows) if !rows.is_empty() => rows.to_vec(),
                    // The default readout, and the one every decode lane
                    // takes: the lane's last row.
                    _ => vec![owned.saturating_sub(1)],
                };
                readout_first[source] = readout_rows.len() as u32;
                readout_count[source] = wanted.len() as u32;
                for row in wanted {
                    if row >= owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(row) + 1,
                            have: u64::from(owned),
                        });
                    }
                    readout_rows.push(i32::try_from(row_offset + row).unwrap_or(0));
                }
            }
        }
        let descriptor = FireDescriptor::of(&composition);

        // 1b. The D2 packing tables: which group each fire lane joins, and
        // per selection the plan reads, the packed order and its CSRs. Built
        // over the composition's fire order from the `(stream, group)` the
        // runtime stated per lane. A plan reading no table builds none.
        let lane_facts: Vec<model_exec::fire::LaneFacts> = lanes
            .iter()
            .map(|seated| model_exec::fire::LaneFacts {
                stream: seated.stream,
                group: seated.group,
            })
            .collect();
        let (group_of_lane, packings) = if self.feeds.selections.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let groups = model_exec::fire::group_of_lane(composition.lanes(), &lane_facts);
            let mut packings = Vec::with_capacity(self.feeds.selections.len());
            for &select in &self.feeds.selections {
                let packed = model_exec::fire::pack(
                    select,
                    composition.lanes(),
                    &lane_facts,
                    &groups,
                    composition.rows(),
                )
                .map_err(model_exec::Error::Fire)?;
                packings.push(packed);
            }
            (groups, packings)
        };

        // 1b'. The ports merged straight into a stream: every lane the arm
        // selects lands its rows in the merged column — from the port when
        // the lane feeds it, zeros otherwise.
        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated
                    .ports
                    .iter()
                    .any(|feed| feed.kind == merged.seat.kind && feed.port == merged.seat.port);
                merge_lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        fire_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    fed,
                });
            }
        }

        // 1c. The D3 port feeds: every port a lane's class reads must be fed
        // from one of its channels, through the instance attached to it (the
        // `SelfCondInput::channels` precedent); the cell must be exactly the
        // lane's rows of the port's width. Checked here, before any stream
        // is touched; the cell's address is resolved at `enqueue`, after the
        // prologue, so it is the cell the instance's own `take` would read.
        let mut port_feeds: Vec<PortFeedPlan> = Vec::new();
        let mut voxel_feeds: Vec<VoxelFeedPlan> = Vec::new();
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for feed in seated.ports {
                if !self
                    .feeds
                    .ports
                    .iter()
                    .any(|(seat, _)| seat.kind == feed.kind && seat.port == feed.port)
                {
                    return Err(Fault::program(
                        "serve::ports",
                        format!(
                            "lane {} feeds {:?} port {} and this plan declares no such port",
                            row.source, feed.kind, feed.port
                        ),
                    ));
                }
            }
            for (seat, readers) in &self.feeds.ports {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let Some(feed) = seated
                    .ports
                    .iter()
                    .find(|feed| feed.kind == seat.kind && feed.port == seat.port)
                else {
                    // The arming pass feeds nothing: its synthetics compute
                    // nobody's numbers, and the rectangle stands as the load
                    // left it (zeros) or the last fire's.
                    if arming {
                        continue;
                    }
                    // A voxel port has TWO feeds (design D8), and a lane
                    // takes one of them: a channel (this loop) or the
                    // payload it submitted beside its clips
                    // (`StepVoxels::payload`, the shell's own door). A lane
                    // that handed over bytes has fed the port already.
                    if seat.kind == engine::fire::PortKind::Voxels
                        && clips_of
                            .get(row.source as usize)
                            .and_then(Option::as_ref)
                            .is_some_and(|shot| !shot.payload.is_empty())
                    {
                        continue;
                    }
                    return Err(Fault::program(
                        "serve::ports",
                        format!(
                            "lane {} runs a class that reads {:?} port {} and feeds it no \
                             channel; a declared port is fed from a channel cell at every \
                             submit (`Lane::ports`)",
                            row.source, seat.kind, seat.port
                        ),
                    ));
                };
                let instance = attachments
                    .iter()
                    .find(|attached| attached.lane == row.source)
                    .map(|attached| attached.instance)
                    .ok_or_else(|| {
                        Fault::program(
                            "serve::ports",
                            format!(
                                "lane {} feeds {:?} port {} off channel {} but attaches no \
                                 instance; a port cell is read through the instance that \
                                 carries the channel",
                                row.source, seat.kind, seat.port, feed.channel
                            ),
                        )
                    })?;
                // A voxel port's cell is the LANE'S CLIP, not its token rows
                // (design D8): its rectangle lives in the voxel store, at the
                // lane's `voxel_offset`, and what the cell owes is one row per
                // voxel of the clips the lane submitted.
                let voxel = seat.kind == engine::fire::PortKind::Voxels;
                let rows = if voxel {
                    row.voxels
                } else if seat.per_lane() {
                    1
                } else {
                    row.rows
                };
                let bytes = u64::from(rows) * seat.row_bytes();
                // A cell in the port's own element copies; an f32 cell into
                // a bf16 port (design D3: latents stay f32 masters on the
                // ring, a text may read them in bf16) is cast on the way.
                let f32_bytes = u64::from(rows) * u64::from(seat.width) * 4;
                let cell = self.programs.feed_cell_bytes(instance, feed.channel)?;
                let cast =
                    cell != bytes && seat.dtype == model_ir::Dtype::Bf16 && cell == f32_bytes;
                if cell != bytes && !cast {
                    return Err(Fault::program(
                        "serve::ports",
                        format!(
                            "lane {} feeds {:?} port {} off channel {} whose cell is {cell} \
                             bytes, and the port wants {rows} row(s) x {} {:?} = {bytes} \
                             bytes (the lane's rows by the port's width; an f32 cell of \
                             {f32_bytes} bytes would be cast)",
                            row.source, seat.kind, seat.port, feed.channel, seat.width, seat.dtype
                        ),
                    ));
                }
                if voxel {
                    voxel_feeds.push(VoxelFeedPlan {
                        lane: row.source,
                        first: row.voxel_offset,
                        rows,
                        width: seat.width,
                        cast,
                        bytes,
                        channel: feed.channel,
                        instance,
                    });
                    continue;
                }
                port_feeds.push(PortFeedPlan {
                    seat: *seat,
                    first: if seat.per_lane() {
                        fire_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows,
                    bytes,
                    cast,
                    channel: feed.channel,
                    instance,
                });
            }
        }
        // A fire whose voxel port is channel-fed stages no payload: the
        // width the tables state is the seat the feed named.
        let fed_width = voxel_feeds.first().map_or(0, |feed| feed.width);
        if voxel_feeds.iter().any(|feed| feed.width != fed_width) {
            return Err(Fault::VoxelPayload {
                lane: voxel_feeds[0].lane,
                what: "two lanes feed voxel ports of two widths in one fire (M0: one voxel \
                       class per fire)",
            });
        }
        // The voxel tables, in fire order. M0: every spatial launch runs over
        // the whole voxel rectangle (`crate::voxels`), so the clips of one
        // fire must fall in one class.
        let voxel_tables = if composition.voxel_rows() == 0 {
            crate::voxels::Tables::default()
        } else {
            let Some(store) = self.voxels.as_ref() else {
                return Err(Fault::VoxelPayload {
                    lane: clips_of.iter().position(Option::is_some).unwrap_or(0) as u32,
                    what: "this load seats no voxel axis",
                });
            };
            if composition.voxel_classes().present_in_order().count() > 1 {
                return Err(Fault::VoxelPayload {
                    lane: clips_of.iter().position(Option::is_some).unwrap_or(0) as u32,
                    what: "the clips of one fire fall in two classes, and a spatial launch \
                           runs over the whole voxel rectangle (M0: one voxel class per fire)",
                });
            }
            let slot_of: Vec<u32> = lanes.iter().map(|seated| seated.lane.slot).collect();
            crate::voxels::Tables::of(
                &store.seat(),
                composition.lanes(),
                &clips_of,
                &slot_of,
                fed_width,
            )?
        };

        // The composition places each lane's images independently of token
        // order: `patch_offset` is where its rows begin in the fire's patch
        // rectangle, `image_offset` where its images begin in the indptr.
        // Routes are rebased here by `row.row_offset`.
        let (
            patch_payload,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
        ) = if composition.patch_rows() == 0 {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        } else {
            let stride = row_bytes as usize;
            let mut payload = vec![0u8; composition.patch_rows() as usize * stride];
            // Default route is the drop sentinel, not zero: every entry no
            // lane writes has no destination, and zero is a legal token row.
            let mut routes = vec![
                if self.drops_patch_rows {
                    PATCH_ROUTE_DROP
                } else {
                    0
                };
                composition.patch_rows() as usize
            ];
            // Rotation stream `(t, h, w)` is each patch's own grid
            // coordinate, copied verbatim (unlike routes, which are rebased).
            let mut positions = vec![0i32; composition.patch_rows() as usize * MROPE_COORDS];
            // How many patch rows this plan folds into one tower output row.
            let fold = (self.patch_fold as usize).max(1);
            let taps = embed_taps as usize;
            let weight_taps = embed_weight_taps as usize;
            let mut embed_rows = vec![0i32; composition.patch_rows() as usize * taps];
            let mut embed_weights = vec![0f32; composition.patch_rows() as usize * weight_taps];
            let mut per_image = vec![0u32; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);
                // Routes land at `patch_offset / fold` (the fold's output
                // space), not at `patch_offset` (patch-row space). A
                // negative (sentinel) route is left untouched.
                let landed = (row.patch_offset as usize) / fold;
                let live = shot.rows.iter().map(|rows| *rows as usize).sum::<usize>() / fold;
                for (j, &route) in shot.routes.iter().take(live).enumerate() {
                    routes[landed + j] = if route < 0 {
                        route
                    } else {
                        route + row.row_offset as i32
                    };
                }
                let triples = row.patch_offset as usize * MROPE_COORDS;
                positions[triples..triples + shot.positions.len()].copy_from_slice(shot.positions);
                let at_ids = row.patch_offset as usize * taps;
                embed_rows[at_ids..at_ids + shot.embed_rows.len()].copy_from_slice(shot.embed_rows);
                let at_w = row.patch_offset as usize * weight_taps;
                embed_weights[at_w..at_w + shot.embed_weights.len()]
                    .copy_from_slice(shot.embed_weights);
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }
            // Indptr for the tower's attention: `images + 1` entries, image
            // `i` owns `[segments[i], segments[i + 1])`.
            let mut segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            segments.push(at);
            for rows in per_image {
                at += rows as i32;
                segments.push(at);
            }
            (
                payload,
                segments,
                routes,
                positions,
                embed_rows,
                embed_weights,
            )
        };
        let rows = composition.rows();

        // 2. Fire's own vectors, in fire (composition) order, not submission
        // order.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut kv_less_seats: Vec<bool> = Vec::with_capacity(lanes.len());
        // One mask entry per lane, seriated with the rest.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        // Device-to-device token injections, one per
        // single-lane device-resolved decode row: filled as `tokens` is
        // assembled so `dst_off` is that row's byte offset in the slab.
        let mut token_injects: Vec<crate::inputs::TokenInject> = Vec::new();
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // `Some((page, offset))` for a row with its own resolved
        // `w_slot`/`w_off`; `None` where `store::kv::geometry_with` derives
        // the landing place instead.
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // Slots that arrive fresh, decided here, zeroed in `enqueue`.
        let mut fresh: Vec<u32> = Vec::new();
        // Recurrent plan, in fire order — see `RsFire`.
        let mut rs_moves: Vec<RsMove<'a>> = Vec::with_capacity(lanes.len());
        let mut rs_lens: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut rs_order: Vec<u32> = vec![0; lanes.len()];
        // One entry per token row (not per lane); empty when no lane routes.
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            // Resolved ports for this lane, or `None` for a Host-geometry
            // lane / one with no attachment — then every read below is the
            // submission's, unchanged.
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            // `have` comes from whoever owns the page table: a shell-owned
            // slot uses `self.held`, a caller-owned one its own count. A
            // device-geometry lane states its post-append extent on
            // `kv_len`, and `have` is derived as `extent - rows`.
            let have = match ports.as_ref().filter(|ports| ports.owns_pages()) {
                // A lane whose reading binds no kv space holds nothing,
                // whatever the shell counted for its slot.
                _ if seated.kv_less => 0,
                Some(ports) => {
                    let after = ports.extent().ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states its own page table and binds no \
                                 `kv_len` port; the page count, the last page's fill and \
                                 the attention schedules are all carved from the extent, \
                                 and no seat in this shell knows it"
                            ),
                        )
                    })?;
                    if after < row.rows {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states a readable KV extent of {after} on \
                                 its `kv_len` port and this fire writes {} row(s) into \
                                 it; the extent is AFTER the append, so it can never be \
                                 shorter than what the append adds",
                                row.rows
                            ),
                        ));
                    }
                    after - row.rows
                }
                None => match seated.held {
                    Some(held) => held,
                    None => self
                        .held
                        .get(lane.slot as usize)
                        .copied()
                        .ok_or(Fault::Ceiling {
                            what: "slots",
                            need: u64::from(lane.slot) + 1,
                            have: self.held.len() as u64,
                        })?,
                },
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );
            // A sequence with `have == 0` gets its recurrent banks zeroed.
            // The decision is here; the memset itself happens in `enqueue`,
            // so a fire that later refuses never destroys state it declined
            // to rebuild. The classification (`seated.rs_reset`) is the RS
            // store's own, not derived from `have`.
            let begins = match seated.rs_reset {
                RsReset::Inferred => have == 0,
                RsReset::Fresh => true,
                RsReset::Held => false,
            };
            if begins {
                fresh.push(lane.slot);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            kv_less_seats.push(seated.kv_less);
            // Page table from whichever author has one: a device-geometry
            // lane's resolved cell, else the submission's own.
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            // Word and mask cross-checked once (`Fault::MaskWord`). The
            // effective mask is device-resolved OR the submission's.
            let masking = device_masks[source].as_ref().or(seated.mask);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if masking.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: masking,
                have,
                rows: row.rows,
                bidirectional: seated.bidirectional,
            });
            slot_ids.push(lane.slot as i32);
            // Fold length resolved here: a `FoldLen::Device` row's count
            // comes from the descriptor port read in step 0b, clamped to
            // the verb's bound.
            let fire_lane = rs_moves.len();
            rs_order[row.source as usize] = fire_lane as u32;
            let port = envelope_of[source].and_then(|(held, _)| resolved[held].fold_len.as_deref());
            let (verb, folded) = match &seated.rs {
                RsVerb::Fold => (RsMove::None, row.rows),
                // `fold == 0` is a pure scatter (boundary = row count,
                // invisible to length/split); nonzero lands the durable
                // state on that row while every row is still written.
                RsVerb::Buffer {
                    pages,
                    at,
                    fold,
                    replay,
                } => {
                    // **The buffer read path**: `replay` buffered tokens at
                    // `[at - replay, at)` are replayed through the recurrence
                    // ahead of this lane's rows, so the rows start from
                    // `folded (+) replay(buffer)`. The recurrent arms run over
                    // the EXTENDED run `[replay | rows]` (`Run::rs_extend`),
                    // and every count below — the fold, the split, the
                    // truncation — is taken in that layout, as the verb
                    // states it.
                    if *replay > *at {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} replays {replay} buffered token(s) below buffer \
                                 position {at}, which has only {at}",
                                row.source
                            ),
                        ));
                    }
                    let extended = replay.saturating_add(row.rows);
                    let fold = match fold {
                        FoldLen::Host(0) => 0,
                        stated => resolve_fold_len(*stated, extended, fire_lane, port)?,
                    };
                    (
                        RsMove::Scatter {
                            pages: pages.as_slice(),
                            at: *at,
                            fold,
                            replay: *replay,
                        },
                        if fold == 0 { extended } else { fold },
                    )
                }
                RsVerb::Window { .. } => {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {} asks the device-resident window verb, which this plane does \
                             not serve yet (it replays a buffered prefix ahead of the rows)",
                            row.source
                        ),
                    ));
                }
                RsVerb::FoldBuffered {
                    pages,
                    at,
                    bound,
                    len,
                } => {
                    let (bound, len) = (*bound, *len);
                    if bound != row.rows {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} replays a buffer bounded at {bound} tokens in a fire \
                                 that gave it {} rows — the bound IS what sizes the launch, so \
                                 the two are one number",
                                row.source, row.rows
                            ),
                        ));
                    }
                    (
                        RsMove::Gather {
                            pages: pages.as_slice(),
                            // A mid-page fold leaves survivors offset inside
                            // a shared page, so replay starts from `at`.
                            at: *at,
                        },
                        resolve_fold_len(len, bound, fire_lane, port)?,
                    )
                }
            };
            if verb != RsMove::None && self.buffers.is_none() {
                return Err(Fault::Unbound {
                    what: format!(
                        "lane {}'s recurrent verb, against a plan that declares no chunked \
                         recurrence to buffer",
                        row.source
                    ),
                });
            }
            rs_moves.push(verb);
            rs_lens.push(narrow(u64::from(folded)));
            // Adapter and word cross-checked once, as mask/word above.
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            // A block drafter's draft fire carries an adapted lane's id and no
            // trunk row: the correction cannot reach its class, so nothing is
            // owed and nothing is refused (`ClassTable::correction_reaches`).
            let unreachable = seated.adapter.is_some()
                && !runs_correction
                && !self
                    .compiled
                    .classes
                    .correction_reaches(&self.corrected, lane.word);
            if seated.adapter.is_some() != runs_correction && !unreachable {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            // Draft/capture axes cross-checked the same way; these carry no
            // payload, so the failure mode is "computed and nobody reads it".
            let runs_draft_arm = self
                .exports
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.classes.contains(row.class as usize));
            if seated.drafts && self.exports.mtp.is_none() {
                return Err(Fault::Draftless { lane: row.source });
            }
            // A BLOCK DRAFTER's column is not a class's: its `mtp` seam is the
            // shared head's output split by the `block_draft` fact, so the
            // writing region runs in every class and the arm costs a lane
            // nothing unless its word carries the fact. The word already
            // says which rows are the block's; there is no second axis to
            // cross-check against.
            let block_drafter = self.trace.drafter.is_some();
            if seated.drafts != runs_draft_arm && !block_drafter {
                return Err(Fault::DraftWord {
                    lane: row.source,
                    word: lane.word,
                    runs_draft_arm,
                });
            }
            let runs_capture_arm = self.exports.capturing.contains(row.class as usize);
            if seated.captures_scores && self.exports.scores.is_empty() {
                return Err(Fault::Scoreless { lane: row.source });
            }
            if seated.captures_scores != runs_capture_arm {
                return Err(Fault::ScoreWord {
                    lane: row.source,
                    word: lane.word,
                    runs_capture_arm,
                });
            }
            if any_adapter {
                // `-1` = base model.
                let id = seated
                    .adapter
                    .map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            // Host-class lane's tokens are the submission's; a
            // device-resolved lane's come from the port cell the previous
            // fire's epilogue wrote.
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    ports.check_extent(have.saturating_add(row.rows))?;
                    // The token is already on the device-only ring; inject it
                    // device-to-device at this row's offset rather than lean
                    // on the host having read it back. Single-lane instances
                    // only (a multi-lane cell packs several lanes' tokens),
                    // and only when the cell's width matches this row's.
                    let dst_off = tokens.len() as u64 * 4;
                    if let Some((src, native)) = token_src_of[source]
                        && native as usize == rows_here * 4
                    {
                        token_injects.push(crate::inputs::TokenInject {
                            dst_off,
                            src,
                            bytes: native as usize,
                        });
                    }
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    // Write descriptor already translated to pool pages
                    // (step 0c); `None` means the seat's own `have + row`
                    // arithmetic stands for that row.
                    match &device_writes[source] {
                        Some((slots, offsets)) => {
                            writes.extend(slots.iter().zip(offsets).map(|(&page, &off)| {
                                Some((narrow(u64::from(page)), narrow(u64::from(off))))
                            }))
                        }
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                }
                None => {
                    // Run-ahead: a decode-envelope lane arrives here (host
                    // path) with a submission placeholder token whose VALUE
                    // is overwritten device-side by the injection below; only
                    // its count matters. Positions are the natural run the
                    // device would have stated (`have + at`).
                    let dst_off = tokens.len() as u64 * 4;
                    if let Some((src, native)) = token_src_of[source]
                        && native as usize == rows_here * 4
                    {
                        token_injects.push(crate::inputs::TokenInject {
                            dst_off,
                            src,
                            bytes: native as usize,
                        });
                    }
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(narrow(u64::from(have) + at as u64));
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        // Trunk's triple-wide position stream, assembled from the scalar
        // one. Empty unless the plan declares it. Default triple is
        // `(p, p, p)`; a lane that states its own triples overwrites its
        // own interval.
        let mut mrope_positions = if !self.mrope_seat {
            Vec::new()
        } else {
            let mut triples = Vec::with_capacity(positions.len() * MROPE_COORDS);
            for &at in &positions {
                triples.extend_from_slice(&[at, at, at]);
            }
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                if shot.token_positions.is_empty() {
                    continue;
                }
                let at = row.row_offset as usize * MROPE_COORDS;
                triples[at..at + shot.token_positions.len()].copy_from_slice(shot.token_positions);
            }
            triples
        };
        // The denoiser's self-conditioning taps, `[rows, taps]` twice. Every
        // fire of a plan that declares them stages them — zeros for a lane
        // that carries none (an encode lane, an arming synthetic) — so the
        // input is bound whichever class runs.
        let taps = self.self_cond_taps as usize;
        let mut self_cond_feeds: Vec<(usize, usize, u64, u64, u64)> = Vec::new();
        let (mut self_cond_rows, mut self_cond_weights) = if taps == 0 {
            (Vec::new(), Vec::new())
        } else {
            let mut ids: Vec<i32> = Vec::with_capacity(positions.len() * taps);
            let mut ws: Vec<f32> = Vec::with_capacity(positions.len() * taps);
            for row in composition.lanes() {
                let seated = &lanes[row.source as usize];
                let cells = row.rows as usize * taps;
                match seated.self_cond {
                    Some(sc) if sc.channels.is_some() => {
                        let (rows_channel, weights_channel) = sc.channels.unwrap_or_default();
                        let instance = attachments
                            .iter()
                            .find(|attached| attached.lane == row.source)
                            .map(|attached| attached.instance)
                            .ok_or_else(|| {
                                Fault::program(
                                    "serve::prepare",
                                    format!(
                                        "lane {} reads its self-conditioning taps off channels but attaches no instance",
                                        row.source
                                    ),
                                )
                            })?;
                        if sc.taps as usize != taps {
                            return Err(Fault::program(
                                "serve::prepare",
                                format!(
                                    "lane {} states {} taps and this plan reads {taps}",
                                    row.source, sc.taps
                                ),
                            ));
                        }
                        self_cond_feeds.push((
                            ids.len(),
                            cells,
                            rows_channel,
                            weights_channel,
                            instance,
                        ));
                        ids.extend(std::iter::repeat_n(0, cells));
                        ws.extend(std::iter::repeat_n(0.0, cells));
                    }
                    Some(sc) => {
                        if sc.taps as usize != taps
                            || sc.rows.len() != cells
                            || sc.weight_bits.len() != cells
                        {
                            return Err(Fault::program(
                                "serve::prepare",
                                format!(
                                    "lane {} states self-conditioning taps of width {} over {} \
                                     ids, and this plan reads {taps} taps over the lane's {} rows",
                                    row.source,
                                    sc.taps,
                                    sc.rows.len(),
                                    row.rows
                                ),
                            ));
                        }
                        ids.extend(sc.rows.iter().map(|&id| i32::try_from(id).unwrap_or(0)));
                        ws.extend(sc.weights());
                    }
                    None => {
                        ids.extend(std::iter::repeat_n(0, cells));
                        ws.extend(std::iter::repeat_n(0.0, cells));
                    }
                }
            }
            (ids, ws)
        };

        super::btrace::mark("lanes");
        // 2b. Admission: the union demand of this step, committed atomically
        // before any of it runs. A demand is a watermark (highest addressed
        // page/slot + 1), not a count, since the arenas grow at the tail.
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        // A plan with no kv space (a denoiser, a VAE) seats no page: its
        // lanes' rows are latents, bounded by the token ceiling alone, and
        // its demand on the pools is nothing.
        let kv_less = self.spaces == 0;
        let demand = Demand {
            kv_pages: if kv_less {
                0
            } else {
                seats
                    .iter()
                    .zip(&tables)
                    .map(|(seat, table)| {
                        let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                        let pages = after.div_ceil(page_size).max(1);
                        if table.is_empty() {
                            // Shell-owned block: one past this lane's last page id.
                            self.pools.paging().base(seat.slot).saturating_add(pages)
                        } else {
                            // Runtime-tabled ids: one past the highest addressed.
                            table
                                .iter()
                                .take(pages as usize)
                                .copied()
                                .max()
                                .map_or(0, |page| u64::from(page).saturating_add(1))
                        }
                    })
                    .max()
                    .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX))
            },
            state_slots: seats
                .iter()
                .map(|seat| seat.slot.saturating_add(1))
                .max()
                .unwrap_or(0),
            workspace: 0,
        };
        // The kv planes are backed under the pages these seats address —
        // a slot's run for a shell-owned block, each tabled id for a
        // runtime-tabled one — not under every page below the watermark.
        let mut kv_ranges: Vec<(u64, u64)> = Vec::new();
        for (seat, table) in seats.iter().zip(&tables).filter(|_| !kv_less) {
            let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
            let pages = after.div_ceil(page_size).max(1);
            if table.is_empty() {
                kv_ranges.push((self.pools.paging().base(seat.slot), pages));
            } else {
                for &page in table.iter().take(pages as usize) {
                    kv_ranges.push((u64::from(page), 1));
                }
            }
        }
        self.pools.commit_frame(demand, &kv_ranges)?;
        super::btrace::mark("commit_frame");

        // 3. Page arithmetic, once per kv space.
        let indptr_host = kv::indptr(&seats)?;
        let paging = self.pools.paging();
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;
        super::btrace::mark("page_arith");
        // 3b. Explicit write descriptor overrides the derived `have + r`
        // landing, since several lanes appending into one shared pool would
        // otherwise collide at `have + 0`.
        if writes.iter().any(Option::is_some) {
            for geometry in &mut geometries {
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "row {row} states an explicit write descriptor and the \
                                 page arithmetic placed {} row(s)",
                                geometry.write_page.len()
                            ),
                        ));
                    };
                    *write_page = page;
                    *write_offset = offset;
                }
            }
        }
        // `pages` (page-id count) read here, before step 4d's lane padding,
        // since the lanes that padding adds own no page.
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        super::btrace::mark("admit");
        // 4. Windows: every template region resolved against this
        // composition's class table. A region that doesn't seat whole gets
        // `Fallback::Split` unless copies are enabled and the fallback table
        // asks for one at this fire's bucket, in which case it gets one
        // gathered window over the compacted rectangle instead.
        let bucket = self
            .budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        // Copy policy is stored in the segmentation memo (`Shell::segments`)
        // since it's the one input `Windows::admits` needs that
        // `record::BodyKey` doesn't carry. A masked fire always takes the
        // split, so mask/present-set alone determines admissibility from the
        // key; only `self.copies` (toggled per fire by `Shell::set_copies`)
        // sits outside it.
        let copies_here = copies && masks.iter().all(|lane| lane.mask.is_none());
        // **THE SAME COMPOSITION RESOLVES THE SAME WINDOWS.** Everything
        // `Windows::of` reads besides the load constants is the two class
        // tables, the row prefix sums, the bucket and the copy flag — and a
        // steady decode frame hands it the same four every step. Resolving
        // afresh walked every template region for ~110 us a frame, so the
        // last few answers are kept and handed back by equality (no hash to
        // collide). A table with a gathered window is never memoised: its
        // payload is built from this fire's page geometry.
        let class_tables = [
            composition.table(model_ir::RowAxis::Tokens),
            composition.table(model_ir::RowAxis::Patches),
            composition.table(model_ir::RowAxis::Voxels),
        ];
        let held = self.windows_memo.iter().position(|memo| {
            memo.bucket == bucket
                && memo.copies == copies_here
                && memo.indptr_host == indptr_host
                && memo.tables[0] == *class_tables[0]
                && memo.tables[1] == *class_tables[1]
                && memo.tables[2] == *class_tables[2]
        });
        let (mut windows, boundaries) = match held {
            Some(at) => {
                let memo = &self.windows_memo[at];
                (memo.windows.clone(), memo.packed.clone())
            }
            None => {
                let windows = Windows::of(
                    &self.trace,
                    &self.compiled,
                    // One table per row axis, addressed by the axis.
                    model_ir::PerAxis::new([class_tables[0], class_tables[1], class_tables[2]]),
                    &indptr_host,
                    crate::window::Copies {
                        bucket,
                        enabled: copies_here,
                        spaces: &geometries,
                    },
                    // Fixed-width slots, so a recorded body's baked `indptr`
                    // pointer is right for every fire of its key.
                    self.inputs.window_slots(),
                )?;
                let packed = windows.packed();
                if windows.copied() == 0 {
                    if self.windows_memo.len() >= WINDOWS_MEMO {
                        self.windows_memo.remove(0);
                    }
                    self.windows_memo.push(WindowsMemo {
                        tables: [
                            class_tables[0].clone(),
                            class_tables[1].clone(),
                            class_tables[2].clone(),
                        ],
                        indptr_host: indptr_host.clone(),
                        bucket,
                        copies: copies_here,
                        windows: windows.clone(),
                        packed: packed.clone(),
                    });
                }
                (windows, packed)
            }
        };
        // The synthetic pass is not the last fire anybody means.
        if !arming {
            self.last = FireCost {
                launches: windows.launches(),
                copied: windows.copied(),
            };
        }
        super::btrace::mark("windows_of");

        // 4b. Mask bits, expanded here once, off the same `have`/`rows` the
        // page geometry used. `None` means no lane masked.
        let staged = crate::mask::stage(&masks)?;
        super::btrace::mark("mask");

        // Body key's class ladder, built from the key's own coordinates
        // (bucket, decode class, lane ceiling), not this fire's actual
        // rows, so two fires of one bucket that split rows differently
        // reach the same body.
        let lane_ceiling = self.lane_ceiling();
        let token_axis = composition.axis(model_ir::RowAxis::Tokens);
        let patches = composition.axis(model_ir::RowAxis::Patches);
        let key = record::BodyKey::of_axes(
            &token_axis.classes,
            token_axis.bucket,
            &self.decoding,
            lane_ceiling,
            self.towered.then_some((&patches.classes, patches.bucket)),
        );
        let ladder = key.classes.clone();
        let patch_ladder = key.patch.as_ref().map(|axis| axis.classes.clone());
        // The four load-level clauses below are hoisted in front of the
        // per-fire ones, since a load that fails any of them would
        // otherwise mint a permanent memo entry no fire will ever read.
        let records_bodies = self.records_bodies();
        let (admits, world): (std::sync::Arc<[crate::window::Admit]>, bool) = if records_bodies {
            self.segmentation(
                &key,
                &windows,
                model_ir::PerAxis::from_fn(|axis| composition.axis(axis).rows),
                copies_here,
            )
        } else {
            (Vec::new().into(), true)
        };
        super::btrace::mark("segmentation");
        let bodied = records_bodies
            && !rs_moves.iter().any(|verb| !matches!(verb, RsMove::None))
            && Self::keyable_units(&self.compiled)
            // This fire must be in the world its key was derived in: a
            // resident body holds one script, so an other-world fire walks
            // eagerly instead of replaying a script cut for somebody else.
            && world
            // Cuttability asked last (the only clause that logs to an
            // operator), through a memo since it's a function of the key.
            && !self.cache.body_refused(&key)
            && self.cuttable(&key, admits.as_ref());
        super::btrace::mark("cuttable");
        // **A BODY BAKES ITS READOUT GRID, SO THE COUNT MUST BE THE KEY'S.**
        // The readout rectangle is carved and gridded at the lane ceiling —
        // the same number the key already carries — and padded with row
        // zero, which the gather reads and the readback never names. A fire
        // wanting more readouts than that ceiling (a multi-row readout on
        // many lanes) is not one a body can serve, so it walks.
        let readout_ceiling = ladder.lane_reach(lane_ceiling).min(self.budget.max_lanes);
        let bodied = bodied && readout_rows.len() <= readout_ceiling as usize;
        if bodied {
            readout_rows.resize(readout_ceiling as usize, 0);
        }

        // Arming pins the key a synthetic fire landed on
        // (`Shell::arm_bodies`).
        if arming {
            self.armed_body = bodied.then(|| key.clone());
        }

        // 4c-b. A bodied fire's whole-fire regions are gridded at the
        // bucket, so token ids/positions/mrope/adapter-routes/write
        // descriptors must all reach that far too. Padding is genuinely
        // empty: id 0, position 0, write descriptors and adapter routes
        // `-1`. Nothing off the bodies path moves a byte: `carve_rows` is
        // this fire's own row count there, and every resize is a no-op.
        let carve_rows = if bodied {
            composition.bucket().max(composition.rows())
        } else {
            composition.rows()
        };
        // The row-to-lane map, `[carve rows]`: each lane's rows name its fire
        // lane; the carve's padding names lane 0, a lane that exists, so a
        // padded row a retirement misses still reads a real vector.
        let mut lane_of_row: Vec<i32> = Vec::with_capacity(carve_rows as usize);
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            lane_of_row.extend(std::iter::repeat_n(fire_lane as i32, row.rows as usize));
        }
        lane_of_row.resize(carve_rows as usize, 0);
        if carve_rows > rows {
            tokens.resize(carve_rows as usize, 0);
            positions.resize(carve_rows as usize, 0);
            if !mrope_positions.is_empty() {
                mrope_positions.resize(carve_rows as usize * MROPE_COORDS, 0);
            }
            if taps > 0 {
                self_cond_rows.resize(carve_rows as usize * taps, 0);
                self_cond_weights.resize(carve_rows as usize * taps, 0.0);
            }
            if any_adapter {
                adapter_routes.resize(carve_rows as usize, -1);
            }
            for geometry in &mut geometries {
                geometry.write_page.resize(carve_rows as usize, -1);
                geometry.write_offset.resize(carve_rows as usize, -1);
            }
        }

        // 4d. Lane tables (pages, tokens, rows) padded to the bucket's lane
        // ceiling too, bodies-path only, so padded lanes read genuinely
        // empty rather than whatever the last fire left there.
        //
        // Ceiling = sum of every present class's rung, each capped at the
        // load's lane ceiling, then clamped to `max_lanes`.
        let mut qo_absolute: Vec<i32> = Vec::new();
        let mut lane_carve = composition.lane_count();
        if bodied {
            let ceiling = ladder.lane_reach(lane_ceiling).min(self.budget.max_lanes) as usize;
            lane_carve = lane_carve.max(ceiling as u32);
            for geometry in &mut geometries {
                geometry.pad_to(ceiling);
            }
            // Fire-wide row vector gets the same padding: entries past live
            // lanes repeat the last bound. Copied, not padded in place,
            // since the table's own vector is what every window's rebased
            // slice was cut from.
            qo_absolute = windows.qo_absolute_host().to_vec();
            kv::pad_indptr(&mut qo_absolute, ceiling);
            windows.stage_qo_absolute(ceiling as u32);
        }
        let geometries = geometries;

        super::btrace::mark("windows");
        // 5. Staging slot, claimed last (after every possible refusal above
        // has had its chance), host only. The slot's pinned bytes back the
        // async H2D `enqueue` issues, so nothing may reuse them until the
        // device has passed that copy.
        let slot = self.inputs.claim()?;
        let packing_fires: Vec<crate::inputs::PackingFire<'_>> = packings
            .iter()
            .map(|packed| crate::inputs::PackingFire {
                group_indptr: &packed.group_indptr,
                lane_indptr: &packed.lane_indptr,
                reference_start: &packed.reference_start,
                reference_tag: &packed.reference_tag,
                permutation: &packed.permutation,
            })
            .collect();
        let staged_lens = self.inputs.write_host(
            &slot,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                readout_rows: &readout_rows,
                // Padded to the bucket (step 4d); empty (no H2D) for an
                // unbodied fire.
                qo_absolute: &qo_absolute,
                // Staged only for a bodied fire; empty keeps the eager path
                // byte-identical to what it always was.
                live: if bodied { windows.live() } else { &[] },
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                // How far this fire's own rows go before the bucket's
                // padding starts.
                live_rows: rows,
                lane_reach: lane_carve,
                lane_of_row: &lane_of_row,
                group_of_lane: &group_of_lane,
                packings: &packing_fires,
            },
        )?;

        // Bound only when it would truncate something — see `RsFire::truncates`.
        // Both counted in the lane's extended layout `[replay | rows]`.
        let rs_replays: Vec<u32> = rs_moves
            .iter()
            .map(|verb| match verb {
                RsMove::Scatter { replay, .. } => *replay,
                _ => 0,
            })
            .collect();
        let rs_truncates = rs_lens
            .iter()
            .zip(&seats)
            .zip(&rs_replays)
            .any(|((len, seat), replay)| *len < narrow(u64::from(seat.rows) + u64::from(*replay)));
        // Split only when a boundary is strictly inside a row — see
        // `RsFire::splits`. `fold == rows` or `fold == 0` are both
        // single-call; only an interior boundary costs a second launch.
        let rs_splits = rs_moves.iter().zip(&seats).any(|(verb, seat)| {
            matches!(verb, RsMove::Scatter { fold, replay, .. } if *fold > 0 && *fold < seat.rows + *replay)
        });
        let rs_rows_ext = if rs_replays.iter().any(|replay| *replay > 0) {
            rows.saturating_add(rs_replays.iter().sum::<u32>())
        } else {
            0
        };
        Ok(Prepared {
            slot: Some(slot),
            lengths: staged_lens,
            token_injects,
            bodied,
            admits,
            ladder,
            lane_ceiling,
            patch_ladder,
            towered: self.towered,
            lanes,
            attachments,
            composition,
            readout_rows,
            readout_first,
            readout_count,
            descriptor,
            patch_payload,
            voxel_tables,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
            mrope_positions,
            self_cond_rows,
            self_cond_weights,
            self_cond_feeds,
            packings,
            port_feeds,
            voxel_feeds,
            merge_lands,
            lane_carve,
            windows,
            seats,
            tables,
            kv_less_seats,
            geometries,
            pages,
            fresh,
            demand,
            rs: RsFire {
                // A fire whose every lane folds and carries no prologue
                // attachment keeps the empty vectors and the two false
                // questions.
                write_state: rs_moves
                    .iter()
                    .any(|verb| !matches!(verb, RsMove::Scatter { fold: 0, .. })),
                predicated: {
                    let scatters = rs_moves
                        .iter()
                        .filter(|verb| matches!(verb, RsMove::Scatter { fold: 0, .. }))
                        .count();
                    let prologue = attachments
                        .iter()
                        .any(|attached| attached.at == Boundary::Prologue);
                    (scatters != 0 && scatters != rs_moves.len()) || prologue
                },
                // Bound only where it can truncate something — tidiness, not
                // a correctness rule.
                truncates: rs_truncates,
                splits: rs_splits,
                buffered: rs_moves.iter().any(|verb| !matches!(verb, RsMove::None)),
                moves: rs_moves,
                lens: rs_lens,
                order: rs_order,
                replays: rs_replays,
                rows_ext: rs_rows_ext,
            },
        })
    }

    /// Wraps [`Shell::enqueue_on`]: on success the slot moves on to
    /// `settle`, whose callback marks its pinned bytes free again. On
    /// failure there is no callback, so this synchronizes before releasing
    /// the slot — the one sync on this path, off the fast path by
    /// construction.
    fn enqueue<'a>(&mut self, prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let mut p = prepared;
        // Weight promotion between fires: copies ride the notify stream
        // behind an event on the compute stream, so no in-flight fire reads
        // a slab being replaced. Skipped during arming.
        if !self.arming {
            let (compute, notify) = (self.device.stream(), self.device.notify_stream());
            if let Some(tier) = self.weights.experts_mut() {
                tier.promote(compute, notify)?;
            }
        }
        // Slot leaves `Prepared` for this call, so a `?` inside cannot drop
        // it early.
        let slot = p
            .slot
            .take()
            .expect("a `Prepared` holds its staging slot until `enqueue` borrows it");
        match self.enqueue_on(&mut p, &slot) {
            Ok((launches, readback)) => {
                p.slot = Some(slot);
                Ok(Enqueued {
                    prepared: p,
                    launches,
                    readback,
                })
            }
            Err(fault) => {
                // This step's copies read the slot's pinned bytes and may
                // still be in flight with no callback coming, so this is
                // the wait that bounds them.
                let _ = self.device.synchronize();
                drop(slot);
                Err(fault)
            }
        }
    }

    /// The no-completion case of [`Shell::settle_step`].
    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Settled>
    where
        Self: 'a,
    {
        self.settle_step(enqueued, None)
    }
}

/// Resolves one lane's fold length: a host-stated length is itself; a
/// device-stated one reads the descriptor port's cell for this lane. Both
/// are clamped to `bound` and refuse zero.
fn resolve_fold_len(len: FoldLen, bound: u32, lane: usize, port: Option<&[u32]>) -> Result<u32> {
    let stated = match len {
        FoldLen::Host(n) => n,
        FoldLen::Device(which) => {
            let cells = port.ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {}, and the \
                         program attached to it resolved no such port",
                        which.name()
                    ),
                )
            })?;
            *cells.get(lane).or_else(|| cells.first()).ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {} whose \
                         cell carries {} entries",
                        which.name(),
                        cells.len()
                    ),
                )
            })?
        }
    };
    let folded = stated.min(bound);
    if folded == 0 {
        return Err(Fault::program(
            "serve::rs",
            format!(
                "lane {lane}'s fold length resolved to 0 against a bound of {bound}, which is \
                 not a dispatchable commit — a speculative commit must fold at least the \
                 bonus token it is guaranteed to accept"
            ),
        ));
    }
    Ok(folded)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::{FoldLen, resolve_fold_len};

    /// Port a device-resident fold length is read from; any consuming
    /// geometry port works, since the resolver only uses the cell.
    const PORT: eta_ir::registry::Port = eta_ir::registry::Port::RsFoldLen;

    /// A device-resolved count is bounded by what the host knows the buffer
    /// holds; a host-stated count is clamped by the same line.
    #[test]
    fn a_device_fold_length_is_clamped_to_the_bound_it_was_promised() {
        let cells = [3u32, 9, 5];
        let port = Some(&cells[..]);
        assert_eq!(
            resolve_fold_len(FoldLen::Device(PORT), 8, 0, port).unwrap(),
            3
        );
        assert_eq!(
            resolve_fold_len(FoldLen::Device(PORT), 8, 1, port).unwrap(),
            8
        );
        assert_eq!(resolve_fold_len(FoldLen::Host(9), 8, 0, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(4), 8, 0, None).unwrap(), 4);
    }

    /// A resolved fold of zero is not a dispatchable commit — refused by
    /// name in both spellings.
    #[test]
    fn a_fold_length_that_resolves_to_zero_is_refused_by_name() {
        let cells = [0u32];
        for len in [FoldLen::Device(PORT), FoldLen::Host(0)] {
            let error = resolve_fold_len(len, 8, 0, Some(&cells[..])).unwrap_err();
            let said = error.to_string();
            assert!(said.contains("bonus token"), "{said}");
        }
        // A bound of zero clamps to zero just as loudly.
        let error = resolve_fold_len(FoldLen::Host(4), 0, 0, None).unwrap_err();
        assert!(error.to_string().contains("bonus token"), "{error}");
    }
}
