//! Device pool for the buffered in-projection planes (`qkv`, `ba`) used by
//! `RsVerb::Buffer` / `RsVerb::FoldBuffered`; layout is `[layer][page_slot][plane][token]`, bf16.

use std::collections::HashMap;

use model_ir::{Attention, Dim, Dtype, Operands, Operation, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::store::kv::Paging;

/// Element type for staged in-projection planes; matches the state slabs (bf16).
pub const PLANE_DTYPE: Dtype = Dtype::Bf16;

/// Bytes of one buffered element.
const ELEMENT: u64 = 2;

/// One buffered plane: which layer's block of the pool it lives in, where
/// inside a page slot its rows begin, and how wide one token's row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Which recurrent layer's block of the pool, in plan order.
    pub layer: u32,
    /// Elements from the front of a page slot to this plane's first row.
    pub at: u64,
    /// Elements one token writes.
    pub width: u64,
}

/// Every buffered plane a plan declares, keyed by the operand that reads it.
#[derive(Debug, Clone, Default)]
pub struct Planes(HashMap<u32, Plane>);

impl Planes {
    /// The plane this operand is buffered as, or `None` for a value no
    /// chunked recurrent op reads.
    #[must_use]
    pub fn of(&self, id: ValueId) -> Option<Plane> {
        self.0.get(&id.0).copied()
    }

    /// How many planes were resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Does this plan buffer anything at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The pool and the plan's reading of it.
#[derive(Debug)]
pub struct Buffers {
    slab: Buffer,
    planes: Planes,
    /// Tokens one page slot holds — the kv page size (dev's rule).
    page_tokens: u32,
    /// Page slots per layer — the state-slot count, which is what the
    /// `Budget` already sized the recurrent banks by.
    slots: u32,
    /// Elements one page slot holds: `page_tokens × Σ plane widths`.
    per_page: u64,
    /// How many recurrent layers the plan declares.
    layers: u32,
    /// Bytes one EXTENDED row costs the read path (`crate::run::RsScratch`):
    /// the widest layer's six rectangles — the two buffered planes, the
    /// conv's output, the prepared gates, the scan's gate projection and its
    /// output — summed.
    ext_row_bytes: u64,
}

impl Buffers {
    /// Reserve the pool one plan needs at one deployment's budget, or `None`
    /// for a plan with no chunked recurrent layer to buffer.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a recurrent layer whose planes cannot be sized;
    /// [`Fault::Device`](crate::Fault::Device) for the allocation.
    pub fn reserve(trace: &Trace, paging: Paging) -> Result<Option<Buffers>> {
        let Some((planes, per_token, layers)) = read(trace, paging.page_size)? else {
            return Ok(None);
        };
        let per_page = u64::from(paging.page_size) * per_token;
        let ext_row_bytes = ext_row_bytes(trace)?;
        let bytes = per_page
            .saturating_mul(u64::from(paging.slots))
            .saturating_mul(u64::from(layers))
            .saturating_mul(ELEMENT);
        Ok(Some(Buffers {
            slab: Buffer::zeroed(usize::try_from(bytes).unwrap_or(usize::MAX))?,
            planes,
            page_tokens: paging.page_size,
            slots: paging.slots,
            per_page,
            layers,
            ext_row_bytes,
        }))
    }
}

/// Bytes one extended row of the read path costs, over the widest chunked
/// recurrent layer: the conv's input and output, the gate prep's input and
/// output, the scan's gate projection and output.
fn ext_row_bytes(trace: &Trace) -> Result<u64> {
    let mut defined: HashMap<u32, usize> = HashMap::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        for out in outs {
            defined.insert(out.0, at);
        }
    }
    let row_bytes = |id: ValueId| -> Result<u64> {
        let decl = trace.values.get(id.0 as usize);
        let Some(Ty::Tensor { shape, dtype }) = decl.map(|d| &d.ty) else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let Some(Dim::Const(width)) = shape.last() else {
            return Err(unsized_plane("an extended-run operand", id));
        };
        let elem = model_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
            what: format!("value {} has a dtype with no element size", id.0),
        })?;
        Ok(*width * elem)
    };
    let mut widest = 0u64;
    for node in &trace.nodes {
        let Operation::Attention(Attention::SsmGatedDeltaChunked { qkv, z, gates, y, .. }) = &node.op
        else {
            continue;
        };
        let mut total = row_bytes(*qkv)? + row_bytes(*z)? + row_bytes(*gates)? + row_bytes(*y)?;
        if let Some(Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. })) =
            defined.get(&qkv.0).map(|n| &trace.nodes[*n].op)
        {
            total += row_bytes(*x)?;
        }
        if let Some(Operation::Attention(Attention::SsmGdnPrep { ba, .. })) =
            defined.get(&gates.0).map(|n| &trace.nodes[*n].op)
        {
            total += row_bytes(*ba)?;
        }
        widest = widest.max(total);
    }
    Ok(widest)
}

/// Reads the plan's chunked recurrent layers with no allocation and no
/// device required; `None` if the plan has no chunked gated-delta scan.
///
/// # Errors
///
/// [`Fault::Unbound`] — see [`Buffers::reserve`].
pub fn read(trace: &Trace, page_tokens: u32) -> Result<Option<(Planes, u64, u32)>> {
    {
        let paging = page_tokens;
        // maps each value to the node that defines it
        let mut defined: HashMap<u32, usize> = HashMap::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            node.op.outputs(&mut outs);
            for out in outs {
                defined.insert(out.0, at);
            }
        }

        let mut planes: HashMap<u32, Plane> = HashMap::new();
        let mut per_token: Option<u64> = None;
        let mut layers = 0u32;
        // The decode class's chain — the step scan over the step conv and
        // the same gate prep — buffers the SAME planes as its layer's
        // chunked chain: a one-row `RsVerb::Buffer` lane scatters its row
        // there, and a one-row lane replaying buffered tokens ahead of it
        // (the read path) gathers from there. Layers are numbered in node
        // order on each chain, which is the model's layer order on both.
        let mut decode_layers = 0u32;
        for (at, node) in trace.nodes.iter().enumerate() {
            let (qkv, gates, chunked) = match &node.op {
                Operation::Attention(Attention::SsmGatedDeltaChunked { qkv, gates, .. }) => {
                    (*qkv, *gates, true)
                }
                Operation::Attention(Attention::SsmGatedDelta { qkv, gates, .. }) => {
                    (*qkv, *gates, false)
                }
                _ => continue,
            };
            let conv = defined.get(&qkv.0).copied().and_then(|n| {
                match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. }) if chunked => {
                        Some((n, *x))
                    }
                    Operation::Attention(Attention::SsmCausalConv1d { x, .. }) if !chunked => {
                        Some((n, *x))
                    }
                    _ => None,
                }
            });
            let prep = defined.get(&gates.0).copied().and_then(|n| {
                match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmGdnPrep { ba, .. }) => Some((n, *ba)),
                    _ => None,
                }
            });
            let (Some((conv_at, x)), Some((prep_at, ba))) = (conv, prep) else {
                if !chunked {
                    // A step scan fed some other way buffers nothing; only
                    // the chunked chain is a refusal, since it is what the
                    // verbs are defined against.
                    continue;
                }
                return Err(Fault::Unbound {
                    what: format!(
                        "the chunked recurrence at node {at}, whose `qkv` and `gates` are not \
                         written by a chunked conv and a gate prep — this shell buffers the \
                         two in-projection planes those ops read and knows no third shape"
                    ),
                });
            };
            // Writer nodes must precede the scan that names them.
            if conv_at >= at || prep_at >= at {
                return Err(Fault::Unbound {
                    what: format!(
                        "the recurrence at node {at} reads planes written at nodes \
                         {conv_at} and {prep_at}, which do not both stand before it"
                    ),
                });
            }
            let qkv_width = width_of(trace, x).ok_or_else(|| unsized_plane("the conv's rows", x))?;
            let ba_width = width_of(trace, ba).ok_or_else(|| unsized_plane("the gate prep's `[b | a]`", ba))?;
            let page = u64::from(paging);
            let layer = if chunked { layers } else { decode_layers };
            planes.insert(
                x.0,
                Plane {
                    layer,
                    at: 0,
                    width: qkv_width,
                },
            );
            planes.insert(
                ba.0,
                Plane {
                    layer,
                    at: page * qkv_width,
                    width: ba_width,
                },
            );
            // one width for every layer: the pool cuts a single page-slot
            // stride shared by all layers.
            let here = qkv_width + ba_width;
            match per_token {
                None => per_token = Some(here),
                Some(first) if first == here => {}
                Some(first) => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "recurrent layer {layer} buffers {here} elements a token where \
                             layer 0 buffers {first} — this pool cuts one page-slot stride \
                             for every layer"
                        ),
                    });
                }
            }
            if chunked {
                layers += 1;
            } else {
                decode_layers += 1;
            }
        }
        if decode_layers != 0 && decode_layers != layers {
            return Err(Fault::Unbound {
                what: format!(
                    "the plan runs {decode_layers} step recurrence(s) and {layers} chunked \
                     one(s); the buffered planes pair the two chains layer by layer"
                ),
            });
        }
        let Some(per_token) = per_token else {
            return Ok(None);
        };
        Ok(Some((Planes(planes), per_token, layers)))
    }
}

impl Buffers {
    /// The plan's reading: which operands are buffered planes.
    #[must_use]
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    /// Tokens one buffer page slot holds.
    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    /// How many page slots a lane's run may name.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// How many recurrent layers this pool holds a block for.
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Every byte this pool holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slab.bytes() as u64
    }

    /// Bytes one extended row of the read path costs (`crate::run::RsScratch::need`).
    #[must_use]
    pub fn ext_row_bytes(&self) -> u64 {
        self.ext_row_bytes
    }

    /// The device address of `plane`'s row `row` inside page slot
    /// `page_slot`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a page slot past the pool or a row past the
    /// page.
    pub fn row(&self, plane: Plane, page_slot: u32, row: u32) -> Result<u64> {
        if page_slot >= self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.slots),
            });
        }
        if row >= self.page_tokens {
            return Err(Fault::Ceiling {
                what: "rs buffer page tokens",
                need: u64::from(row) + 1,
                have: u64::from(self.page_tokens),
            });
        }
        let block = (u64::from(plane.layer) * u64::from(self.slots) + u64::from(page_slot))
            * self.per_page;
        Ok(self.slab.ptr() + (block + plane.at + u64::from(row) * plane.width) * ELEMENT)
    }
}

fn unsized_plane(what: &str, id: ValueId) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what} at value {}, whose row width is not a constant this shell can reserve a \
             buffered page for",
            id.0
        ),
    }
}

/// One value's declared row width, when it is a constant this shell can
/// reserve for.
fn width_of(trace: &Trace, id: ValueId) -> Option<u64> {
    let decl = trace.values.get(id.0 as usize)?;
    let Ty::Tensor { shape, .. } = &decl.ty else {
        return None;
    };
    match shape.last()? {
        Dim::Const(width) => Some(*width),
        _ => None,
    }
}

/// The fold predicate and the accepted lengths, as one resident region:
/// standing committed/uncommitted words, per-lane commit addresses, an
/// identity CSR (recurrent scans index by lane, not token/row), the mask,
/// and each request's accepted prefix length.
#[derive(Debug)]
pub struct Predicate {
    region: Buffer,
    at_one: u64,
    at_zero: u64,
    at_commits: u64,
    at_indptr: u64,
    at_mask: u64,
    at_len: u64,
    lanes: u32,
}

impl Predicate {
    /// Carve the region one deployment's lane ceiling needs, and write the
    /// two standing words and the identity CSR into it once.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) for the allocation or the two
    /// initial writes.
    pub fn reserve(max_lanes: u32) -> Result<Predicate> {
        let lanes = u64::from(max_lanes);
        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(256);
            here
        };
        let at_one = take(4);
        let at_zero = take(4);
        let at_commits = take(lanes * 8);
        let at_indptr = take((lanes + 1) * 4);
        let at_mask = take(lanes);
        let at_len = take(lanes * 4);
        let mut region = Buffer::zeroed(usize::try_from(at).unwrap_or(usize::MAX))?;
        region.write(at_one, &1u32.to_le_bytes())?;
        let identity: Vec<u8> = (0..=max_lanes)
            .flat_map(|l| (l as i32).to_le_bytes())
            .collect();
        region.write(at_indptr, &identity)?;
        Ok(Predicate {
            region,
            at_one,
            at_zero,
            at_commits,
            at_indptr,
            at_mask,
            at_len,
            lanes: max_lanes,
        })
    }

    /// The standing "this pass committed" word — what an unattached lane's
    /// commit slot points at.
    #[must_use]
    pub fn always(&self) -> u64 {
        self.region.ptr() + self.at_one
    }

    /// The standing "this pass did not commit" word — what a lane whose verb
    /// is a buffered scatter points at.
    #[must_use]
    pub fn never(&self) -> u64 {
        self.region.ptr() + self.at_zero
    }

    /// Stage this fire's commit addresses and accepted lengths on `stream`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] past the lane ceiling, [`Fault::Device`] for the
    /// copies.
    pub fn write(
        &mut self,
        stream: *mut core::ffi::c_void,
        commits: &[u64],
        lens: &[i32],
    ) -> Result<()> {
        if commits.len() as u64 > u64::from(self.lanes) {
            return Err(Fault::Ceiling {
                what: "rs fold predicates",
                need: commits.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let words: Vec<u8> = commits.iter().flat_map(|c| c.to_le_bytes()).collect();
        let at_commits = self.at_commits;
        let at_len = self.at_len;
        self.region.stage(stream, at_commits, &words)?;
        let lens: Vec<u8> = lens.iter().flat_map(|n| n.to_le_bytes()).collect();
        self.region.stage(stream, at_len, &lens)
    }

    /// Where the per-lane commit addresses live.
    #[must_use]
    pub fn commits(&self) -> u64 {
        self.region.ptr() + self.at_commits
    }

    /// Where the identity CSR lives.
    #[must_use]
    pub fn indptr(&self) -> u64 {
        self.region.ptr() + self.at_indptr
    }

    /// The fold predicate, as the handle a `RecurrentPool` seats.
    #[must_use]
    pub fn mask(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_mask, lanes, 1, Dtype::U8)
    }

    /// The accepted lengths, as the handle a `RecurrentPool` seats.
    #[must_use]
    pub fn commit_len(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_len, lanes, 1, Dtype::I32)
    }

    /// Reads the mask back for a gate that needs it. Not on the fire path:
    /// synchronous D2H.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the read.
    pub fn read_mask(&self, lanes: u32) -> Result<Vec<u8>> {
        let mut out = vec![0u8; lanes as usize];
        self.region.read(self.at_mask, &mut out)?;
        Ok(out)
    }
}

