//! The DFlash block drafter, in its three shapes: v1 (DFlash), v2 (DFlash2:
//! dynamic convolutions and a candidate selector), and DSpark (v1's backbone
//! with a markov readout). Model text any family can carry: a head's own
//! numbers arrive as a [`Head`], the trunk's as a [`Trunk`].

use checkpoint::contract::{Expr, TensorType};
use checkpoint_dsl::{Builder, Error, extents};
use model_dsl::{
    BlockDrafter, Dtype, HybridSpec, Input, KvSpace, Predicate, Value, Weight, ops, seam,
};

/// The trunk numbers the drafter's banks are shaped by.
#[derive(Clone, Copy, Debug)]
pub struct Trunk {
    pub hidden: u64,
    pub vocab: u64,
    pub norm_eps: f32,
    /// Element type for the drafter's projections; a drafter is quantized like the trunk.
    pub weights: Dtype,
    /// Element type of norms and codebooks.
    pub dense: Dtype,
    pub tp: u32,
}

/// A DFlash block drafter: its own decoder stack fed by the trunk, running a
/// whole block of rows in one pass instead of one token at a time.
pub struct DFlash {
    /// Trunk layers feeding the fusion, in the order their banks are sliced out of `fc`.
    pub taps: Vec<u32>,
    /// Per-tap `[hidden, hidden]` slices of the stored `[hidden, taps·hidden]`
    /// fc bank, summed with residual_add (the IR has no concat).
    pub fc: Vec<Weight>,
    /// Scales the fused stream before the first block.
    pub hidden_norm: Weight,
    pub hidden_norm_eps: f32,
    pub blocks: Vec<DFlashBlock>,
    /// Final norm before the readout through the target's `lm_head`.
    pub norm: Weight,
    pub norm_eps: f32,
    /// Rows one draft pass proposes; the width of the `mtp.drafts` seam.
    pub block: u32,
    /// Token every block row but the first carries on the way in.
    pub mask_token: u32,
    /// DFlash2's candidate selector; `None` on a v1 head (readout is per-slot argmax).
    pub selector: Option<Selector>,
    /// First proposing block row: 1 for DFlash (anchor proposes nothing), 0 for DSpark.
    pub proposals_from: u32,
    /// The published head this was declared from.
    pub head: &'static Head,
}

/// DFlash2's candidate selector readout: each mask slot keeps its
/// `top_k` candidates and a path through them is walked from the anchor.
pub struct Selector {
    /// `[rank, hidden]`; `None` for a plain bigram lattice (DSpark) with no hidden term.
    pub hidden_projection: Option<Weight>,
    /// `[vocab, rank]`, indexed by the predecessor's id.
    pub pred: Weight,
    /// `[vocab, rank]`, indexed by the candidate's id.
    pub succ: Weight,
    pub top_k: u32,
}

/// One layer of a [`DFlash`] drafter: a pre-norm decoder block with its own
/// ungated attention and a sliding window on all but the last.
pub struct DFlashBlock {
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub attn: DraftAttn,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: DraftMlp,
    /// Attention window; `None` on the one full-attention layer a v1 head ends with (v2 has none).
    pub window: Option<u32>,
    /// DFlash2's dynamic convolution around the attention sublayer; `None` on a v1 head.
    pub attn_conv: Option<DynConv>,
    /// Same, around the MLP sublayer.
    pub mlp_conv: Option<DynConv>,
}

/// A draft block's gated MLP, routing to no experts.
pub struct DraftMlp {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

/// DFlash2's two-tap grouped dynamic convolution around one sublayer:
/// coefficients from the normed input convolve it before and its output after.
pub struct DynConv {
    /// `[2·taps, hidden]`, row `side·taps + tap`: the stored
    /// `base_kernel [2, taps, hidden]` read flat.
    pub base: Weight,
    /// `[2·taps·groups, hidden]`: `kernel_projection.weight`.
    pub proj: Weight,
    pub taps: u32,
    /// Channels sharing one correction.
    pub group: u32,
}

/// A [`DFlash`] layer's plain (ungated) attention, with per-head q/k norms.
pub struct DraftAttn {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    /// Projection biases, where the head has them (`attn_bias` heads carry q/k/v/o biases).
    pub q_bias: Option<Weight>,
    pub k_bias: Option<Weight>,
    pub v_bias: Option<Weight>,
    pub o_bias: Option<Weight>,
    /// This layer's kv row, in the trunk's page-id space.
    pub kv: String,
}

/// A published head's own numbers (its `config.json`); stated by the family
/// that carries it, never here.
#[derive(Debug, PartialEq)]
pub struct Head {
    /// Trunk layers whose hidden states feed the fusion, in tap order (`dflash_config.target_layer_ids`).
    pub taps: &'static [u32],
    /// Per layer: a sliding window, or `None` for full attention, bidirectional over the block.
    pub windows: &'static [Option<u32>],
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub theta: f32,
    /// Rows one draft pass proposes (`block_size`); the width of the `mtp.drafts` seam.
    pub block: u32,
    /// `dflash_config.mask_token_id`.
    pub mask_token: u32,
    /// First proposing block row (DSpark's `logits_start`): 1 or 0.
    pub proposals_from: u32,
    /// The dynamic convolution around every sublayer, where the head has one.
    pub conv: Option<Conv>,
    pub readout: Readout,
    /// Whether the head's attention projections carry biases (`attention_bias`).
    pub attn_bias: bool,
}

/// A [`DynConv`]'s taps and channel-group size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv {
    pub taps: u32,
    pub group: u32,
}

/// How a head reads its proposals off the shared `lm_head`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// The per-slot argmax.
    Argmax,
    /// DFlash2's candidate selector over the slot's `top_k`.
    Selector { rank: u32, top_k: u32 },
    /// DSpark's markov bigram head; biases only the slot's top-`K`, lossless because the verify decides.
    Markov { rank: u32, top_k: u32 },
}

impl DFlash {
    /// Declare the head's planes under `prefix` (`aux` for an `--aux`
    /// overlay), shaped by the trunk's hidden width and element types.
    #[must_use]
    pub fn declare(head: &'static Head, prefix: &str, trunk: &Trunk) -> DFlash {
        let (hidden, w, dense, tp) = (trunk.hidden, trunk.weights, trunk.dense, trunk.tp);
        let n = |s: &str| format!("{prefix}.{s}");
        let conv = |l: u32, which: &str| {
            head.conv.map(|c| DynConv {
                base: Weight::sym(
                    format!("{prefix}.layers.{l}.{which}.base_kernel"),
                    [2 * u64::from(c.taps), hidden],
                    dense,
                ),
                proj: Weight::sym(
                    format!("{prefix}.layers.{l}.{which}.kernel_projection"),
                    [2 * u64::from(c.taps) * (hidden / u64::from(c.group)), hidden],
                    w,
                )
                .columns(),
                taps: c.taps,
                group: c.group,
            })
        };
        let (dq, dkv, dhd) = (head.q_heads / tp, head.kv_heads / tp, head.head_dim);
        let hd = u64::from(dhd);
        let inter = head.inter / tp;
        let codebook = |s: &str, rank: u32| Weight::sym(n(s), [trunk.vocab, u64::from(rank)], dense);
        DFlash {
            taps: head.taps.to_vec(),
            fc: (0..head.taps.len())
                .map(|i| Weight::sym(n(&format!("fc_tap{i}")), [hidden, hidden], w))
                .collect(),
            hidden_norm: Weight::sym(n("hidden_norm"), [hidden], dense),
            hidden_norm_eps: trunk.norm_eps,
            blocks: head
                .windows
                .iter()
                .zip(0u32..)
                .map(|(&window, l)| {
                    let b = |s: &str| format!("{prefix}.layers.{l}.{s}");
                    DFlashBlock {
                        mixer_norm: Weight::sym(b("mixer_norm"), [hidden], dense),
                        mixer_norm_eps: trunk.norm_eps,
                        attn: DraftAttn {
                            q_heads: dq,
                            kv_heads: dkv,
                            head_dim: dhd,
                            rotary_dim: dhd,
                            theta: head.theta,
                            sm_scale: (dhd as f32).sqrt().recip(),
                            q_proj: Weight::sym(b("q_proj"), [u64::from(dq) * hd, hidden], w).columns(),
                            k_proj: Weight::sym(b("k_proj"), [u64::from(dkv) * hd, hidden], w).columns(),
                            v_proj: Weight::sym(b("v_proj"), [u64::from(dkv) * hd, hidden], w).columns(),
                            o_proj: Weight::sym(b("o_proj"), [hidden, u64::from(dq) * hd], w).rows(),
                            q_norm: Weight::sym(b("q_norm"), [hd], dense),
                            q_norm_eps: trunk.norm_eps,
                            k_norm: Weight::sym(b("k_norm"), [hd], dense),
                            k_norm_eps: trunk.norm_eps,
                            q_bias: head.attn_bias.then(|| Weight::sym(b("q_bias"), [u64::from(dq) * hd], dense)),
                            k_bias: head.attn_bias.then(|| Weight::sym(b("k_bias"), [u64::from(dkv) * hd], dense)),
                            v_bias: head.attn_bias.then(|| Weight::sym(b("v_bias"), [u64::from(dkv) * hd], dense)),
                            o_bias: head.attn_bias.then(|| Weight::sym(b("o_bias"), [hidden], dense)),
                            kv: format!("kv.dflash.{l}"),
                        },
                        mlp_norm: Weight::sym(b("mlp_norm"), [hidden], dense),
                        mlp_norm_eps: trunk.norm_eps,
                        mlp: DraftMlp {
                            gate_up: Weight::sym(b("gate_up"), [2 * u64::from(inter), hidden], w)
                                .packed([u64::from(inter), u64::from(inter)]),
                            down: Weight::sym(b("down"), [hidden, u64::from(inter)], w).rows(),
                            inter,
                        },
                        window,
                        attn_conv: conv(l, "attention_conv"),
                        mlp_conv: conv(l, "mlp_conv"),
                    }
                })
                .collect(),
            norm: Weight::sym(n("norm"), [hidden], dense),
            norm_eps: trunk.norm_eps,
            block: head.block,
            mask_token: head.mask_token,
            proposals_from: head.proposals_from,
            selector: match head.readout {
                Readout::Argmax => None,
                Readout::Selector { rank, top_k } => Some(Selector {
                    hidden_projection: Some(
                        Weight::sym(
                            n("candidate_selector.hidden_projection"),
                            [u64::from(rank), hidden],
                            w,
                        )
                        .columns(),
                    ),
                    pred: codebook("candidate_selector.predecessor_codebook", rank),
                    succ: codebook("candidate_selector.successor_codebook", rank),
                    top_k,
                }),
                Readout::Markov { rank, top_k } => Some(Selector {
                    hidden_projection: None,
                    pred: codebook("markov_w1", rank),
                    succ: codebook("markov_w2", rank),
                    top_k,
                }),
            },
            head,
        }
    }

    /// Register the drafter's kv rows in `space`, the trunk's page-id space.
    pub fn declare_caches(&self, c: &mut HybridSpec, space: KvSpace) {
        for b in &self.blocks {
            let a = &b.attn;
            let plane = u64::from(a.kv_heads) * u64::from(a.head_dim);
            c.kv(space, a.attn_kv(), [plane, plane]);
        }
    }

    /// Call after every trunk layer `layer` with its residual stream `y`; at
    /// a tapped layer folds `y · fcᵢ` into `fused`. The residual stream is one
    /// aliased buffer, so a tap must fuse the hidden state as it passes.
    pub fn tap(&self, layer: u32, y: &Value, fused: &mut Option<Value>) {
        let Some(at) = self.taps.iter().position(|t| *t == layer) else {
            return;
        };
        let part = ops::linear::matmul(y, &self.fc[at]);
        *fused = Some(match fused.take() {
            Some(sum) => ops::elemwise::residual_add(&part, &sum),
            None => part,
        });
    }

    /// The drafter's two arms: writes the context into its kv rows, then runs
    /// the block. `fused` is what [`tap`](DFlash::tap) accumulated; `mask` is
    /// the guest's mask channel, read by v1's full layer alone. Returns the
    /// block rows' final hidden for the caller to merge before the readout.
    pub fn arm<F>(
        &self,
        inputs: &Input<F>,
        fused: &Value,
        h_block: &Value,
        mask: &Value,
        block_draft: &Predicate,
    ) -> Value {
        let d = self;
        // written on every trunk fire, not only a drafting one
        let h_ctx = ops::elemwise::rmsnorm_plus_one(fused, &d.hidden_norm, d.hidden_norm_eps);
        let (_, ctx_positions) = inputs.positions().split(block_draft);
        for b in &d.blocks {
            let a = &b.attn;
            let hd = a.head_dim;
            let k = biased(ops::linear::matmul(&h_ctx, &a.k_proj), a.k_bias.as_ref());
            let v = biased(ops::linear::matmul(&h_ctx, &a.v_proj), a.v_bias.as_ref());
            let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, hd, a.k_norm_eps);
            let k = ops::elemwise::rope_partial_q(&k, &ctx_positions, a.rotary_dim, hd, a.theta);
            ops::attn::kv_append(
                &k,
                &v,
                inputs.kv(&a.kv),
                &inputs.write_page(&a.kv),
                &inputs.write_offset(&a.kv),
            );
        }

        // The block.
        let (input_block, _) = inputs.split(block_draft);
        let (block_positions, _) = inputs.positions().split(block_draft);
        let mut h = h_block.clone();
        for b in &d.blocks {
            let a = &b.attn;
            let hd = a.head_dim;
            let plan = ops::attn::plan_prefill(&input_block, a.q_heads, a.kv_heads, hd, None);
            let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mixer_norm, b.mixer_norm_eps);
            let (x, attn_coeff) = conv_prepare(&x, b.attn_conv.as_ref());
            let q = biased(ops::linear::matmul(&x, &a.q_proj), a.q_bias.as_ref());
            let k = biased(ops::linear::matmul(&x, &a.k_proj), a.k_bias.as_ref());
            let v = biased(ops::linear::matmul(&x, &a.v_proj), a.v_bias.as_ref());
            let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, hd, a.q_norm_eps);
            let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, hd, a.k_norm_eps);
            let (q, k) =
                ops::elemwise::rope_partial(&q, &k, &block_positions, a.rotary_dim, hd, a.theta);
            ops::attn::kv_append(
                &k,
                &v,
                inputs.kv(&a.kv),
                &inputs.write_page(&a.kv),
                &inputs.write_offset(&a.kv),
            );
            // v1's full-attention layer is bidirectional over the block; the mask is the guest's.
            let o = match b.window {
                Some(w) => ops::attn::prefill(
                    &q,
                    &plan,
                    inputs.kv(&a.kv),
                    Some(w),
                    hd,
                    a.kv_heads,
                    a.sm_scale,
                ),
                None => ops::attn::masked(
                    &q,
                    &plan,
                    mask,
                    inputs.kv(&a.kv),
                    None,
                    hd,
                    a.kv_heads,
                    false,
                    a.sm_scale,
                ),
            };
            let o = biased(ops::linear::matmul(&o, &a.o_proj), a.o_bias.as_ref());
            let o = conv_finish(&o, b.attn_conv.as_ref(), attn_coeff.as_ref());
            h = ops::elemwise::residual_add(&o, &h);

            let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mlp_norm, b.mlp_norm_eps);
            let (x, mlp_coeff) = conv_prepare(&x, b.mlp_conv.as_ref());
            let f = ops::linear::matmul(
                &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &b.mlp.gate_up), b.mlp.inter),
                &b.mlp.down,
            );
            let f = conv_finish(&f, b.mlp_conv.as_ref(), mlp_coeff.as_ref());
            h = ops::elemwise::residual_add(&f, &h);
        }
        ops::elemwise::rmsnorm_plus_one(&h, &d.norm, d.norm_eps)
    }

    /// Plant proposals at the `mtp.drafts` seam: a v1 head's per-slot argmax,
    /// or DFlash2's selector walk. `hb` is the hidden [`arm`](DFlash::arm) returned.
    pub fn plant_readout<F>(
        &self,
        logits: &Value,
        inputs: &Input<F>,
        hb: Option<&Value>,
        block_draft: &Predicate,
    ) {
        logits.rec().block_drafter(BlockDrafter {
            rows: self.block,
            mask_token: self.mask_token,
            bidirectional: self.blocks.iter().any(|b| b.window.is_none()),
            proposals_from: self.proposals_from,
        });
        let (dlogits, _) = logits.split(block_draft);
        seam::at(seam::MTP, &[&dlogits]);
        let picks = match (&self.selector, hb) {
            (Some(sel), Some(hb)) => {
                let (unary, cand) = ops::layout::topk(&dlogits, sel.top_k);
                let hp = sel
                    .hidden_projection
                    .as_ref()
                    .map(|proj| ops::linear::matmul(hb, proj));
                let (toks, _) = inputs.tokens().split(block_draft);
                ops::attn::selector_walk(
                    &cand,
                    &unary,
                    hp.as_ref(),
                    &toks,
                    &sel.pred,
                    &sel.succ,
                    self.proposals_from,
                )
            }
            _ => ops::layout::argmax(&[&dlogits]),
        };
        seam::at(seam::MTP_DRAFTS, &[&picks]);
    }

    /// Bind the head's `--aux` planes; `norm` reads an RMSNorm weight per the family's rule.
    ///
    /// # Errors
    ///
    /// Whatever the builder refuses.
    pub fn bind_aux(
        &self,
        b: &mut Builder,
        src: &ztensor::Source,
        norm: &dyn Fn(String) -> Expr,
    ) -> Result<(), Error> {
        b.read_expr(&self.hidden_norm, norm("aux.hidden_norm.weight".to_string()))?;
        // `fc.weight` is one `[hidden, taps·hidden]` bank; tap `i` is columns `i·hidden .. (i+1)·hidden`.
        let span = extents(&self.fc[0])[1];
        for (i, bank) in self.fc.iter().enumerate() {
            let at = span * i as i64;
            b.read_expr(bank, Expr::src("aux.fc.weight".to_string()).slice(1, at, span))?;
        }
        for (l, block) in self.blocks.iter().enumerate() {
            let n = |s: &str| format!("aux.layers.{l}.{s}");
            let a = &block.attn;
            b.read_expr(&block.mixer_norm, norm(n("input_layernorm.weight")))?;
            b.read(&a.q_proj, n("self_attn.q_proj.weight"))?;
            b.read(&a.k_proj, n("self_attn.k_proj.weight"))?;
            b.read(&a.v_proj, n("self_attn.v_proj.weight"))?;
            b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
            for (bias, leaf) in [
                (&a.q_bias, "q_proj"),
                (&a.k_bias, "k_proj"),
                (&a.v_bias, "v_proj"),
                (&a.o_bias, "o_proj"),
            ] {
                if let Some(bias) = bias {
                    b.read(bias, n(&format!("self_attn.{leaf}.bias")))?;
                }
            }
            b.read_expr(&a.q_norm, norm(n("self_attn.q_norm.weight")))?;
            b.read_expr(&a.k_norm, norm(n("self_attn.k_norm.weight")))?;
            b.read_expr(&block.mlp_norm, norm(n("post_attention_layernorm.weight")))?;
            for (conv, which) in [(&block.attn_conv, "attention_conv"), (&block.mlp_conv, "mlp_conv")] {
                if let Some(c) = conv {
                    let want: Vec<i64> = extents(&c.base);
                    b.read_expr(&c.base, flat(src, n(&format!("{which}.base_kernel")), want)?)?;
                    b.read(&c.proj, n(&format!("{which}.kernel_projection.weight")))?;
                }
            }
            b.read_concat(
                &block.mlp.gate_up,
                [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            )?;
            b.read(&block.mlp.down, n("mlp.down_proj.weight"))?;
        }
        b.read_expr(&self.norm, norm("aux.norm.weight".to_string()))?;
        match (&self.selector, self.head.readout) {
            (Some(sel), Readout::Selector { .. }) => {
                if let Some(proj) = &sel.hidden_projection {
                    b.read(proj, "aux.candidate_selector.hidden_projection.weight".to_string())?;
                }
                b.read(&sel.pred, "aux.candidate_selector.predecessor_codebook".to_string())?;
                b.read(&sel.succ, "aux.candidate_selector.successor_codebook".to_string())?;
            }
            (Some(sel), Readout::Markov { .. }) => {
                b.read(&sel.pred, "aux.markov_head.markov_w1.weight".to_string())?;
                b.read(&sel.succ, "aux.markov_head.markov_w2.weight".to_string())?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl DraftAttn {
    /// The kv row's name, owned, for a cache declaration.
    fn attn_kv(&self) -> String {
        self.kv.clone()
    }
}

/// A projection's bias, where the head has one; passes `x` through otherwise.
fn biased(x: Value, bias: Option<&Weight>) -> Value {
    match bias {
        Some(b) => ops::elemwise::add_bias(b, &x),
        None => x,
    }
}

/// The input side of a [`DynConv`]: convolve the normed input with side 0;
/// the coefficients come back for the output side. Passes `x` through where there is no conv.
fn conv_prepare(x: &Value, conv: Option<&DynConv>) -> (Value, Option<Value>) {
    match conv {
        Some(c) => {
            let coeff = ops::linear::matmul(x, &c.proj);
            let x = ops::attn::block_dyn_conv(x, &coeff, &c.base, 0, c.taps, c.group);
            (x, Some(coeff))
        }
        None => (x.clone(), None),
    }
}

/// The output side of a [`DynConv`], with the coefficients its input side projected.
fn conv_finish(y: &Value, conv: Option<&DynConv>, coeff: Option<&Value>) -> Value {
    match (conv, coeff) {
        (Some(c), Some(coeff)) => ops::attn::block_dyn_conv(y, coeff, &c.base, 1, c.taps, c.group),
        _ => y.clone(),
    }
}

/// The same bytes read as `want`: a stored rank-N kernel as a rank-2 bank.
fn flat(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let stored: i128 = shape.iter().map(|&n| i128::from(n)).product();
    let asked: i128 = want.iter().map(|&n| i128::from(n)).product();
    if stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as {want:?} \
             ({asked} elements)"
        )));
    }
    let stored = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, stored)))
}
