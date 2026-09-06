use model_dsl::{Dtype, Weight};

pub use crate::adapter::Adapters;

/// Inkling's text decoder (`thinkingmachines/Inkling`, HF `inkling_mm_model`):
/// a sparse mixture-of-experts stack with NO rotary embedding — every score
/// carries a learned relative-position bias — a short causal convolution on
/// four seams of every layer, and two sink (shared) experts routed beside the
/// top-k. Only the text is declared here: the hMLP vision tower, the discrete
/// audio embedding and the eight chained MTP heads are separate bring-ups.
///
/// Per layer, in order:
///
/// ```text
/// x  = rmsnorm(y)
/// q  = x·Wq                       k = conv(x·Wk)     v = conv(x·Wv)     r = x·Wr
/// q  = rmsnorm_head(q)            k = rmsnorm_head(k)
/// b  = r ⊗ P                      -- [rows, heads · extent] bias over backward distance
/// a  = softmax(q·kᵀ / d + b)·v    -- 1/d, not 1/√d: q and k are unit-normed per head
/// y += conv(a·Wo)
/// x  = rmsnorm(y)
/// y += conv(mlp(x))               -- dense SwiGLU × scale, or the routed mixture
/// ```
///
/// `conv` is the depthwise causal width-4 convolution with the input added
/// back (`Attention::ShortConv`), kept per slot as the recurrent mixers keep
/// theirs. Local layers (five of six) attend a 512-token window over 16 kv
/// heads with a 512-deep bias; global layers attend everything over 8 kv
/// heads with a 1024-deep one.
pub struct Model {
    pub hidden: u32,
    /// The embedding table's rows (padded).
    pub vocab: u32,
    /// The head's rows: the unpadded vocabulary the readout is cut to.
    pub head_rows: u32,
    pub tp: u32,

    pub heads: u32,
    pub head_dim: u32,
    /// Per-head width of the relative projection `Wr`.
    pub d_rel: u32,
    /// The local reading's window (and its bias extent).
    pub window: u32,
    /// The short convolution's taps.
    pub conv_width: u32,
    /// `1 / head_dim`.
    pub sm_scale: f32,
    pub norm_eps: f32,
    /// `1 / logits_mup_width_multiplier`, over the final norm's output.
    pub head_scale: f32,
    /// The global reading's log attention scaling: `(log_scaling_n_floor,
    /// log_scaling_alpha)`, past which a query's scores grow as
    /// `1 + alpha · ln(n / floor)`. Local layers have none.
    pub log_scaling: (u32, f32),

    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub embed_norm: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    /// `unembed`, `[head_rows, hidden]`; never tied.
    pub unembed: Weight,
}

/// Which reading of the sequence a layer takes; the discriminant indexes
/// the per-reading plan arrays.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Local = 0,
    Global = 1,
}

pub struct Layer {
    pub reading: Reading,
    pub kv_heads: u32,
    /// The bias profile's depth: how many backward distances carry a
    /// learned bias (512 local, 1024 global).
    pub extent: u32,

    pub attn_norm: Weight,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    /// `[heads · d_rel, hidden]`.
    pub r_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    /// `[d_rel, extent]`: the bank of bias-vs-distance profiles.
    pub rel_proj: Weight,
    /// The four short convolutions, `[channels, conv_width]` each.
    pub k_conv: Weight,
    pub v_conv: Weight,
    pub attn_conv: Weight,
    pub mlp_conv: Weight,
    pub kv: String,
    pub k_state: String,
    pub v_state: String,
    pub attn_state: String,
    pub mlp_state: String,

    pub mlp_norm: Weight,
    pub mlp: Mlp,

    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub enum Mlp {
    /// The two dense foundation layers: SwiGLU, then a learned `[1]` scale.
    Dense {
        /// `[2 · inter, hidden]`, gate first.
        gate_up: Weight,
        inter: u32,
        down: Weight,
        scale: Weight,
    },
    /// The routed mixture. The banks stack the `sink` shared experts after
    /// the `experts` routed ones, so the router's fixed sink routes select
    /// them like any other expert.
    Routed {
        /// `gate.weight`, `[experts + sink, hidden]`.
        router: Weight,
        /// `gate.bias`, `[experts]` f32: steers the choice only.
        bias: Weight,
        /// `gate.global_scale`, `[1]` f32, over every weight.
        scale: Weight,
        /// `[experts + sink, 2 · inter, hidden]`, gate first.
        gate_up: Weight,
        /// `[experts + sink, hidden, inter]`.
        down: Weight,
        experts: u32,
        top_k: u32,
        sink: u32,
        inter: u32,
        /// `route_scale`.
        scaling: f32,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    vocab: u32,
    head_rows: u32,
    heads: u32,
    head_dim: u32,
    local_kv_heads: u32,
    global_kv_heads: u32,
    d_rel: u32,
    window: u32,
    global_extent: u32,
    conv_width: u32,
    /// Every sixth layer is global: `(l + 1) % global_every == 0`.
    global_every: u32,
    /// The leading dense layers.
    dense_layers: u32,
    dense_inter: u32,
    experts: u32,
    top_k: u32,
    sink: u32,
    moe_inter: u32,
    route_scale: f32,
    mup: f32,
    norm_eps: f32,
    log_floor: u32,
    log_alpha: f32,
}

impl Model {
    /// `thinkingmachines/Inkling`'s text, off its `text_config`.
    pub fn full(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::dims())
    }

    /// The text cut to its first `layers` layers over `experts` routed
    /// experts — the miniature `benches/shrink_checkpoint.py` carves
    /// (`--layers 0-6 --experts 8`: both dense layers, four local sparse
    /// ones and the first global).
    pub fn mini(layers: u32, experts: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::dims();
        d.layers = layers;
        d.experts = experts;
        d.top_k = d.top_k.min(experts);
        Model::new(w, kv, tp, d)
    }

    fn dims() -> Dims {
        Dims {
            hidden: 6144,
            layers: 66,
            vocab: 201_024,
            head_rows: 200_058,
            heads: 64,
            head_dim: 128,
            local_kv_heads: 16,
            global_kv_heads: 8,
            d_rel: 16,
            window: 512,
            global_extent: 1024,
            conv_width: 4,
            global_every: 6,
            dense_layers: 2,
            dense_inter: 24_576,
            experts: 256,
            top_k: 6,
            sink: 2,
            moe_inter: 3072,
            route_scale: 8.0,
            mup: 24.0,
            norm_eps: 1e-6,
            log_floor: 128_000,
            log_alpha: 0.1,
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let dense = crate::dense(w);
        // The 2-D projections take the tiled 4-bit placement on CUDA (see
        // `muse_glimmer::model`); the expert banks and the embedding do not.
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
        let heads = d.heads / tp;
        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(heads) * hd;
        let r_w = u64::from(heads) * u64::from(d.d_rel);
        let kw = u64::from(d.conv_width);

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let vec = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let global = (l + 1) % d.global_every == 0;
                let (reading, kv_heads, extent) = if global {
                    (Reading::Global, d.global_kv_heads / tp, d.global_extent)
                } else {
                    (Reading::Local, d.local_kv_heads / tp, d.window)
                };
                let kv_w = u64::from(kv_heads) * hd;
                // A conv over a columns-cut channel axis is cut with it.
                let conv = |s: &str, channels: u64| {
                    Weight::sym(n(s), [channels, kw], dense).columns()
                };
                let mlp = if l < d.dense_layers {
                    let iw = u64::from(d.dense_inter / tp);
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
                        inter: d.dense_inter / tp,
                        down: Weight::sym(n("down"), [hidden, iw], proj).rows(),
                        scale: vec("mlp_scale", 1),
                    }
                } else {
                    let bank = u64::from(d.experts + d.sink);
                    let mi = u64::from(d.moe_inter / tp);
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [bank, hidden], proj),
                        bias: Weight::sym(n("router_bias"), [u64::from(d.experts)], Dtype::F32),
                        scale: Weight::sym(n("router_scale"), [1], Dtype::F32),
                        gate_up: Weight::sym(n("experts_gate_up"), [bank, 2 * mi, hidden], w)
                            .bank([mi, mi]),
                        down: Weight::sym(n("experts_down"), [bank, hidden, mi], w).rows(),
                        experts: d.experts,
                        top_k: d.top_k,
                        sink: d.sink,
                        inter: d.moe_inter / tp,
                        scaling: d.route_scale,
                    }
                };
                Layer {
                    reading,
                    kv_heads,
                    extent,
                    attn_norm: vec("attn_norm", hidden),
                    q_proj: Weight::sym(n("q_proj"), [q_w, hidden], proj).columns(),
                    k_proj: Weight::sym(n("k_proj"), [kv_w, hidden], proj).columns(),
                    v_proj: Weight::sym(n("v_proj"), [kv_w, hidden], proj).columns(),
                    r_proj: Weight::sym(n("r_proj"), [r_w, hidden], proj).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], proj).rows(),
                    q_norm: vec("q_norm", hd),
                    k_norm: vec("k_norm", hd),
                    rel_proj: Weight::sym(
                        n("rel_proj"),
                        [u64::from(d.d_rel), u64::from(extent)],
                        dense,
                    ),
                    k_conv: conv("k_conv", kv_w),
                    v_conv: conv("v_conv", kv_w),
                    attn_conv: Weight::sym(n("attn_conv"), [hidden, kw], dense),
                    mlp_conv: Weight::sym(n("mlp_conv"), [hidden, kw], dense),
                    kv: format!("kv.{l}"),
                    k_state: format!("conv.{l}.k"),
                    v_state: format!("conv.{l}.v"),
                    attn_state: format!("conv.{l}.attn"),
                    mlp_state: format!("conv.{l}.mlp"),
                    mlp_norm: vec("mlp_norm", hidden),
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            head_rows: d.head_rows,
            tp,
            heads,
            head_dim: d.head_dim,
            d_rel: d.d_rel,
            window: d.window,
            conv_width: d.conv_width,
            sm_scale: 1.0 / d.head_dim as f32,
            norm_eps: d.norm_eps,
            head_scale: 1.0 / d.mup,
            log_scaling: (d.log_floor, d.log_alpha),
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [u64::from(d.vocab), hidden], w),
            embed_norm: Weight::sym("embed_norm", [hidden], dense),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            unembed: Weight::sym("unembed", [u64::from(d.head_rows), hidden], proj),
        }
    }
}

/// What every SKU seats. A deployment ceiling, not a checkpoint fact.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };
