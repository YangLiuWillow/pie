use model_dsl::{Dtype, Weight};

pub use crate::adapter::Adapters;

/// Muse Glimmer's text decoder (`muse_glimmer_text`): a dense Gemma-3-shaped
/// stack — four `(1 + w)` norms a layer around a gated attention and a
/// SwiGLU MLP — read out through its own head under a tanh softcap.
///
/// What is this family's own, against gemma's:
///
/// * the attention output is gated: `o = attn(x) * sigmoid(gate_proj(x))`
///   before `o_proj` (Qwen3-Next's gate, on a rotary stack);
/// * the q/k norms carry no scale, and q is multiplied by `qk_scale_factor`
///   after its norm — folded here into [`Model::sm_scale`], since a rotation
///   commutes with a scalar;
/// * every fourth layer attends over the whole sequence with NO rotation
///   (`layer_rope_theta` is 0 there); the other three rotate at
///   `rope_theta` over a 2048-token window;
/// * the token embedding passes through a scale-free RMSNorm, and the
///   readout is `softcap · tanh(logits · output_multiplier / softcap)`.
pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// The sliding reading's window; the full reading has none.
    pub window: u32,
    /// The sliding reading's rope base; the full reading rotates nothing.
    pub theta: f32,
    /// `qk_scale_factor · head_dim^-0.5`.
    pub sm_scale: f32,
    /// `rms_norm_eps`: the q/k norms', the embedding norm's, and the two
    /// pre-norms'.
    pub norm_eps: f32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub softcap: f32,
    pub output_multiplier: f32,
    pub embed: Weight,
    /// `lm_head.weight`, `[vocab, hidden]`; never tied.
    pub lm_head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

/// Which of the text's two readings of the one sequence a layer takes. The
/// discriminant is the index into the per-reading plan arrays.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Sliding = 0,
    Full = 1,
}

pub struct Layer {
    pub reading: Reading,
    /// `[q | k | v]`, `[(q_heads + 2 kv_heads) · head_dim, hidden]`.
    pub qkv: Weight,
    /// `self_attn.gate_proj`, `[q_heads · head_dim, hidden]`: the sigmoid
    /// gate over the attention output.
    pub gate: Weight,
    pub o_proj: Weight,
    pub kv: String,

    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub post_attn_norm: Weight,
    pub post_attn_norm_eps: f32,
    pub pre_ffw_norm: Weight,
    pub pre_ffw_norm_eps: f32,
    pub post_ffw_norm: Weight,
    pub post_ffw_norm_eps: f32,

    /// `[2 · inter, hidden]`, gate first.
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,

    /// The adapter bank at the attention sublayer's correction site — the
    /// `attn_norm`ed input and the reduced `o_proj` output, both replicated.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

struct Dims {
    hidden: u32,
    layers: u32,
    full_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    vocab: u32,
    window: u32,
    theta: f32,
    qk_scale: f32,
    softcap: f32,
    output_multiplier: f32,
    norm_eps: f32,
    post_norm_eps: f32,
}

impl Model {
    /// `meta-models/Muse-Glimmer-30B`'s text, off its `text_config`.
    pub fn b30(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::b30_dims())
    }

    /// The 30B cut to its first `layers` layers — the miniature a parity
    /// gate reads against an external reference of the same depth. The
    /// layer pattern is the full stack's (`full_every` 4), so a miniature
    /// carved as whole periods keeps every layer kind.
    pub fn b30_mini(layers: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b30_dims();
        d.layers = layers;
        Model::new(w, kv, tp, d)
    }

    fn b30_dims() -> Dims {
        Dims {
            hidden: 6656,
            layers: 52,
            full_every: 4,
            q_heads: 32,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 19_968,
            vocab: 202_048,
            window: 2048,
            theta: 500_000.0,
            qk_scale: 3.87,
            softcap: 20.0,
            output_multiplier: 0.196_116_135_138_184_04,
            norm_eps: 1e-5,
            post_norm_eps: 1e-8,
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2),
            "tp {tp} is not a world this catalog ships (two kv heads divide two ways)"
        );
        let dense = crate::dense(w);
        // `U4g64tiled` reorders `U4g64` codes into m16n8k16 fragment order for
        // `linear::tiled` — the 2-D projections `ops::linear::matmul` and
        // `lm_head` read. The embedding table stays row-major: a gather has
        // no tiled reader. `model_dsl::place` resolves the layout per
        // platform (CUDA tiled, Metal canonical). Without it a 4-bit row's
        // prefill decodes each projection into a per-region bf16 rectangle,
        // which on a 52-layer stack is tens of gigabytes.
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let intermediate = d.intermediate / tp;

        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(q_heads) * hd;
        let kv_w = u64::from(kv_heads) * hd;
        let iw = u64::from(intermediate);
        let full_at = |l: u32| l % d.full_every == d.full_every - 1;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str| Weight::sym(n(s), [hidden], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                Layer {
                    reading: if full_at(l) {
                        Reading::Full
                    } else {
                        Reading::Sliding
                    },
                    qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], proj)
                        .packed([q_w, kv_w, kv_w]),
                    gate: Weight::sym(n("gate"), [q_w, hidden], proj).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], proj).rows(),
                    kv: format!("kv.{l}"),
                    attn_norm: norm("attn_norm"),
                    attn_norm_eps: d.norm_eps,
                    post_attn_norm: norm("post_attn_norm"),
                    post_attn_norm_eps: d.post_norm_eps,
                    pre_ffw_norm: norm("pre_ffw_norm"),
                    pre_ffw_norm_eps: d.norm_eps,
                    post_ffw_norm: norm("post_ffw_norm"),
                    post_ffw_norm_eps: d.post_norm_eps,
                    gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
                    inter: intermediate,
                    down: Weight::sym(n("down"), [hidden, iw], proj).rows(),
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            kv_heads,
            head_dim: d.head_dim,
            window: d.window,
            theta: d.theta,
            sm_scale: d.qk_scale / (d.head_dim as f32).sqrt(),
            norm_eps: d.norm_eps,
            adapters: ADAPTERS,
            kv,
            softcap: d.softcap,
            output_multiplier: d.output_multiplier,
            embed: Weight::sym("embed", [u64::from(d.vocab), hidden], w),
            // The untied head is `vocab x hidden` of its own and every rank
            // streamed all of it. Band it on the vocab axis: each rank lands
            // its slice and `forward` all-gathers the logits shard. Exact —
            // partitioning a GEMM's output changes no reduction.
            // `PIE_NO_VOCAB_SHARD` restores the replicated head.
            lm_head: {
                let banded = tp > 1 && std::env::var_os("PIE_NO_VOCAB_SHARD").is_none();
                let rows = if banded { u64::from(d.vocab / tp) } else { u64::from(d.vocab) };
                let bank = Weight::sym("lm_head", [rows, hidden], proj);
                if banded { bank.packed([rows]) } else { bank }
            },
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
        }
    }
}

/// What every SKU seats. A deployment ceiling, not a checkpoint fact.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };
