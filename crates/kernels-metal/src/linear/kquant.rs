//! `kquant`: the ggml K-quant super-block decode-in-dot, the Metal twin of
//! `kernels_cuda::linear::kquant`. A K-quant weight reaches here as one `U8`
//! byte rectangle (`engine_metal::Run::maybe_stored` re-badges the stored
//! `Dense` handle); the scales live inside each 256-element super-block and are
//! decoded in the dot rather than off a companion plane, which is what lets pie
//! serve a GGUF at the size it shipped instead of inflating it to the
//! activation dtype at import.
//!
//! A K-quant carries no second plane and no dtype to dispatch on: a `[n, k]`
//! weight is `n` rows of consecutive super-blocks ([`SUPER`] elements each),
//! and each scheme's super-block byte width is distinct, so a row's byte width
//! over the activation's contraction names the scheme unambiguously.
//!
//! PR2 lands q4_k and q6_k — the Q4_K_M mix (q4_k bodies, q6_k `output.weight`)
//! — diffed bit-exact against the `checkpoint` host reference decoder. q2_k /
//! q3_k / q5_k refuse until their points land.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "linear/kquant.metal";

/// Elements in one K-quant super-block, all five schemes.
const SUPER: u32 = 256;

/// Per-scheme super-block byte widths (`block_q*_K` sizes), the ladder a row's
/// byte width is matched against.
const Q2K_BYTES: u32 = 84;
const Q3K_BYTES: u32 = 110;
const Q4K_BYTES: u32 = 144;
const Q5K_BYTES: u32 = 176;
const Q6K_BYTES: u32 = 210;

/// Weight rows one simdgroup folds, and the simdgroups a threadgroup launches
/// — kept equal to `kquant.metal`'s `ROWS_PER_WARP` template arg and `KNUM_SIMD`.
const ROWS_PER_WARP: u32 = 4;
const NUM_SIMD: u32 = 4;
const LANES: u32 = 32;

/// The family ascending by super-block width: `(bytes, tag, on_metal)`. `tag`
/// spells the `kquant.metal` entry and the GGUF name; `on_metal` marks the
/// points that have a shader (all five now; the flag stays for a width that is
/// recognized but not yet stamped).
const FAMILY: [(u32, &str, bool); 5] = [
    (Q2K_BYTES, "q2k", true),
    (Q3K_BYTES, "q3k", true),
    (Q4K_BYTES, "q4k", true),
    (Q5K_BYTES, "q5k", true),
    (Q6K_BYTES, "q6k", true),
];

/// Which K-quant a weight row's byte width names, over a `k`-wide contraction.
/// `k / SUPER` super-blocks per row divides the row's byte width to exactly one
/// scheme's block size.
fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<&'static str, Error> {
    let blocks = k / SUPER;
    for (width, tag, on_metal) in FAMILY {
        if blocks.checked_mul(width) == Some(row_bytes) {
            if !on_metal {
                return Err(refuse(
                    op,
                    format!(
                        "{tag} is a K-quant width Metal recognizes but has no \
                         shader for: a {row_bytes}-byte row over {blocks} super-blocks"
                    ),
                ));
            }
            return Ok(tag);
        }
    }
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is none of the five K-quant widths \
             over a {k}-wide contraction ({blocks} super-blocks)"
        ),
    ))
}

/// Interns a composed entry name into a `&'static str`; names are few, so the
/// leak is bounded (mirrors `linear::quant::symbol`).
fn symbol(name: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}

/// `y = act x w^T` where `w` is a stored K-quant super-block rectangle.
pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.kquant.matmul", act, w, y)
}

/// The head's stored-block arm; a Q4_K_M mix stores `output.weight` at q6_k.
pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.kquant.lm_head", act, w, y)
}

fn kquant(ctx: &Ctx<'_>, op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "bfloat16", F16 => "float16" });
    debug_assert_eq!(w.dtype, Dtype::U8, "a stored K-quant plane binds as bytes");
    debug_assert_eq!(act.rows, y.rows, "the activation's rows are the rows the result lands");

    let m = act.rows;
    let n = y.width;
    let k = act.width;
    if n == 0 || k == 0 {
        return Err(refuse(op, "a projection with a zero axis"));
    }
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    debug_assert_eq!(w.rows, n, "one weight row per column this projection lands");

    let tag = scheme(op, k, w.width)?;
    // A fire with no rows lands nothing.
    if m == 0 {
        return Ok(());
    }
    let entry = symbol(&format!("kquant_matmul_{tag}_{t}_r_{ROWS_PER_WARP}"));

    // Grid mirrors CUDA: threadgroup (token, column-tile); NUM_SIMD simdgroups
    // per group each fold ROWS_PER_WARP rows, 32 lanes stride the super-blocks.
    // `lanes` is the total thread grid (see `quant::qmv_grid`): x = m tokens x
    // 32 lanes; y = n rows / ROWS_PER_WARP, over group.y = NUM_SIMD simdgroups.
    let x = m
        .checked_mul(LANES)
        .ok_or_else(|| refuse(op, format!("{m} tokens will not launch")))?;
    let lanes = [x, n.div_ceil(ROWS_PER_WARP), 1];
    let group = [LANES, NUM_SIMD, 1];

    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, group)),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )
}
