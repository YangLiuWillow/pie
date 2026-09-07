//! **THE FIRST READING OF K-QUANT ON METAL.** `kernels_metal::linear::kquant`
//! decodes a stored ggml K-quant super-block inside the dot; this file fires it
//! over a small `[m, k] x [n, k]^T` projection and reads the result against an
//! independent host reference — a Rust transcription of
//! `checkpoint::executor::walk::decode_gguf_q{4,6}_k_block_into` (the same
//! oracle the CUDA kernel is graded against) followed by a plain f32 dot.
//!
//! The decode arithmetic is exact by construction: the super-scales `d`/`dmin`
//! are chosen as exact powers of two (so their f16 bytes round-trip), and every
//! payload/scale byte the kernel reads is the byte the reference reads. What is
//! NOT bit-identical is the accumulation: the kernel folds per-super-block via
//! the affine identity and stores bf16, the reference sums per element in f32,
//! so the two agree to a bf16-scale tolerance, not to the ULP.
//!
//! Skips (does not fail) when no Metal device binds, so a non-GPU `cargo test`
//! still type-checks it. Runs on the M5:
//!
//! ```text
//! cargo test -p engine-metal --release \
//!   --test the_kquant_matmul_answers_the_host_decoded_dot -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::kquant;
use model_ir::Dtype;

// Exact-in-f16 super-scales: 2^-4 = 0x2C00, 2^-5 = 0x2800 (mantissa zero, so
// the f16 -> f32 the kernel does is exact and the reference can use the f32
// constant directly).
const D: f32 = 0.0625; // 2^-4
const DMIN: f32 = 0.03125; // 2^-5
const D_LE: [u8; 2] = [0x00, 0x2C];
const DMIN_LE: [u8; 2] = [0x00, 0x28];

const Q4K_BYTES: usize = 144;
const Q6K_BYTES: usize = 210;
const SUPER: usize = 256;

/// A tiny deterministic byte stream — no `rand` in this crate's dev-deps.
struct Lcg(u64);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

/// ggml `get_scale_min_k4`, the reference twin of the shader's `q4k_scale_min`.
fn q4k_scale_min(sub: usize, s: &[u8]) -> (u8, u8) {
    if sub < 4 {
        (s[sub] & 63, s[sub + 4] & 63)
    } else {
        let scale = (s[sub + 4] & 0x0f) | ((s[sub - 4] >> 6) << 4);
        let mn = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4);
        (scale, mn)
    }
}

/// Reference decode of one Q4_K super-block (mirrors `decode_gguf_q4_k_block_into`).
fn decode_q4k(blk: &[u8], out: &mut [f32; SUPER]) {
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for pair in 0..4 {
        let (sc_lo, m_lo) = q4k_scale_min(pair * 2, scales);
        let (sc_hi, m_hi) = q4k_scale_min(pair * 2 + 1, scales);
        let (d_lo, min_lo) = (D * f32::from(sc_lo), DMIN * f32::from(m_lo));
        let (d_hi, min_hi) = (D * f32::from(sc_hi), DMIN * f32::from(m_hi));
        let packed = &qs[pair * 32..pair * 32 + 32];
        let o = pair * 64;
        for i in 0..32 {
            out[o + i] = d_lo * f32::from(packed[i] & 0x0f) - min_lo;
            out[o + 32 + i] = d_hi * f32::from(packed[i] >> 4) - min_hi;
        }
    }
}

/// Reference decode of one Q6_K super-block (mirrors `decode_gguf_q6_k_block_into`).
fn decode_q6k(blk: &[u8], out: &mut [f32; SUPER]) {
    for hlf in 0..2 {
        let ql = &blk[hlf * 64..hlf * 64 + 64];
        let qh = &blk[128 + hlf * 32..128 + hlf * 32 + 32];
        let sc = &blk[192 + hlf * 8..192 + hlf * 8 + 8];
        let o = hlf * 128;
        for i in 0..32 {
            let sub = i / 16;
            for quarter in 0..4 {
                let nibble = if quarter < 2 {
                    ql[i + 32 * quarter] & 0x0f
                } else {
                    ql[i + 32 * (quarter - 2)] >> 4
                };
                let top = (qh[i] >> (2 * quarter)) & 3;
                let q = i32::from(nibble | (top << 4)) - 32;
                let scale = f32::from(sc[sub + 2 * quarter] as i8);
                out[o + quarter * 32 + i] = D * scale * q as f32;
            }
        }
    }
}

/// Build a stored weight `[n, k]` and its host-decoded f32 twin. `block_bytes`
/// and `decode` pick the scheme; the super-scale bytes are stamped exact, the
/// rest filled from the LCG.
fn build_weight(
    n: usize,
    k: usize,
    block_bytes: usize,
    is_q6k: bool,
    decode: fn(&[u8], &mut [f32; SUPER]),
) -> (Vec<u8>, Vec<f32>) {
    let blocks = k / SUPER;
    let row_bytes = blocks * block_bytes;
    let mut bytes = vec![0u8; n * row_bytes];
    let mut wref = vec![0f32; n * k];
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    for r in 0..n {
        for g in 0..blocks {
            let base = r * row_bytes + g * block_bytes;
            let blk = &mut bytes[base..base + block_bytes];
            for b in blk.iter_mut() {
                *b = lcg.byte();
            }
            // Stamp the exact super-scales over the random fill.
            if is_q6k {
                blk[208] = D_LE[0];
                blk[209] = D_LE[1];
            } else {
                blk[0] = D_LE[0];
                blk[1] = D_LE[1];
                blk[2] = DMIN_LE[0];
                blk[3] = DMIN_LE[1];
            }
            let mut vals = [0f32; SUPER];
            decode(blk, &mut vals);
            for (e, v) in vals.iter().enumerate() {
                wref[r * k + g * SUPER + e] = *v;
            }
        }
    }
    (bytes, wref)
}

/// One scheme, end to end: fire the kernel, read it against the host-decoded dot.
fn check(scheme: &str, block_bytes: usize, is_q6k: bool, decode: fn(&[u8], &mut [f32; SUPER])) {
    let Ok(device) = Context::bind() else {
        eprintln!("skip {scheme}: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();

    // n not a multiple of the 16-row tile, so the row<n store guard is exercised.
    let n = 40usize;
    let k = 2048usize; // 8 super-blocks per row -> the lane super-block stride runs
    let m = 5usize;

    let (w_bytes, w_ref) = build_weight(n, k, block_bytes, is_q6k, decode);

    // Activations: small deterministic bf16 values; the reference reads the same
    // bf16-rounded numbers the kernel does.
    let mut act_bytes = Vec::with_capacity(m * k * 2);
    let mut act_f32 = vec![0f32; m * k];
    let mut lcg = Lcg(0x2545_F491_4F6C_DD1D);
    for t in 0..m {
        for e in 0..k {
            let raw = (i32::from(lcg.byte()) - 128) as f32 * (1.0 / 512.0); // ~[-0.25, 0.25)
            let bits = f32_to_bf16(raw);
            act_f32[t * k + e] = bf16_to_f32(bits);
            act_bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }

    let mut w_b = Buffer::zeroed(&device, w_bytes.len() as u64).expect("w buffer");
    w_b.write(0, &w_bytes).expect("write w");
    let mut a_b = Buffer::zeroed(&device, act_bytes.len() as u64).expect("act buffer");
    a_b.write(0, &act_bytes).expect("write act");
    let y_b = Buffer::zeroed(&device, (m * n * 2) as u64).expect("y buffer");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hw, ha, hy) = (bind(&w_b), bind(&a_b), bind(&y_b));
    let row_bytes = (k / SUPER) * block_bytes;
    let w = Tensor::new(hw, n as u32, row_bytes as u32, Dtype::U8);
    let act = Tensor::new(ha, m as u32, k as u32, Dtype::Bf16);
    let y = Tensor::new(hy, m as u32, n as u32, Dtype::Bf16);

    let frame = device.frame().expect("a frame");
    let sink = Sink::new(&device, &frame, &pipelines, &handles);
    kquant::matmul(&sink, act, w, y).expect("the kquant launch");
    frame.commit().expect("the commit");

    let mut out = vec![0u8; m * n * 2];
    y_b.read(0, &mut out).expect("read y");

    // The reference dot, in f32 over the same bf16 activations.
    let mut worst = 0f32;
    let mut scale = 1e-6f32;
    for t in 0..m {
        for r in 0..n {
            let mut acc = 0f32;
            for e in 0..k {
                acc += act_f32[t * k + e] * w_ref[r * k + e];
            }
            let got = bf16_to_f32(u16::from_le_bytes([out[(t * n + r) * 2], out[(t * n + r) * 2 + 1]]));
            worst = worst.max((got - acc).abs());
            scale = scale.max(acc.abs());
        }
    }
    let rel = worst / scale;
    eprintln!("{scheme}: worst abs {worst:.5}, scale {scale:.3}, rel {rel:.5}");
    // bf16 output carries ~2^-8 relative; the affine-identity regrouping adds a
    // little more. 2% is comfortably above that and well below a decode fault.
    assert!(rel < 0.02, "{scheme}: kquant matmul strayed {rel:.5} from the host-decoded dot");
}

#[test]
fn q4_k_matmul_answers_the_host_decoded_dot() {
    check("q4_k", Q4K_BYTES, false, decode_q4k);
}

#[test]
fn q6_k_matmul_answers_the_host_decoded_dot() {
    check("q6_k", Q6K_BYTES, true, decode_q6k);
}
