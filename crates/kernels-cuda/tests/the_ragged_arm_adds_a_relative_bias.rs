//! `attention.ragged` under `RaggedMask::RelativeBias` lands what an f32
//! host reference lands: every scaled logit of a group has
//! `table[h][kj − qi + max_len − 1]` added before the softmax, per QUERY
//! head, over three groups of unequal length at 4 and at 64 heads; and a
//! table of zeros lands the plain arm's answer bit for bit.
//!
//! `CUDA_VISIBLE_DEVICES=<n> cargo test -p kernels-cuda --features cuda --test the_ragged_arm_adds_a_relative_bias`

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;
const HEAD_DIM: u32 = 64;
/// Three groups of unequal length: one under a 128-row tile, one past it,
/// one exactly half of it.
const SIZES: [u32; 3] = [37, 130, 64];
/// The longest group, so every distance the fire holds reads its own column.
const MAX_LEN: u32 = 130;

fn indptr(sizes: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in sizes {
        out.push(out.last().unwrap() + n as i32);
    }
    out
}

/// `[heads, 2·max_len − 1]`, values in `[-2, 2)`: wide enough to move the
/// softmax visibly, narrow enough to keep it from saturating.
fn random_table(rng: &mut Lcg, heads: usize, max_len: usize) -> Vec<f32> {
    (0..heads * (2 * max_len - 1))
        .map(|_| 2.0 * rng.unit())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    table: &[i32],
    bias: &[f32],
    max_len: usize,
    q_heads: usize,
    kv_heads: usize,
    sm_scale: f32,
) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let span = 2 * max_len - 1;
    let rows = q.len() / (q_heads * hd);
    let mut o = vec![0f32; rows * q_heads * hd];
    let group = q_heads / kv_heads;
    for g in 0..table.len() - 1 {
        let (r0, r1) = (table[g] as usize, table[g + 1] as usize);
        for r in r0..r1 {
            for h in 0..q_heads {
                let kh = h / group;
                let qv = &q[(r * q_heads + h) * hd..][..hd];
                let mut scores: Vec<f32> = (r0..r1)
                    .map(|j| {
                        let kv = &k[(j * kv_heads + kh) * hd..][..hd];
                        let dot = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>();
                        let d =
                            (j as i64 - r as i64 + max_len as i64 - 1).clamp(0, span as i64 - 1);
                        dot * sm_scale + bias[h * span + d as usize]
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut o[(r * q_heads + h) * hd..][..hd];
                for (j, p) in (r0..r1).zip(&scores) {
                    let vv = &v[(j * kv_heads + kh) * hd..][..hd];
                    for (d, x) in out.iter_mut().zip(vv) {
                        *d += p / sum * x;
                    }
                }
            }
        }
    }
    o
}

/// The device's answer under `mask`, as raw bf16 bits.
#[allow(clippy::too_many_arguments)]
fn fire(
    gpu: &mut Gpu,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    groups: Tensor,
    sm_scale: f32,
    mask: RaggedMask,
    rows: usize,
    qw: usize,
) -> Vec<u16> {
    let o_at = gpu.zeros(rows * qw * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows as u32, qw as u32, Dtype::Bf16);
    attn_ragged::ragged(
        &ctx, q, k, v, groups, groups, HEAD_DIM, sm_scale, mask, &mut o,
    )
    .expect("the ragged arm fires");
    gpu.sync();
    gpu.down(o_at, rows * qw)
}

fn check(q_heads: u32, kv_heads: u32, seed: u64) {
    let table = indptr(&SIZES);
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = (
        (q_heads * HEAD_DIM) as usize,
        (kv_heads * HEAD_DIM) as usize,
    );
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, q_f) = lcg.row(rows * qw);
    let (k_raw, k_f) = lcg.row(rows * kw);
    let (v_raw, v_f) = lcg.row(rows * kw);
    let bias = random_table(&mut lcg, q_heads as usize, MAX_LEN as usize);
    let sm_scale = 2.0 / (HEAD_DIM as f32).sqrt();
    let want = reference(
        &q_f,
        &k_f,
        &v_f,
        &table,
        &bias,
        MAX_LEN as usize,
        q_heads as usize,
        kv_heads as usize,
        sm_scale,
    );

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let bias_at = gpu.up(&bias);
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let mask = RaggedMask::RelativeBias {
        table: Tensor::new(bias_at, q_heads, 2 * MAX_LEN - 1, Dtype::F32),
        max_len: MAX_LEN,
    };
    let got = fire(&mut gpu, q, k, v, groups, sm_scale, mask, rows, qw);
    let mut worst = 0f32;
    for r in 0..rows {
        for c in 0..qw {
            let (g, w) = (from_bf16(got[r * qw + c]), want[r * qw + c]);
            let err = (g - w).abs();
            assert!(
                err <= TOLERANCE * w.abs().max(1.0),
                "{q_heads}/{kv_heads} heads: row {r} column {c} landed {g} against {w}"
            );
            worst = worst.max(err);
        }
    }
    eprintln!(
        "{q_heads}/{kv_heads} heads, groups {SIZES:?}, max_len {MAX_LEN}: worst |diff| {worst:.2e}"
    );
}

#[test]
fn the_relative_bias_lands_the_host_reference_at_four_heads() {
    check(4, 2, 0x5b1a);
}

#[test]
fn the_relative_bias_lands_the_host_reference_at_sixty_four_heads() {
    check(64, 16, 0x5b64);
}

/// At a power-of-two `sm_scale` (every head width's default at 64 and 256)
/// the bias arm's `(q·k · s) · log2e` and the plain arm's `q·k · (s · log2e)`
/// are the same reals rounded the same way, so a zero table is the plain
/// arm bit for bit — which is what makes the arm a strict extension.
#[test]
fn a_zero_table_is_the_plain_arm_bit_for_bit() {
    let (q_heads, kv_heads) = (4u32, 2u32);
    let table = indptr(&SIZES);
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = (
        (q_heads * HEAD_DIM) as usize,
        (kv_heads * HEAD_DIM) as usize,
    );
    let mut lcg = Lcg::seeded(0x0);
    let (q_raw, _) = lcg.row(rows * qw);
    let (k_raw, _) = lcg.row(rows * kw);
    let (v_raw, _) = lcg.row(rows * kw);
    let zeros = vec![0f32; q_heads as usize * (2 * MAX_LEN as usize - 1)];
    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let bias_at = gpu.up(&zeros);
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let plain = fire(
        &mut gpu,
        q,
        k,
        v,
        groups,
        sm_scale,
        RaggedMask::None,
        rows,
        qw,
    );
    let mask = RaggedMask::RelativeBias {
        table: Tensor::new(bias_at, q_heads, 2 * MAX_LEN - 1, Dtype::F32),
        max_len: MAX_LEN,
    };
    let biased = fire(&mut gpu, q, k, v, groups, sm_scale, mask, rows, qw);
    assert_eq!(plain, biased, "a zero bias is the plain arm, bit for bit");
}

#[test]
fn a_misshapen_table_is_refused() {
    let mut gpu = Gpu::open();
    let q_at = gpu.zeros(8 * 128 * 2);
    let o_at = gpu.zeros(8 * 128 * 2);
    let table = gpu.up(&[0i32, 4, 8]);
    let bias = gpu.up(&[0f32; 2 * 7]);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, 8, 128, Dtype::Bf16);
    let mut o = Tensor::new(o_at, 8, 128, Dtype::Bf16);
    let groups = Tensor::new(table, 3, 1, Dtype::I32);
    // Two heads of 64; the table is two rows of 7 = 2·4 − 1, but max_len
    // says 8 (a row of 15).
    let mask = RaggedMask::RelativeBias {
        table: Tensor::new(bias, 2, 7, Dtype::F32),
        max_len: 8,
    };
    let refused = attn_ragged::ragged(&ctx, q, q, q, groups, groups, 64, 0.0, mask, &mut o);
    assert!(
        refused.is_err(),
        "a table narrower than 2·max_len − 1 is refused"
    );
}
