//! **A FIRE TALLER THAN 65535 ROWS IS SPLIT WHOLE.**
//!
//! `layout.split_qkv` used to launch `grid = [width tiles, rows, 1]`, and
//! CUDA caps `gridDim.y` at 65535. A fire with more rows than that did not
//! fail: it split the first 65535 and left the rest holding whatever the
//! destination rectangles happened to contain, which is a wrong answer that
//! looks like a plausible one. `layout.split_rows` had already been moved to
//! `grid.x` for this reason; this one had not.
//!
//! Rows ride `grid.x` now (2^31 − 1 blocks) and the width tiles ride `y`,
//! where a projection needs a handful. The test asks the question that
//! distinguishes the two: split a rectangle one row TALLER than the old
//! ceiling and read the last row back.
//!
//! A video fire is why this matters and not a hypothetical: Wan's 832x480x17
//! clip is 1950 latent rows today, but the row count grows with the volume
//! and a longer or larger clip walks straight into the cap.

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

/// One past `gridDim.y`'s ceiling: the first row count the old launch could
/// not reach.
const TALL: u32 = 65_536;

#[test]
fn a_qkv_split_reaches_every_row_of_a_tall_fire() {
    common::arm_cache();
    // Narrow on purpose. The question is the ROW axis, and a wide rectangle
    // at this height is gigabytes for no extra answer.
    let (q_width, kv_width) = (8u32, 4u32);
    let stride = (q_width + 2 * kv_width) as usize;

    // A split is a BYTE COPY, so the reference is the source itself and the
    // comparison is bitwise — no tolerance, and no tagging scheme that bf16
    // would round away (65534 is not a bf16, which an earlier draft of this
    // test learned the hard way). A row the launch never reached keeps the
    // zeros it was allocated with, and a random row is not zeros.
    let mut lcg = Lcg::seeded(0x51);
    let (packed, _) = lcg.row(TALL as usize * stride);

    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let q_at = gpu.zeros(TALL as usize * q_width as usize * 2);
    let k_at = gpu.zeros(TALL as usize * kv_width as usize * 2);
    let v_at = gpu.zeros(TALL as usize * kv_width as usize * 2);

    let src = Tensor::new(packed_at, TALL, stride as u32, Dtype::Bf16);
    let mut q = Tensor::new(q_at, TALL, q_width, Dtype::Bf16);
    let mut k = Tensor::new(k_at, TALL, kv_width, Dtype::Bf16);
    let mut v = Tensor::new(v_at, TALL, kv_width, Dtype::Bf16);
    layout::split_qkv(&gpu.ctx(), src, q_width, kv_width, &mut q, &mut k, &mut v)
        .expect("a tall split is a split");
    gpu.sync();

    let q_out: Vec<u16> = gpu.down(q_at, TALL as usize * q_width as usize);
    let k_out: Vec<u16> = gpu.down(k_at, TALL as usize * kv_width as usize);
    let v_out: Vec<u16> = gpu.down(v_at, TALL as usize * kv_width as usize);

    // Every row, not a sample: at this width the whole rectangle is 1.5 MB
    // and a partial check would not say WHERE the split stopped.
    let mut first_wrong = None;
    for n in 0..TALL as usize {
        let row = &packed[n * stride..(n + 1) * stride];
        let q_ok = q_out[n * q_width as usize..(n + 1) * q_width as usize]
            == row[..q_width as usize];
        let k_ok = k_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[q_width as usize..(q_width + kv_width) as usize];
        let v_ok = v_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[(q_width + kv_width) as usize..];
        if !(q_ok && k_ok && v_ok) {
            first_wrong = Some(n);
            break;
        }
    }
    assert_eq!(
        first_wrong, None,
        "the split stopped at row {:?} of {TALL}; `gridDim.y` caps at 65535, so a \
         split whose rows ride `y` silently drops everything past it",
        first_wrong
    );
}
