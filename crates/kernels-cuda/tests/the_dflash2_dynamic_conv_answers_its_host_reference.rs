//! **DFlash2's two-tap grouped dynamic convolution**, held against a host
//! transcription of the reference's rule
//! (`mlx_dspark.dflash_model.DFlashGroupedConv._convolve`):
//!
//! ```text
//! coeff[i, t, c] = base[side, t, c] + delta[i, t, g(c)]
//! y[i, c]        = Σ_t coeff[i, t, c] · x[i − t, c],   x[i − t] = 0 for i < t
//! ```
//!
//! within each request's own span of rows — a request's first row has no
//! in-block predecessor, so its taps past zero read nothing. Both sides of
//! the projection are checked (`side` 0 convolves a sublayer's input, 1 its
//! output), over three requests of different lengths including a one-row
//! span, which is the case where every tap but the zeroth is dead.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda --test the_dflash2_dynamic_conv_answers_its_host_reference
//! ```

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dynconv;
use kernels_cuda::tensor::{RaggedTensor, Tensor};

#[test]
fn the_dynamic_conv_mixes_each_row_with_the_one_before_it() {
    // Three requests: spans of 5, 1 and 3 rows. The one-row span is the
    // anchor-only case; the others exercise the in-span walk.
    let indptr: [i32; 4] = [0, 5, 6, 9];
    let rows = 9usize;
    let channels = 128usize;
    let taps = 2usize;
    let group = 32usize;
    let groups = channels / group;
    let coeff_width = 2 * taps * groups;

    let mut lcg = Lcg::seeded(0x2d);
    let (x_raw, x) = lcg.row(rows * channels);
    let (coeff_raw, coeff) = lcg.row(rows * coeff_width);
    let (base_raw, base) = lcg.row(2 * taps * channels);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let indptr_at = gpu.up(&indptr);
    let coeff_at = gpu.up(&coeff_raw);
    let base_at = gpu.up(&base_raw);

    for side in 0..2usize {
        let y_at = gpu.zeros(rows * channels * 2);
        let mut y = Tensor::new(y_at, rows as u32, channels as u32, Dtype::Bf16);
        dynconv::block_dyn_conv(
            &gpu.ctx(),
            RaggedTensor {
                data: Tensor::new(x_at, rows as u32, channels as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr_at, 4, 1, Dtype::I32),
            },
            Tensor::new(coeff_at, rows as u32, coeff_width as u32, Dtype::Bf16),
            Tensor::new(base_at, (2 * taps) as u32, channels as u32, Dtype::Bf16),
            side as u32,
            taps as u32,
            group as u32,
            &mut y,
        )
        .expect("the dynamic conv fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(y_at, rows * channels);

        for lane in 0..3usize {
            let (begin, end) = (indptr[lane] as usize, indptr[lane + 1] as usize);
            for t in 0..end - begin {
                let row = begin + t;
                for c in 0..channels {
                    let g = c / group;
                    let mut want = 0f32;
                    for k in 0..taps {
                        if t < k {
                            break;
                        }
                        let at = side * taps + k;
                        let coef = base[at * channels + c] + coeff[row * coeff_width + at * groups + g];
                        want += coef * x[(begin + t - k) * channels + c];
                    }
                    let got = from_bf16(got[row * channels + c]);
                    assert!(
                        close(got, want),
                        "side {side} lane {lane} row {row} channel {c}: {got} against {want}"
                    );
                }
            }
        }
    }
}
