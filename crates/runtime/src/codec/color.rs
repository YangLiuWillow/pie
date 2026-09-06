//! RGB8 -> YUV 4:2:0, in the two layouts the two consumers want.
//!
//! NUMERICS CONTRACT. BT.601 studio swing (Y in 16..235, chroma in 16..240),
//! integer coefficients, rounding by `+128 >> 8`:
//!
//! ```text
//! Y =  (( 66R + 129G +  25B + 128) >> 8) +  16
//! U =  ((-38R -  74G + 112B + 128) >> 8) + 128
//! V =  ((112R -  94G -  18B + 128) >> 8) + 128
//! ```
//!
//! This is the matrix NVENC assumes for an NV12 input surface when nothing
//! says otherwise, and the one a y4m reader assumes for `C420mpeg2`, so the
//! two outputs of one handle describe the same picture. Chroma is subsampled
//! by averaging the four RGB triples of a 2x2 block BEFORE the transform —
//! averaging in RGB rather than in chroma, which is one rounding step instead
//! of two.
//!
//! Odd dimensions have no 2x2 block at the edge and are refused by the
//! callers rather than guessed at here; both functions assume even.

/// One 2x2 block's chroma, from the block's mean RGB.
#[inline]
fn chroma(r: i32, g: i32, b: i32) -> (u8, u8) {
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (u.clamp(0, 255) as u8, v.clamp(0, 255) as u8)
}

/// One pixel's luma.
#[inline]
fn luma(r: i32, g: i32, b: i32) -> u8 {
    (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255) as u8
}

/// The mean RGB of the 2x2 block whose top-left pixel is `(x, y)`.
#[inline]
fn block_mean(rgb: &[u8], width: usize, x: usize, y: usize) -> (i32, i32, i32) {
    let mut acc = [0i32; 3];
    for dy in 0..2 {
        for dx in 0..2 {
            let p = ((y + dy) * width + (x + dx)) * 3;
            acc[0] += rgb[p] as i32;
            acc[1] += rgb[p + 1] as i32;
            acc[2] += rgb[p + 2] as i32;
        }
    }
    ((acc[0] + 2) / 4, (acc[1] + 2) / 4, (acc[2] + 2) / 4)
}

/// Write one interleaved RGB8 frame as planar I420 (Y plane, then U, then V)
/// into `out`, appending. `width` and `height` must be even.
pub fn rgb8_to_i420(rgb: &[u8], width: usize, height: usize, out: &mut Vec<u8>) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let p = (y * width + x) * 3;
            out.push(luma(rgb[p] as i32, rgb[p + 1] as i32, rgb[p + 2] as i32));
        }
    }
    let mut vs = Vec::with_capacity(width * height / 4);
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let (r, g, b) = block_mean(rgb, width, x, y);
            let (u, v) = chroma(r, g, b);
            out.push(u);
            vs.push(v);
        }
    }
    out.extend_from_slice(&vs);
}

/// Write one interleaved RGB8 frame as NV12 (Y plane, then one interleaved UV
/// plane) into a destination whose rows are `pitch` bytes apart — the shape
/// [`crate::codec::nvenc`] gets back from `NvEncLockInputBuffer`. `width` and
/// `height` must be even, and `dst` must hold `pitch * height * 3 / 2` bytes.
pub fn rgb8_to_nv12_pitched(rgb: &[u8], width: usize, height: usize, pitch: usize, dst: &mut [u8]) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert!(pitch >= width);
    debug_assert!(dst.len() >= pitch * height + pitch * height / 2);
    for y in 0..height {
        let row = &mut dst[y * pitch..y * pitch + width];
        for (x, cell) in row.iter_mut().enumerate() {
            let p = (y * width + x) * 3;
            *cell = luma(rgb[p] as i32, rgb[p + 1] as i32, rgb[p + 2] as i32);
        }
    }
    let uv_base = pitch * height;
    for y in (0..height).step_by(2) {
        let row = uv_base + (y / 2) * pitch;
        for x in (0..width).step_by(2) {
            let (r, g, b) = block_mean(rgb, width, x, y);
            let (u, v) = chroma(r, g, b);
            dst[row + x] = u;
            dst[row + x + 1] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_is_grey_in_both_layouts() {
        // A flat mid-grey: every chroma sample must land on the neutral 128,
        // which is the one value that proves the matrix is centred right.
        let rgb = vec![128u8; 4 * 4 * 3];
        let mut i420 = Vec::new();
        rgb8_to_i420(&rgb, 4, 4, &mut i420);
        assert_eq!(i420.len(), 16 + 4 + 4);
        assert!(i420[16..].iter().all(|&c| c == 128));

        let mut nv12 = vec![0u8; 8 * 4 + 8 * 2];
        rgb8_to_nv12_pitched(&rgb, 4, 4, 8, &mut nv12);
        // Y is the same value in both, and the pitch padding is untouched.
        assert_eq!(i420[0], nv12[0]);
        for y in 0..2 {
            for x in 0..4 {
                assert_eq!(nv12[8 * 4 + y * 8 + x], 128);
            }
        }
    }

    #[test]
    fn primaries_land_where_bt601_puts_them() {
        // Pure red: Y = ((66*255 + 128) >> 8) + 16 = 82, V well above neutral,
        // U well below. The chroma pair is checked as an inequality, not a
        // golden, because the claim there is the matrix's sign structure.
        let rgb = vec![255u8, 0, 0].repeat(4);
        let mut out = Vec::new();
        rgb8_to_i420(&rgb, 2, 2, &mut out);
        assert_eq!(out[0], 82);
        assert!(out[4] < 100, "U for red should sit below neutral");
        assert!(out[5] > 200, "V for red should sit above neutral");
    }
}
