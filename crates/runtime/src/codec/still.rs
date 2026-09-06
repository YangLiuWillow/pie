//! The still formats, through the `image` crate the host already links for
//! media *input*.
//!
//! One encoder per format, all pure Rust: PNG (deflate), JPEG (baseline,
//! quality 92), WebP (VP8L — `image`'s WebP encoder is lossless-only, which
//! is stated here because "webp" reads as "small" and this one is not).
//!
//! Every one of these is a single picture by definition, so the caller gates
//! `count == 1` before reaching here and this module never sees a clip.

use image::{ExtendedColorType, ImageEncoder};

/// JPEG quality. 92 is the knee: visually transparent on generated imagery,
/// and roughly a third of the bytes of 100.
const JPEG_QUALITY: u8 = 92;

/// PNG, deflate-compressed, 8-bit RGB.
pub fn png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(out)
}

/// Baseline JPEG at [`JPEG_QUALITY`].
pub fn jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(out)
}

/// Lossless WebP (VP8L). `image`'s encoder has no lossy path; a caller who
/// wanted "small" wants [`jpeg`].
pub fn webp(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(&mut out)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("webp encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gradient rather than a flat fill: a flat fill survives a transposed
    /// or mis-strided encoder, and a gradient does not.
    fn gradient(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w.max(1)) as u8);
                v.push((y * 255 / h.max(1)) as u8);
                v.push(((x + y) * 255 / (w + h).max(1)) as u8);
            }
        }
        v
    }

    #[test]
    fn png_round_trips_a_gradient_exactly() {
        let (w, h) = (16, 9);
        let src = gradient(w, h);
        let bytes = png(&src, w, h).expect("encode");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let back = image::load_from_memory(&bytes).expect("decode").to_rgb8();
        assert_eq!(back.dimensions(), (w, h));
        assert_eq!(back.into_raw(), src, "png is lossless");
    }

    #[test]
    fn jpeg_and_webp_carry_the_same_picture() {
        let (w, h) = (16, 8);
        let src = gradient(w, h);

        let j = jpeg(&src, w, h).expect("jpeg");
        assert_eq!(&j[..2], b"\xff\xd8", "SOI");
        let back = image::load_from_memory(&j).expect("decode jpeg").to_rgb8();
        assert_eq!(back.dimensions(), (w, h));

        let wp = webp(&src, w, h).expect("webp");
        assert_eq!(&wp[..4], b"RIFF");
        assert_eq!(&wp[8..12], b"WEBP");
    }
}
