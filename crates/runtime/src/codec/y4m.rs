//! YUV4MPEG2 — the uncompressed clip format, and the parity format for mp4.
//!
//! One ASCII header, then `FRAME\n` + planar I420 per frame. Every decoder
//! reads it and nothing about it is lossy, so a y4m of the same handle is what
//! makes an H.264 regression provable: the two decode to pictures that should
//! differ only by the codec.
//!
//! `C420mpeg2` names the chroma siting that matches [`super::color`]'s
//! studio-swing BT.601 transform. `Ip` is progressive; `A1:1` is square
//! pixels.

use super::color::rgb8_to_i420;

/// Frame rate as an exact `num:den` — y4m's header wants a ratio, and a
/// float would round 23.976 into a lie. `fps` is multiplied by 1000 and
/// rounded, which is exact for every rate a generative model states.
pub fn frame_rate(fps: f32) -> (u32, u32) {
    if !fps.is_finite() || fps <= 0.0 {
        // A still, or a producer that stated no rate. y4m's header is
        // mandatory, so say one frame per second rather than refuse.
        return (1, 1);
    }
    let num = (fps * 1000.0).round().max(1.0) as u32;
    (num, 1000)
}

/// Write `count` interleaved-RGB8 frames as one y4m stream.
///
/// Errors on odd dimensions: 4:2:0 has no 2x2 block at an odd edge, and
/// silently cropping a row would be a worse answer than a refusal that names
/// the size.
pub fn encode(
    rgb: &[u8],
    width: u32,
    height: u32,
    count: u32,
    fps: f32,
) -> Result<Vec<u8>, String> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!(
            "y4m is 4:2:0 and needs even dimensions; this handle is {width}x{height}"
        ));
    }
    let (w, h, n) = (width as usize, height as usize, count as usize);
    let frame_bytes = w * h * 3;
    if rgb.len() != frame_bytes * n {
        return Err(format!(
            "y4m: {} bytes for {n} frames of {width}x{height} (expected {})",
            rgb.len(),
            frame_bytes * n
        ));
    }
    let (num, den) = frame_rate(fps);
    let mut out = Vec::with_capacity(64 + n * (6 + w * h * 3 / 2));
    out.extend_from_slice(
        format!("YUV4MPEG2 W{width} H{height} F{num}:{den} Ip A1:1 C420mpeg2\n").as_bytes(),
    );
    for f in 0..n {
        out.extend_from_slice(b"FRAME\n");
        rgb8_to_i420(&rgb[f * frame_bytes..(f + 1) * frame_bytes], w, h, &mut out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_is_a_header_and_then_one_frame_each() {
        let rgb = vec![0u8; 4 * 2 * 3 * 3];
        let out = encode(&rgb, 4, 2, 3, 25.0).expect("encode");
        assert!(out.starts_with(b"YUV4MPEG2 W4 H2 F25000:1000 Ip A1:1 C420mpeg2\n"));
        assert_eq!(out.windows(6).filter(|w| *w == b"FRAME\n").count(), 3);
        let header = out.iter().position(|&b| b == b'\n').unwrap() + 1;
        // 3 frames of (Y = 8) + (U = 2) + (V = 2) = 12 bytes, each behind a
        // 6-byte FRAME marker.
        assert_eq!(out.len() - header, 3 * (6 + 12));
    }

    #[test]
    fn odd_dimensions_are_refused_by_name() {
        let err = encode(&vec![0u8; 3 * 3 * 3], 3, 3, 1, 1.0).unwrap_err();
        assert!(err.contains("3x3"), "{err}");
    }
}
