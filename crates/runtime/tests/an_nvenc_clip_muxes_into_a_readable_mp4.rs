//! **REAL NVENC OUTPUT MUXES INTO AN MP4 WHOSE BOXES PARSE BACK.**
//!
//! ```text
//! PIE_NVENC_TEST=1 CUDA_VISIBLE_DEVICES=0 \
//!   cargo test -p runtime --features cuda \
//!   --test an_nvenc_clip_muxes_into_a_readable_mp4 -- --ignored
//! ```
//!
//! The muxer's own gates (`crate::codec::mp4`) feed it a synthetic Annex-B
//! stream, which proves the boxes and proves nothing about the encoder. This
//! is the other half: eight frames through the driver's H.264 encoder and out
//! the far side of `frames.encode(mp4-h264)`, checked for the structure a
//! player needs — `ftyp`/`mdat`/`moov`, an `avcC` carrying a real SPS, one
//! sample per frame, and the chunk offset landing inside `mdat`.
//!
//! 160x96 rather than something smaller: NVENC's H.264 floor is 145x49
//! (measured, and what the SDK documents), and `codec::nvenc` refuses below it
//! by name rather than passing the driver's own unhelpful message through.
//!
//! `#[ignore]` because it needs a GPU with an encoder; the `PIE_NVENC_TEST`
//! guard is belt and braces for a `--ignored` sweep on a machine without one.

#![cfg(feature = "cuda")]

use runtime::codec::mp4;
use runtime::inferlet::{Frames, ImageFormat};

/// Moving content, so the encoder has inter prediction to do rather than
/// eight identical intra frames.
fn moving_gradient(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * n * 3) as usize);
    for f in 0..n {
        for y in 0..h {
            for x in 0..w {
                v.push(((x + f * 4) % 256) as u8);
                v.push((y % 256) as u8);
                v.push(((x + y + f * 8) % 256) as u8);
            }
        }
    }
    v
}

#[test]
#[ignore = "needs an NVIDIA encoder; run with PIE_NVENC_TEST=1 -- --ignored"]
fn eight_frames_encode_and_the_container_reads_back() {
    if std::env::var("PIE_NVENC_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PIE_NVENC_TEST=1 to run the NVENC gate");
        return;
    }
    let (w, h, n) = (160u32, 96u32, 8u32);
    let clip = Frames::from_rgb8(moving_gradient(w, h, n), w, h, n, 25.0).expect("build handle");

    let bytes = clip
        .encode(ImageFormat::Mp4H264)
        .expect("NVENC encode + mux");
    assert!(
        bytes.len() > 256,
        "an 8-frame mp4 cannot be {} bytes",
        bytes.len()
    );

    let tree = mp4::parse_boxes(&bytes).expect("the muxer's own walker reads its own file");
    let names: Vec<&str> = tree.iter().map(|b| b.name()).collect();
    assert_eq!(names, ["ftyp", "mdat", "moov"]);
    assert_eq!(
        tree.iter().map(|b| b.size).sum::<usize>(),
        bytes.len(),
        "the walk consumed the file exactly"
    );

    // The sample entry must carry a real sequence parameter set: profile,
    // compatibility and level are the encoder's, not the muxer's.
    let avcc = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stsd/avc1/avcC").expect("avcC");
    let cfg = &bytes[avcc.body..avcc.offset + avcc.size];
    assert_eq!(cfg[0], 1, "configurationVersion");
    assert_eq!(cfg[4], 0xff, "lengthSizeMinusOne = 3");
    assert_eq!(cfg[5] & 0x1f, 1, "exactly one SPS");
    let sps_len = u16::from_be_bytes(cfg[6..8].try_into().unwrap()) as usize;
    assert!(sps_len >= 4, "SPS is {sps_len} bytes");
    assert_eq!(cfg[8] & 0x1f, 7, "the SPS NAL is really a SPS");

    // One sample per submitted frame, each non-empty, and the first is a sync
    // sample (there is no other way for a decoder to start).
    let stsz = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stsz").expect("stsz");
    let body = &bytes[stsz.body..];
    assert_eq!(u32::from_be_bytes(body[8..12].try_into().unwrap()), n);
    for i in 0..n as usize {
        let at = 12 + i * 4;
        let size = u32::from_be_bytes(body[at..at + 4].try_into().unwrap());
        assert!(size > 0, "sample {i} is empty");
    }

    // The chunk offset points into `mdat`'s payload, which is what makes the
    // file playable rather than merely well-formed.
    let mdat = tree.iter().find(|b| b.name() == "mdat").unwrap();
    let stco = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stco").expect("stco");
    let offset =
        u32::from_be_bytes(bytes[stco.body + 8..stco.body + 12].try_into().unwrap()) as usize;
    assert_eq!(offset, mdat.body);
    assert!(offset < mdat.offset + mdat.size);

    // The frame rate survives: 25 fps is 25000/1000 in the media header.
    let mdhd = mp4::find(&tree, "moov/trak/mdia/mdhd").expect("mdhd");
    let m = &bytes[mdhd.body..];
    assert_eq!(u32::from_be_bytes(m[12..16].try_into().unwrap()), 25_000);
}
