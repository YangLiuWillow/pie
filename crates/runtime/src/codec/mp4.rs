//! A pure-Rust ISO base media file muxer, exactly wide enough for one H.264
//! video track — and the box walker that reads one back.
//!
//! WHY IN-TREE RATHER THAN A CRATE. design.md D11 allows either. The published
//! `mp4`-class crates carry `serde`, `serde_json`, `num-rational` and a
//! `thiserror` a major version behind this workspace's, which is four extra
//! graph entries and one duplicated crate to write eleven boxes whose layout
//! has not changed since 2001. The boxes are below, in about the space the
//! dependency's justification would have taken.
//!
//! WHAT IT WRITES. `ftyp` / `mdat` / `moov`, in that order — `mdat` first so
//! every sample offset is known by the time `stco` is written, which is what
//! lets the whole file be produced in one pass with no seeking and no
//! placeholder patching beyond each box's own length.
//!
//!   * one track, handler `vide`, sample entry `avc1` with an `avcC` built
//!     from the encoder's own SPS/PPS;
//!   * one chunk holding every sample, so `stsc` is a single run;
//!   * `stts` a single run, because the encoder is driven at a fixed rate;
//!   * `stss` listing the IDR samples, omitted when every sample is one.
//!
//! WHAT IT DOES NOT WRITE, deliberately: no `ctts`. The encoder is configured
//! with `frameIntervalP = 1` (IPP, no B-frames), so decode order is
//! presentation order and composition offsets are all zero. A muxer that
//! writes no `ctts` and an encoder that emits reordered frames would produce a
//! file that plays backwards in places, so the two decisions are one decision
//! and [`super::nvenc`] states the other half.
//!
//! Media timescale is the frame rate's numerator and every sample's delta is
//! its denominator (see [`super::y4m::frame_rate`]), so 23.976 fps is exact
//! rather than rounded.

use super::y4m::frame_rate;

/// Movie timescale: milliseconds. Independent of the media timescale, which
/// tracks the frame rate.
const MOVIE_TIMESCALE: u32 = 1000;

// -- writing -----------------------------------------------------------------

/// A byte sink that knows how to open a box, write into it, and go back and
/// fill in its length.
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Start a box, returning the position of its length field.
    fn open(&mut self, kind: &[u8; 4]) -> usize {
        let at = self.buf.len();
        self.buf.extend_from_slice(&[0, 0, 0, 0]);
        self.buf.extend_from_slice(kind);
        at
    }

    /// Finish the box opened at `at`.
    fn close(&mut self, at: usize) {
        let size = (self.buf.len() - at) as u32;
        self.buf[at..at + 4].copy_from_slice(&size.to_be_bytes());
    }

    /// A FullBox's version byte and 24-bit flags, as one word.
    fn full(&mut self, version: u8, flags: u32) {
        self.u32(((version as u32) << 24) | (flags & 0x00ff_ffff));
    }

    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }

    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// The identity transform, as every `tkhd`/`mvhd` on earth writes it.
    fn unity_matrix(&mut self) {
        for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            self.u32(v);
        }
    }
}

/// One coded picture, as the muxer wants it: length-prefixed NAL units and
/// whether it is a random access point.
#[derive(Debug, Clone)]
pub struct Sample {
    /// NAL units, each behind a 4-byte big-endian length (the `avcC` form).
    pub avcc: Vec<u8>,
    /// True for an IDR picture — a `stss` entry.
    pub is_sync: bool,
}

/// Everything the muxer needs that is not a sample.
#[derive(Debug, Clone)]
pub struct Track {
    pub width: u32,
    pub height: u32,
    /// Media timescale (ticks per second).
    pub timescale: u32,
    /// Ticks each sample occupies.
    pub sample_delta: u32,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

/// Build the `avcC` payload (everything after the box header) from the
/// sequence and picture parameter sets the encoder produced.
fn avcc_payload(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>, String> {
    if sps.len() < 4 {
        return Err(format!("avcC: SPS is {} bytes, need at least 4", sps.len()));
    }
    if pps.is_empty() {
        return Err("avcC: no PPS".to_string());
    }
    let mut v = Vec::with_capacity(11 + sps.len() + pps.len());
    v.push(1); // configurationVersion
    v.push(sps[1]); // AVCProfileIndication
    v.push(sps[2]); // profile_compatibility
    v.push(sps[3]); // AVCLevelIndication
    v.push(0xff); // 6 bits reserved | lengthSizeMinusOne = 3
    v.push(0xe1); // 3 bits reserved | numOfSequenceParameterSets = 1
    v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    v.extend_from_slice(sps);
    v.push(1); // numOfPictureParameterSets
    v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    v.extend_from_slice(pps);
    Ok(v)
}

/// Mux one H.264 track into a complete `.mp4`.
pub fn mux(track: &Track, samples: &[Sample]) -> Result<Vec<u8>, String> {
    if samples.is_empty() {
        return Err("mp4: no samples to mux".to_string());
    }
    if track.timescale == 0 || track.sample_delta == 0 {
        return Err(format!(
            "mp4: degenerate timing (timescale {}, delta {})",
            track.timescale, track.sample_delta
        ));
    }
    // `tkhd` carries the size as 16.16 fixed point and `avc1` as a u16, so
    // 65535 is the container's own ceiling, not this muxer's shortcut.
    if track.width > u16::MAX as u32 || track.height > u16::MAX as u32 {
        return Err(format!(
            "mp4: {}x{} exceeds the container's 16-bit frame size",
            track.width, track.height
        ));
    }
    let avcc = avcc_payload(&track.sps, &track.pps)?;

    let mut w = Writer::new();

    // -- ftyp
    let b = w.open(b"ftyp");
    w.bytes(b"isom");
    w.u32(512);
    for brand in [b"isom", b"iso2", b"avc1", b"mp41"] {
        w.bytes(brand);
    }
    w.close(b);

    // -- mdat. Written before moov so `stco` needs no back-patching beyond
    //    this box's own length.
    let mdat = w.open(b"mdat");
    let data_offset = w.buf.len();
    let mut sizes = Vec::with_capacity(samples.len());
    for s in samples {
        sizes.push(s.avcc.len() as u32);
        w.bytes(&s.avcc);
    }
    w.close(mdat);

    let n = samples.len() as u32;
    let media_duration = (n as u64) * track.sample_delta as u64;
    let movie_duration = media_duration * MOVIE_TIMESCALE as u64 / track.timescale.max(1) as u64;

    // -- moov
    let moov = w.open(b"moov");

    let mvhd = w.open(b"mvhd");
    w.full(0, 0);
    w.u32(0); // creation_time
    w.u32(0); // modification_time
    w.u32(MOVIE_TIMESCALE);
    w.u32(movie_duration.min(u32::MAX as u64) as u32);
    w.u32(0x0001_0000); // rate 1.0
    w.u16(0x0100); // volume 1.0
    w.u16(0); // reserved
    w.zeros(8); // reserved
    w.unity_matrix();
    w.zeros(24); // pre_defined
    w.u32(2); // next_track_ID
    w.close(mvhd);

    let trak = w.open(b"trak");

    let tkhd = w.open(b"tkhd");
    w.full(0, 0x7); // enabled | in movie | in preview
    w.u32(0); // creation_time
    w.u32(0); // modification_time
    w.u32(1); // track_ID
    w.u32(0); // reserved
    w.u32(movie_duration.min(u32::MAX as u64) as u32);
    w.zeros(8); // reserved
    w.u16(0); // layer
    w.u16(0); // alternate_group
    w.u16(0); // volume (0 for video)
    w.u16(0); // reserved
    w.unity_matrix();
    w.u32(track.width << 16); // 16.16 fixed point
    w.u32(track.height << 16);
    w.close(tkhd);

    let mdia = w.open(b"mdia");

    let mdhd = w.open(b"mdhd");
    w.full(0, 0);
    w.u32(0);
    w.u32(0);
    w.u32(track.timescale);
    w.u32(media_duration.min(u32::MAX as u64) as u32);
    w.u16(0x55c4); // language "und", packed 5-bit ISO-639-2
    w.u16(0); // pre_defined
    w.close(mdhd);

    let hdlr = w.open(b"hdlr");
    w.full(0, 0);
    w.u32(0); // pre_defined
    w.bytes(b"vide");
    w.zeros(12); // reserved
    w.bytes(b"pie video\0");
    w.close(hdlr);

    let minf = w.open(b"minf");

    let vmhd = w.open(b"vmhd");
    w.full(0, 1); // the flag every vmhd sets
    w.u16(0); // graphicsmode
    w.zeros(6); // opcolor
    w.close(vmhd);

    let dinf = w.open(b"dinf");
    let dref = w.open(b"dref");
    w.full(0, 0);
    w.u32(1); // entry_count
    let url = w.open(b"url ");
    w.full(0, 1); // "the media is in this same file"
    w.close(url);
    w.close(dref);
    w.close(dinf);

    let stbl = w.open(b"stbl");

    let stsd = w.open(b"stsd");
    w.full(0, 0);
    w.u32(1); // entry_count
    let avc1 = w.open(b"avc1");
    w.zeros(6); // reserved
    w.u16(1); // data_reference_index
    w.u16(0); // pre_defined
    w.u16(0); // reserved
    w.zeros(12); // pre_defined[3]
    w.u16(track.width as u16);
    w.u16(track.height as u16);
    w.u32(0x0048_0000); // horizresolution 72 dpi
    w.u32(0x0048_0000); // vertresolution 72 dpi
    w.u32(0); // reserved
    w.u16(1); // frame_count
    // compressorname: one length byte then 31 bytes of padded name.
    let name = b"pie NVENC H.264";
    w.bytes(&[name.len() as u8]);
    w.bytes(name);
    w.zeros(31 - name.len());
    w.u16(0x0018); // depth = 24-bit colour
    w.u16(0xffff); // pre_defined = -1
    let avcc_box = w.open(b"avcC");
    w.bytes(&avcc);
    w.close(avcc_box);
    w.close(avc1);
    w.close(stsd);

    let stts = w.open(b"stts");
    w.full(0, 0);
    w.u32(1); // entry_count
    w.u32(n); // sample_count
    w.u32(track.sample_delta);
    w.close(stts);

    // `stss` is absent when every sample is a sync sample — its absence is
    // what "all samples are random access points" means, and writing it out
    // in full would say the same thing at n words a clip.
    if !samples.iter().all(|s| s.is_sync) {
        let stss = w.open(b"stss");
        w.full(0, 0);
        let syncs: Vec<u32> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_sync)
            .map(|(i, _)| i as u32 + 1) // sample numbers are 1-based
            .collect();
        w.u32(syncs.len() as u32);
        for s in syncs {
            w.u32(s);
        }
        w.close(stss);
    }

    let stsc = w.open(b"stsc");
    w.full(0, 0);
    w.u32(1); // entry_count
    w.u32(1); // first_chunk
    w.u32(n); // samples_per_chunk — one chunk holds them all
    w.u32(1); // sample_description_index
    w.close(stsc);

    let stsz = w.open(b"stsz");
    w.full(0, 0);
    w.u32(0); // sample_size 0 => the table below
    w.u32(n);
    for s in &sizes {
        w.u32(*s);
    }
    w.close(stsz);

    let stco = w.open(b"stco");
    w.full(0, 0);
    w.u32(1); // entry_count
    w.u32(data_offset as u32);
    w.close(stco);

    w.close(stbl);
    w.close(minf);
    w.close(mdia);
    w.close(trak);
    w.close(moov);

    Ok(w.buf)
}

// -- Annex-B -> the muxer's sample form ---------------------------------------

/// Split an Annex-B byte stream into its NAL units, dropping the start codes.
pub fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        // The next NAL's start code begins 3 or 4 bytes before its payload;
        // the trailing zero belongs to the start code, not to this NAL.
        let mut e = starts.get(k + 1).map_or(data.len(), |&n| n - 3);
        while e > s && data[e - 1] == 0 {
            e -= 1;
        }
        if e > s {
            out.push(&data[s..e]);
        }
    }
    out
}

/// Turn one picture's Annex-B bytes into a [`Sample`], hoisting any parameter
/// sets it carries into `sps` / `pps`.
///
/// Parameter sets and access-unit delimiters are LIFTED OUT of the sample
/// rather than carried in it: the parameter sets belong in `avcC` (a decoder
/// that finds them only in-band cannot configure itself before the first
/// sample), and an AUD in an `avc1` sample is at best ignored.
pub fn sample_from_annexb(
    annexb: &[u8],
    sps: &mut Option<Vec<u8>>,
    pps: &mut Option<Vec<u8>>,
) -> Sample {
    let mut avcc = Vec::with_capacity(annexb.len());
    let mut is_sync = false;
    for nal in nal_units(annexb) {
        match nal[0] & 0x1f {
            7 => {
                if sps.is_none() {
                    *sps = Some(nal.to_vec());
                }
            }
            8 => {
                if pps.is_none() {
                    *pps = Some(nal.to_vec());
                }
            }
            9 | 12 => {} // access unit delimiter, filler
            kind => {
                if kind == 5 {
                    is_sync = true;
                }
                avcc.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                avcc.extend_from_slice(nal);
            }
        }
    }
    Sample { avcc, is_sync }
}

/// The whole path: per-picture Annex-B in, a complete `.mp4` out.
pub fn mux_annexb(
    width: u32,
    height: u32,
    fps: f32,
    pictures: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let mut sps = None;
    let mut pps = None;
    let samples: Vec<Sample> = pictures
        .iter()
        .map(|p| sample_from_annexb(p, &mut sps, &mut pps))
        .collect();
    let (num, den) = frame_rate(fps);
    let track = Track {
        width,
        height,
        timescale: num,
        sample_delta: den,
        sps: sps.ok_or("mp4: the stream carries no SPS")?,
        pps: pps.ok_or("mp4: the stream carries no PPS")?,
    };
    mux(&track, &samples)
}

// -- reading back -------------------------------------------------------------

/// One box, as [`parse_boxes`] found it.
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: [u8; 4],
    /// Offset of the box header within the file.
    pub offset: usize,
    /// Total size including the header.
    pub size: usize,
    /// Offset of the box's payload (past the header, and past whatever fixed
    /// preamble a container-with-a-header carries).
    pub body: usize,
    pub children: Vec<Node>,
}

impl Node {
    /// The box's four-character name.
    pub fn name(&self) -> &str {
        std::str::from_utf8(&self.kind).unwrap_or("????")
    }
}

/// How many bytes of a container box's payload are its own fields rather than
/// child boxes. `None` for a box that has no children at all.
fn container_preamble(kind: &[u8; 4]) -> Option<usize> {
    match kind {
        b"moov" | b"trak" | b"edts" | b"mdia" | b"minf" | b"dinf" | b"stbl" | b"udta" | b"mvex"
        | b"moof" | b"traf" => Some(0),
        // FullBox header + entry_count, then the sample entries.
        b"stsd" | b"dref" => Some(8),
        // VisualSampleEntry's fixed 78 bytes, then `avcC` and friends.
        b"avc1" | b"avc3" | b"hvc1" | b"hev1" => Some(78),
        _ => None,
    }
}

/// Walk a file (or one box's payload) into a tree. Errors on a length that
/// does not fit, which is the only structural lie a well-formed reader has to
/// catch.
pub fn parse_boxes(data: &[u8]) -> Result<Vec<Node>, String> {
    parse_at(data, 0, data.len())
}

fn parse_at(data: &[u8], from: usize, to: usize) -> Result<Vec<Node>, String> {
    let mut out = Vec::new();
    let mut at = from;
    while at + 8 <= to {
        let size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let mut kind = [0u8; 4];
        kind.copy_from_slice(&data[at + 4..at + 8]);
        // size 0 means "to the end of the file"; size 1 means a 64-bit length
        // follows. Neither is written here, but a reader that silently
        // mis-parses them is worse than one that says so.
        let size = match size {
            0 => to - at,
            1 => {
                if at + 16 > to {
                    return Err(format!(
                        "box '{}' claims a 64-bit size it cannot hold",
                        String::from_utf8_lossy(&kind)
                    ));
                }
                u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize
            }
            n => n,
        };
        if size < 8 || at + size > to {
            return Err(format!(
                "box '{}' at {at} claims {size} bytes, {} remain",
                String::from_utf8_lossy(&kind),
                to - at
            ));
        }
        let body = at + 8;
        let children = match container_preamble(&kind) {
            Some(pre) if body + pre <= at + size => parse_at(data, body + pre, at + size)?,
            _ => Vec::new(),
        };
        out.push(Node {
            kind,
            offset: at,
            size,
            body,
            children,
        });
        at += size;
    }
    Ok(out)
}

/// Look one box up by a `/`-separated path of four-character names, e.g.
/// `"moov/trak/mdia/minf/stbl/stsd/avc1/avcC"`.
pub fn find<'a>(nodes: &'a [Node], path: &str) -> Option<&'a Node> {
    let mut here: &'a [Node] = nodes;
    let mut found: Option<&'a Node> = None;
    for step in path.split('/') {
        let node = here.iter().find(|n| n.name() == step)?;
        here = &node.children;
        found = Some(node);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A syntactically plausible SPS/PPS/IDR/P stream. The NAL payloads are
    /// not decodable video — this gates the CONTAINER, and the container has
    /// no opinion about what a slice says.
    fn synthetic(pictures: usize) -> Vec<Vec<u8>> {
        let sps = [0x67u8, 0x64, 0x00, 0x28, 0xac, 0xd9, 0x40, 0x78];
        let pps = [0x68u8, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];
        (0..pictures)
            .map(|i| {
                let mut v = Vec::new();
                v.extend_from_slice(&[0, 0, 0, 1, 9, 0x10]); // AUD, to be dropped
                if i == 0 {
                    v.extend_from_slice(&[0, 0, 0, 1]);
                    v.extend_from_slice(&sps);
                    v.extend_from_slice(&[0, 0, 0, 1]);
                    v.extend_from_slice(&pps);
                }
                v.extend_from_slice(&[0, 0, 0, 1]);
                // 5 = IDR on the first picture, 1 = non-IDR after it.
                v.push(if i == 0 { 0x65 } else { 0x41 });
                // Never a trailing zero: a NAL may not end in one (that byte
                // belongs to the next start code), and `nal_units` trims it.
                v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, i as u8 + 1]);
                v
            })
            .collect()
    }

    #[test]
    fn nal_units_ignore_start_code_length() {
        let s = [0u8, 0, 0, 1, 0xaa, 0xbb, 0, 0, 1, 0xcc];
        let n = nal_units(&s);
        assert_eq!(n.len(), 2);
        assert_eq!(n[0], &[0xaa, 0xbb]);
        assert_eq!(n[1], &[0xcc]);
    }

    #[test]
    fn parameter_sets_leave_the_sample_and_land_in_avcc() {
        let pics = synthetic(3);
        let mp4 = mux_annexb(64, 64, 25.0, &pics).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");
        let avcc = find(&tree, "moov/trak/mdia/minf/stbl/stsd/avc1/avcC").expect("avcC");
        let payload = &mp4[avcc.body..avcc.offset + avcc.size];
        assert_eq!(payload[0], 1, "configurationVersion");
        assert_eq!(payload[1], 0x64, "profile from SPS[1]");
        assert_eq!(payload[3], 0x28, "level from SPS[3]");
        assert_eq!(payload[4], 0xff, "lengthSizeMinusOne = 3");

        // The samples themselves carry only the slices: 4 bytes of length
        // plus a 6-byte NAL each, with the AUD and the parameter sets gone.
        let stsz = find(&tree, "moov/trak/mdia/minf/stbl/stsz").expect("stsz");
        let body = &mp4[stsz.body..];
        assert_eq!(
            u32::from_be_bytes(body[4..8].try_into().unwrap()),
            0,
            "sample_size table"
        );
        assert_eq!(
            u32::from_be_bytes(body[8..12].try_into().unwrap()),
            3,
            "sample_count"
        );
        for i in 0..3 {
            let at = 12 + i * 4;
            assert_eq!(u32::from_be_bytes(body[at..at + 4].try_into().unwrap()), 10);
        }
    }

    #[test]
    fn every_box_the_structure_needs_is_present_and_well_sized() {
        let mp4 = mux_annexb(64, 48, 30.0, &synthetic(8)).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");
        assert_eq!(tree[0].name(), "ftyp");
        assert_eq!(tree[1].name(), "mdat");
        assert_eq!(tree[2].name(), "moov");
        // The walk consumed the file exactly: no slack, no overlap.
        assert_eq!(tree.iter().map(|n| n.size).sum::<usize>(), mp4.len());
        for path in [
            "moov/mvhd",
            "moov/trak/tkhd",
            "moov/trak/mdia/mdhd",
            "moov/trak/mdia/hdlr",
            "moov/trak/mdia/minf/vmhd",
            "moov/trak/mdia/minf/dinf/dref",
            "moov/trak/mdia/minf/stbl/stsd/avc1/avcC",
            "moov/trak/mdia/minf/stbl/stts",
            "moov/trak/mdia/minf/stbl/stss",
            "moov/trak/mdia/minf/stbl/stsc",
            "moov/trak/mdia/minf/stbl/stsz",
            "moov/trak/mdia/minf/stbl/stco",
        ] {
            assert!(find(&tree, path).is_some(), "missing {path}");
        }
        // Only the first picture is an IDR, so `stss` names exactly it.
        let stss = find(&tree, "moov/trak/mdia/minf/stbl/stss").unwrap();
        let body = &mp4[stss.body..];
        assert_eq!(u32::from_be_bytes(body[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(body[8..12].try_into().unwrap()), 1);
    }

    #[test]
    fn stco_points_at_the_first_sample_and_stts_states_the_rate() {
        let pics = synthetic(4);
        let mp4 = mux_annexb(64, 64, 23.976, &pics).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");

        let mdat = tree.iter().find(|n| n.name() == "mdat").unwrap();
        let stco = find(&tree, "moov/trak/mdia/minf/stbl/stco").unwrap();
        let offset = u32::from_be_bytes(mp4[stco.body + 8..stco.body + 12].try_into().unwrap());
        assert_eq!(offset as usize, mdat.body, "chunk offset is mdat's payload");

        // 23.976 fps is 23976/1000 exactly, not 24 and not 23.
        let stts = find(&tree, "moov/trak/mdia/minf/stbl/stts").unwrap();
        let b = &mp4[stts.body..];
        assert_eq!(
            u32::from_be_bytes(b[4..8].try_into().unwrap()),
            1,
            "one run"
        );
        assert_eq!(
            u32::from_be_bytes(b[8..12].try_into().unwrap()),
            4,
            "four samples"
        );
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            1000,
            "delta"
        );
        let mdhd = find(&tree, "moov/trak/mdia/mdhd").unwrap();
        let m = &mp4[mdhd.body..];
        assert_eq!(
            u32::from_be_bytes(m[12..16].try_into().unwrap()),
            23_976,
            "timescale"
        );
    }

    #[test]
    fn a_stream_without_parameter_sets_is_refused_by_name() {
        let err = mux_annexb(16, 16, 25.0, &[vec![0, 0, 0, 1, 0x41, 0x00]]).unwrap_err();
        assert!(err.contains("SPS"), "{err}");
    }

    #[test]
    fn a_truncated_box_length_is_caught_rather_than_walked_past() {
        let mut mp4 = mux_annexb(16, 16, 25.0, &synthetic(1)).expect("mux");
        let n = mp4.len();
        mp4[0..4].copy_from_slice(&((n + 16) as u32).to_be_bytes());
        assert!(parse_boxes(&mp4).is_err());
    }
}
