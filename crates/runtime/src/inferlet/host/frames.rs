//! pie:inferlet/frames — the pixel and sample OUTPUT resources.
//!
//! The inverse of [`super::media`], and the reason design.md D11 exists: a
//! generative pass produces megabytes of picture, and the one thing that must
//! not happen to them is a trip through WASM linear memory on the way to the
//! client. So the payload lives here, the guest holds a handle, and the
//! encoders are [`crate::codec`]'s.
//!
//! [`FrameStore`] is where that promise will be cashed. Today every handle is
//! `Host(Vec<u8>)` — `frames.from-rgb8` is the only producer, and it is fed
//! from the guest. When a `vae.decode` reading writes its pixels to the
//! device, the second variant carries the device pointer and the encoders
//! reach it without a host copy: [`crate::codec::nvenc`] would register the
//! allocation with `NvEncRegisterResource` instead of filling a system-memory
//! input buffer, and the still encoders would copy down once. Nothing outside
//! this file and that one changes, which is what the enum is for.

use crate::codec::{mp4, still, wav, y4m};
use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pie::inferlet::frames::{AudioFormat, ImageFormat};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Where a handle's pixels are.
pub enum FrameStore {
    /// Interleaved RGB8, `count * height * width * 3` bytes, frame-major.
    Host(Vec<u8>),
    /// **Not built yet.** The same pixels as a device allocation, for the
    /// path where a VAE decode writes them there and nothing brings them
    /// down: the H.264 encoder takes a `CUdeviceptr` directly, and only the
    /// still formats — one frame, already small — would copy.
    ///
    /// Held as a variant rather than added later so the shape of the store is
    /// settled while there is exactly one producer to change.
    #[allow(dead_code)]
    Device(DevicePlane),
}

/// The device half of [`FrameStore`], deliberately empty until there is a
/// producer: an allocation handle plus a pitch is what it will carry, and
/// naming those fields before the allocator is chosen would be a guess.
#[allow(dead_code)]
pub struct DevicePlane {
    _private: (),
}

/// A host-side clip: `count` frames of `width` x `height` RGB8.
pub struct Frames {
    pub store: FrameStore,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub fps: f32,
}

/// A host-side audio buffer: interleaved f32 in [-1, 1].
pub struct Pcm {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: u32,
}

// `Debug` by hand on both: a derived one would print megabytes of payload
// into a panic message, which is the opposite of what a reader wants from a
// handle whose whole point is that the payload stays out of sight.
impl std::fmt::Debug for Frames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let store = match &self.store {
            FrameStore::Host(v) => format!("host {} B", v.len()),
            FrameStore::Device(_) => "device".to_string(),
        };
        write!(
            f,
            "Frames({}x{} x{} @{} fps, {store})",
            self.width, self.height, self.count, self.fps
        )
    }
}

impl std::fmt::Debug for Pcm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pcm({} samples, {} Hz x{})",
            self.samples.len(),
            self.rate,
            self.channels
        )
    }
}

/// The CUDA ordinal NVENC opens its session on.
///
/// Device 0 unless `PIE_NVENC_DEVICE` says otherwise. A follow-up once the
/// `Device` store lands: the encoder must run on the card the pixels are on,
/// which is the engine's, not an environment variable's.
#[cfg(feature = "cuda")]
fn nvenc_device() -> usize {
    std::env::var("PIE_NVENC_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl Frames {
    /// The pixels, wherever they are. Only the host store answers today.
    pub fn rgb8(&self) -> Result<&[u8], String> {
        match &self.store {
            FrameStore::Host(v) => Ok(v),
            FrameStore::Device(_) => Err("frames: this handle's pixels are on the device and the \
                     device path is not built yet"
                .to_string()),
        }
    }

    /// **THE WAY IN**: an encoded still becomes a one-frame handle, extent
    /// and all read off the picture. The pair to [`Frames::encode`], and
    /// what lets a guest hand a picture to a `vae.encode` reading.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let (rgb, width, height) = still::decode(bytes)?;
        Ok(Frames {
            store: FrameStore::Host(rgb),
            width,
            height,
            // A still is one frame at no rate, the same shape `from-rgb8`
            // gives a caller who states `count = 1, fps = 0`.
            count: 1,
            fps: 0.0,
        })
    }

    /// These pixels as the f32 plane a `pixels` port reads: `[-1, 1]`, one
    /// row per voxel in `(t, h, w)` order with `w` fastest. Exactly undoes
    /// [`Frames::from_pixels`]'s `(x + 1) / 2`, so a handle that made the
    /// round trip through a channel comes back to the bytes it started as
    /// (up to the 8-bit quantisation the store holds).
    pub fn to_pixels(&self) -> Result<Vec<f32>, String> {
        Ok(self
            .rgb8()?
            .iter()
            .map(|b| f32::from(*b) / 255.0 * 2.0 - 1.0)
            .collect())
    }

    /// Build a handle from raw interleaved RGB8, checking the one invariant
    /// every encoder below depends on.
    pub fn from_rgb8(
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || count == 0 {
            return Err(format!(
                "frames.from-rgb8: a handle needs a non-zero extent, got \
                 {width}x{height} x{count}"
            ));
        }
        let want = (width as u64) * (height as u64) * (count as u64) * 3;
        if bytes.len() as u64 != want {
            return Err(format!(
                "frames.from-rgb8: {} bytes for {count} frames of {width}x{height} RGB8 \
                 (expected {want})",
                bytes.len()
            ));
        }
        Ok(Self {
            store: FrameStore::Host(bytes),
            width,
            height,
            count,
            fps,
        })
    }

    /// Build a handle from a `pixels` seam's plane: `count * height * width`
    /// rows of RGB f32 in the model's own `[-1, 1]`, one row per output voxel
    /// in `(t, h, w)` order — which is presentation order — mapped to RGB8 by
    /// `(x + 1) / 2` and clamped.
    ///
    /// The clamp is deliberate. A VAE's last convolution is not bounded, and
    /// the reference pipelines clamp too; refusing an overshoot would turn a
    /// picture that is right everywhere but four pixels into no picture at
    /// all.
    pub fn from_pixels(
        values: &[f32],
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || count == 0 {
            return Err(format!(
                "frames.from-channel: a handle needs a non-zero extent, got \
                 {width}x{height} x{count}"
            ));
        }
        let want = (width as u64) * (height as u64) * (count as u64) * 3;
        if values.len() as u64 != want {
            return Err(format!(
                "frames.from-channel: the cell holds {} f32 and {count} frame(s) of \
                 {width}x{height} RGB is {want}; a `pixels` seam lands one row per \
                 output voxel, three wide",
                values.len()
            ));
        }
        let bytes = values
            .iter()
            .map(|v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok(Self {
            store: FrameStore::Host(bytes),
            width,
            height,
            count,
            fps,
        })
    }

    /// The whole encoder dispatch. One place, so the refusals read the same
    /// way whichever door reached them (`frames.encode` or
    /// `session.send-frames`).
    pub fn encode(&self, format: ImageFormat) -> Result<Vec<u8>, String> {
        let rgb = self.rgb8()?;
        let still_only = |what: &str| {
            format!(
                "{what} is a still format and this handle has {} frames; \
                 use y4m or mp4-h264",
                self.count
            )
        };
        match format {
            ImageFormat::Png => {
                if self.count != 1 {
                    return Err(still_only("png"));
                }
                still::png(rgb, self.width, self.height)
            }
            ImageFormat::Jpeg => {
                if self.count != 1 {
                    return Err(still_only("jpeg"));
                }
                still::jpeg(rgb, self.width, self.height)
            }
            ImageFormat::Webp => {
                if self.count != 1 {
                    return Err(still_only("webp"));
                }
                still::webp(rgb, self.width, self.height)
            }
            ImageFormat::RawRgb8 => Ok(rgb.to_vec()),
            ImageFormat::Y4m => y4m::encode(rgb, self.width, self.height, self.count, self.fps),
            ImageFormat::Mp4H264 => self.mp4_h264(rgb),
        }
    }

    #[cfg(feature = "cuda")]
    fn mp4_h264(&self, rgb: &[u8]) -> Result<Vec<u8>, String> {
        let pictures = crate::codec::nvenc::encode_h264(
            rgb,
            self.width,
            self.height,
            self.count,
            self.fps,
            nvenc_device(),
        )?;
        mp4::mux_annexb(self.width, self.height, self.fps, &pictures)
    }

    /// The refusal a build without the CUDA shell gives. It names the missing
    /// encoder rather than the missing feature flag, because the person
    /// reading it is holding a clip, not a `Cargo.toml`.
    #[cfg(not(feature = "cuda"))]
    fn mp4_h264(&self, _rgb: &[u8]) -> Result<Vec<u8>, String> {
        let _ = &mp4::mux_annexb; // the muxer is built on every platform
        Err(
            "mp4-h264 needs the NVIDIA H.264 encoder (NVENC), and this runtime \
             was built without the CUDA shell. Encode `y4m` for an \
             uncompressed clip, or rebuild with `--features cuda`."
                .to_string(),
        )
    }
}

impl Pcm {
    pub fn from_f32(samples: Vec<f32>, rate: u32, channels: u32) -> Result<Self, String> {
        if rate == 0 || channels == 0 {
            return Err(format!(
                "pcm.from-f32: a buffer needs a rate and a channel count, got \
                 {rate} Hz x{channels}"
            ));
        }
        if samples.len() % channels as usize != 0 {
            return Err(format!(
                "pcm.from-f32: {} samples do not divide into {channels} channels",
                samples.len()
            ));
        }
        Ok(Self {
            samples,
            rate,
            channels,
        })
    }

    pub fn encode(&self, format: AudioFormat) -> Vec<u8> {
        match format {
            AudioFormat::Wav => wav::encode(&self.samples, self.rate, self.channels),
            AudioFormat::RawF32 => wav::raw_f32(&self.samples),
        }
    }
}

/// The file-name extension each format wants, so `session.send-frames` can
/// finish a name the guest left bare.
pub fn image_extension(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::Webp => "webp",
        ImageFormat::RawRgb8 => "rgb",
        ImageFormat::Y4m => "y4m",
        ImageFormat::Mp4H264 => "mp4",
    }
}

/// The same, for audio.
pub fn audio_extension(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::RawF32 => "f32",
    }
}

impl pie::inferlet::frames::Host for ProcessCtx {}

impl pie::inferlet::frames::HostFrames for ProcessCtx {
    async fn from_rgb8(
        &mut self,
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Result<Resource<Frames>, String>> {
        match Frames::from_rgb8(bytes, width, height, count, fps) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    /// The VAE road (design D8/D11): the channel's committed cell becomes a
    /// handle without the pixels entering linear memory. The take is the same
    /// one `channel.take-blocking` does — same await discipline, same poison —
    /// and the only difference is that the bytes stop here.
    async fn from_channel(
        &mut self,
        ch: Resource<super::forward::Channel>,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<Result<Resource<Frames>, String>> {
        let cell = match super::forward::materialize_channel_blocking(
            self,
            ch,
            super::forward::ChannelReadMode::Take,
        )
        .await?
        {
            Ok(bytes) => bytes,
            Err(why) => return Ok(Err(why)),
        };
        if !cell.len().is_multiple_of(4) {
            return Ok(Err(format!(
                "frames.from-channel: the cell is {} bytes, which is not whole f32 rows; \
                 a `pixels` seam's channel is f32",
                cell.len()
            )));
        }
        let values: Vec<f32> = cell
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        match Frames::from_pixels(&values, width, height, count, fps) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    /// The way in, sniffed: PNG / JPEG / GIF / WebP to a one-frame handle.
    async fn decode(&mut self, bytes: Vec<u8>) -> Result<Result<Resource<Frames>, String>> {
        match Frames::decode(&bytes) {
            Ok(f) => Ok(Ok(self.ctx().table.push(f)?)),
            Err(why) => Ok(Err(why)),
        }
    }

    /// The VAE road run backwards: these pixels into `ch`'s cell as the f32
    /// plane a pixel port reads, without the bytes entering linear memory.
    /// The put is `channel.set`'s, so the cell is SEEDED and the channel
    /// must be one a pass binds as an input.
    async fn to_channel(
        &mut self,
        this: Resource<Frames>,
        ch: Resource<super::forward::Channel>,
    ) -> Result<Result<(), String>> {
        let values = match self.ctx().table.get(&this)?.to_pixels() {
            Ok(values) => values,
            Err(why) => return Ok(Err(why)),
        };
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // `put`, not `set`: a picture handed to a `vae.encode` port has to be
        // there for the FIRST fire, and `set` rewrites a cell already in the
        // ring — a channel has no ring until a fire has run. `put` before the
        // first fire is the seed.
        let cell = self.ctx().table.get(&ch)?.cell.clone();
        let result = cell
            .lock()
            .unwrap()
            .put_ref(&bytes)
            .map_err(|error| format!("frames.to-channel: {error}"));
        Ok(result)
    }

    async fn width(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.width)
    }

    async fn height(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.height)
    }

    async fn count(&mut self, this: Resource<Frames>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.count)
    }

    async fn fps(&mut self, this: Resource<Frames>) -> Result<f32> {
        Ok(self.ctx().table.get(&this)?.fps)
    }

    async fn encode(
        &mut self,
        this: Resource<Frames>,
        format: ImageFormat,
    ) -> Result<Result<Vec<u8>, String>> {
        Ok(self.ctx().table.get(&this)?.encode(format))
    }

    async fn drop(&mut self, this: Resource<Frames>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::frames::HostPcm for ProcessCtx {
    async fn from_f32(
        &mut self,
        samples: Vec<f32>,
        rate: u32,
        channels: u32,
    ) -> Result<Result<Resource<Pcm>, String>> {
        match Pcm::from_f32(samples, rate, channels) {
            Ok(p) => Ok(Ok(self.ctx().table.push(p)?)),
            Err(e) => Ok(Err(e)),
        }
    }

    async fn rate(&mut self, this: Resource<Pcm>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.rate)
    }

    async fn channels(&mut self, this: Resource<Pcm>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.channels)
    }

    async fn encode(&mut self, this: Resource<Pcm>, format: AudioFormat) -> Result<Vec<u8>> {
        Ok(self.ctx().table.get(&this)?.encode(format))
    }

    async fn drop(&mut self, this: Resource<Pcm>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
