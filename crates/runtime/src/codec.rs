//! Encoders for what a generative pass produces: pixels and samples.
//!
//! The counterpart of [`crate::inferlet::host::media::decode`], which turns a
//! client's bytes into something a model reads. This turns what a model wrote
//! into bytes a client can open, and it is the runtime's job rather than the
//! guest's for the reason design.md D11 gives: the payload never enters WASM
//! linear memory, so the encoder has to live where the payload does.
//!
//! Five of the six image formats are pure Rust and unconditional:
//!
//! | format | who | notes |
//! |---|---|---|
//! | `png` / `jpeg` / `webp` | the `image` crate | stills only (`count == 1`) |
//! | `raw-rgb8` | [`still`] | the bytes as held, no header |
//! | `y4m` | [`y4m`] | uncompressed I420 clip — the parity format |
//! | `mp4-h264` | [`nvenc`] + [`mp4`] | NVENC, then an in-tree ISO-BMFF mux |
//!
//! The sixth is the only one that needs hardware. [`nvenc`] dlopens
//! `libnvidia-encode.so.1` the way cudarc dlopens every CUDA library — no
//! link-time dependency, no `cc`, no ffmpeg — and is compiled only under the
//! crate's `cuda` feature, since opening an encode session needs a CUDA
//! context. Without that feature `mp4-h264` refuses by name and says which
//! half is missing. [`mp4`] is unconditional: the muxer is pure Rust and its
//! gates run on every machine.

pub mod color;
pub mod mp4;
#[cfg(feature = "cuda")]
pub mod nvenc;
pub mod still;
pub mod wav;
pub mod y4m;
