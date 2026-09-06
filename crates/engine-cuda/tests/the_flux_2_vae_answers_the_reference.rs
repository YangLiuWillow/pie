//! **FLUX.2'S AUTOENCODER, LOADED OUT OF THE ROW'S OWN IMPORTED ARTIFACT,
//! DECODES A 32x32 TOKEN GRID INTO THE REFERENCE'S 512x512 PIXELS AND
//! ENCODES THOSE PIXELS BACK INTO THE REFERENCE'S NORMALISED POSTERIOR
//! MEAN.** (design D8, milestone M2 parity)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test the_flux_2_vae_answers_the_reference -- --nocapture
//! ```
//!
//! The golden is `scripts/imagegen/flux2_golden.py --vae`:
//! `AutoencoderKLFlux2` in fp32 over the centre 32x32 tokens of the full
//! run's final packed latent — `latent.f32` (`[32·32, 128]`, DiT space at
//! `/16`), `pixels.f32` (`[512·512, 3]` in `[-1, 1]`) and `mean.f32`
//! (`[32·32, 128]`, the BatchNorm-normalised posterior mean of those
//! pixels), rows of voxels in `(h, w)` order under
//! `$PIE_IMAGEGEN_GOLDEN/flux2/flux2_vae/`. Both sides cut at the 128-wide
//! `/16` grid, which is where `models::flux_2::vae` puts its port: the 2x2
//! pixel shuffle and the frozen `BatchNorm2d(128)` are INSIDE the plan, so
//! this gate covers them too.
//!
//! The weights come from the row's IMPORTED ARTIFACT
//! (`$PIE_IMAGEGEN_ARTIFACTS/flux2-klein-4b.zt`, default
//! `/root/.cache/pie-imagegen/flux2-klein-4b.zt`), read plane-by-plane
//! through `checkpoint_dsl::own_contract` — the same identity contract a
//! server builds for a stamped artifact — and not from the HuggingFace
//! snapshot directly. That is forced, not chosen: the family's BatchNorm
//! planes are `√(var + eps)` and its reciprocal, and a SERVING load may
//! not apply a `Unary` on the way in (`checkpoint::plan::passes::validate`
//! — the function is of the checkpoint's values, so the artifact holds its
//! answer). So the gate runs against what `pie model import` wrote, which
//! makes it an end-to-end check of the import as well as of the arms:
//!
//! ```text
//! pie model import <the FLUX.2-klein-4B snapshot>
//!   --sku flux2-klein-4b-bf16-kv-bf16
//!   --out ~/.cache/pie-imagegen/flux2-klein-4b.zt --force
//! ```
//!
//! Only the VAE's own planes are read (the trace of one arm names no
//! `dit.`/`te.` param), so a 14 GB artifact costs a few hundred megabytes
//! here. Each reading is its own plan and load — the decoder against a
//! 1 032-voxel ladder, the encoder against a 262 152-voxel one — so the
//! arena is sized for the reading it serves.
//!
//! Gates (bf16 activations against an fp32 reference): decode `cos ≥
//! 0.999`, `mean |err| ≤ 0.005` and at most one value in ten thousand past
//! 0.05 (`max |err| ≤ 0.2`) on pixels in `[-1, 1]` — measured: cos
//! 0.999994, mean 0.0017, max 0.040, not one value past 0.05, the worst
//! scattered in the picture's interior and not on its border (a border
//! run would be a padding fault); encode `cos ≥ 0.9995`, `mean |err| ≤
//! 0.02` and `max |err| ≤ 0.5` on the normalised mean (range `[-6.1,
//! 6.9]`) — measured: cos 0.999957, mean 0.0069, max 0.115 at two
//! interior voxels. Both are tighter than Z-Image's same gate: this VAE's
//! latent is BatchNorm-normalised rather than scaled by one factor, so
//! the mean's rows sit in a narrower range and lose less to bf16. Skipped
//! by name without a device, the artifact or the golden.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use model_compiler::{Budget, VoxelLadder};
use model_dsl::{Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Value, trace_hybrid};
use models::flux_2::forward::Facts;
use models::flux_2::model::{IN_CHANNELS, Model};
use models::flux_2::vae;

/// One VAE reading as a plan of its own: the whole input is that arm.
struct OneArm {
    model: Model,
    decode: bool,
}

impl ForwardHybrid for OneArm {
    type Facts = Facts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<Facts>) -> Value {
        let v = self
            .model
            .vae
            .as_ref()
            .expect("the flagship carries the VAE");
        if self.decode {
            vae::decode(&inputs, v)
        } else {
            vae::encode(&inputs, v)
        }
    }
}

/// The row's imported artifact — what `pie model import` wrote.
fn artifact() -> Option<PathBuf> {
    let path = std::env::var_os("PIE_IMAGEGEN_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen"))
        .join("flux2-klein-4b.zt");
    path.is_file().then_some(path)
}

fn golden() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let dir = root.join("flux2/flux2_vae");
    dir.join("shapes.json").is_file().then_some(dir)
}

fn f32s(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| {
            let bits = v.to_bits();
            let rounding = 0x7fff + ((bits >> 16) & 1);
            (((bits + rounding) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

struct Score {
    cos: f64,
    max_abs: f64,
    mean_abs: f64,
}

fn score(got: &[f32], want: &[f32]) -> Score {
    assert_eq!(got.len(), want.len(), "one value per reference value");
    let (mut dot, mut gg, mut ww, mut max_abs, mut sum_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (f64::from(*g), f64::from(*w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        let err = (g - w).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    Score {
        cos: dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
        max_abs,
        mean_abs: sum_abs / want.len() as f64,
    }
}

/// Load one reading of the VAE out of the artifact and fire one clip through it.
fn fire(
    root: &PathBuf,
    decode: bool,
    max_voxels: u32,
    clip: [u32; 3],
    payload: &[f32],
) -> (Vec<f32>, Vec<[u32; 3]>, f64, f64) {
    let model = Model::klein_4b(Dtype::Bf16, 1);
    let arm = OneArm { model, decode };
    let trace = trace_hybrid(
        if decode {
            "flux2-vae-decode"
        } else {
            "flux2-vae-encode"
        },
        &arm,
        Platform::Cuda,
    );
    // The artifact's own contract, over the params of THIS arm alone: the
    // planes the other arm and the transformer hold are never named, so
    // never read.
    let src = ztensor::Source::open(root).unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the artifact does not hold this arm's planes: {why}"));
    drop(src);
    let contract = &contract;
    let word =
        Facts::of(&model_dsl::Request::new(1, false).on_stream(model_dsl::Stream::Image)).word();
    let started = Instant::now();
    let mut shell = Shell::load(Boot {
        classify: |request| Facts::of(request).word(),
        trace,
        contract,
        checkpoint: root,
        budget: Budget::new(2, 16),
        patches: None,
        voxels: Some(VoxelLadder::new(max_voxels, 2)),
        profile: None,
        page_size: 16,
        context: 64,
        slots: 2,
        pages: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs {
            recording: Recording::Off,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| {
        panic!(
            "the VAE {} does not load: {why}",
            if decode { "decoder" } else { "encoder" }
        )
    });
    let load_s = started.elapsed().as_secs_f64();
    shell.open(0).expect("slot 0 opens");
    let tokens = [0u32];
    let lanes = [Seated::of(Lane {
        slot: 0,
        word,
        tokens: &tokens,
    })];
    let bytes = bf16_bytes(payload);
    let clips = [Clips {
        lane: 0,
        clips: &[clip],
        payload: &bytes,
    }];
    // Once to warm the JIT, once for the clock.
    let _ = shell.fire_voxels(&lanes, &clips).expect("the first fire");
    let started = Instant::now();
    let mut answered = shell.fire_voxels(&lanes, &clips).expect("the second fire");
    let fire_s = started.elapsed().as_secs_f64();
    assert_eq!(answered.len(), 1);
    let (values, boxes) = answered.remove(0);
    drop(shell);
    (values, boxes, load_s, fire_s)
}

#[test]
fn the_vae_decodes_and_encodes_the_golden_clip() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE parity gate: no CUDA device");
        return;
    }
    let Some(root) = artifact() else {
        eprintln!(
            "skipping the VAE parity gate: no flux2-klein-4b.zt (run `pie model import` first)"
        );
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE parity gate: no flux2_golden.py --vae dump");
        return;
    };
    // The golden's clips are square stills: the box is read off each
    // plane's length (`shapes.json` beside them says the same).
    let latent = f32s(&gold.join("latent.f32"));
    let pixels = f32s(&gold.join("pixels.f32"));
    let mean = f32s(&gold.join("mean.f32"));
    let square = |len: usize, channels: usize| -> [u32; 3] {
        let side = ((len / channels) as f64).sqrt().round() as u32;
        assert_eq!(
            side as usize * side as usize * channels,
            len,
            "a square still"
        );
        [1, side, side]
    };
    let width = IN_CHANNELS as usize;
    let latent_box = square(latent.len(), width);
    let pixel_box = square(pixels.len(), 3);
    let mean_box = square(mean.len(), width);
    let voxels = |b: [u32; 3]| (b[0] * b[1] * b[2]) as usize;
    assert_eq!(latent.len(), voxels(latent_box) * width);
    assert_eq!(pixels.len(), voxels(pixel_box) * 3);
    assert_eq!(mean.len(), voxels(mean_box) * width);
    assert_eq!(
        pixel_box[1],
        latent_box[1] * models::flux_2::model::TOKEN_COMPRESSION,
        "one token is 16 pixels a side"
    );

    // ---- decode ----------------------------------------------------------
    let (got, boxes, load_s, fire_s) = fire(
        &root,
        true,
        voxels(latent_box) as u32 + 8,
        latent_box,
        &latent,
    );
    assert_eq!(boxes, vec![pixel_box], "the clip comes back at 16x");
    let s = score(&got, &pixels);
    // Where the worst pixels are: a border-only error would be a padding
    // bug, a scattered one is bf16 through sixty convolutions.
    {
        let side = pixel_box[1] as usize;
        let mut worst: Vec<(f32, usize, usize, usize)> = got
            .iter()
            .zip(&pixels)
            .enumerate()
            .map(|(i, (g, w))| ((g - w).abs(), (i / 3) / side, (i / 3) % side, i % 3))
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let over = worst.iter().filter(|w| w.0 > 0.05).count();
        eprintln!(
            "decode: {over} of {} values past 0.05; worst (err, y, x, c): {:?}",
            got.len(),
            &worst[..8]
        );
        assert!(
            over * 10_000 <= got.len(),
            "{over} values past 0.05 is more than one in ten thousand"
        );
    }
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "decode {latent_box:?} -> {pixel_box:?}: load {load_s:.1} s, fire {fire_s:.3} s, cos {:.6}, \
         max |err| {:.4}, mean |err| {:.5}, range [{lo:.3}, {hi:.3}]",
        s.cos, s.max_abs, s.mean_abs
    );
    assert!(
        s.cos >= 0.999 && s.mean_abs <= 0.005 && s.max_abs <= 0.2,
        "the decode drifts from the reference: cos {}, mean |err| {}, max |err| {}",
        s.cos,
        s.mean_abs,
        s.max_abs
    );

    // ---- encode ----------------------------------------------------------
    let (got, boxes, load_s, fire_s) = fire(
        &root,
        false,
        voxels(pixel_box) as u32 + 8,
        pixel_box,
        &pixels,
    );
    assert_eq!(boxes, vec![mean_box], "the clip comes back at 1/16");
    let s = score(&got, &mean);
    {
        let side = mean_box[1] as usize;
        let mut worst: Vec<(f32, usize, usize, usize)> = got
            .iter()
            .zip(&mean)
            .enumerate()
            .map(|(i, (g, w))| {
                (
                    (g - w).abs(),
                    (i / width) / side,
                    (i / width) % side,
                    i % width,
                )
            })
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let over = worst.iter().filter(|w| w.0 > 0.1).count();
        let edge = worst
            .iter()
            .filter(|w| w.0 > 0.1)
            .filter(|w| w.1 == 0 || w.2 == 0 || w.1 + 1 == side || w.2 + 1 == side)
            .count();
        let (lo, hi) = mean
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        eprintln!(
            "encode: {over} of {} values past 0.1 ({edge} on the border); reference range \
             [{lo:.3}, {hi:.3}]; worst (err, y, x, c): {:?}",
            got.len(),
            &worst[..8]
        );
    }
    eprintln!(
        "encode {pixel_box:?} -> {mean_box:?}: load {load_s:.1} s, fire {fire_s:.3} s, cos {:.6}, \
         max |err| {:.4}, mean |err| {:.5}",
        s.cos, s.max_abs, s.mean_abs
    );
    assert!(
        s.cos >= 0.9995 && s.mean_abs <= 0.02 && s.max_abs <= 0.5,
        "the encode drifts from the reference: cos {}, mean |err| {}, max |err| {}",
        s.cos,
        s.mean_abs,
        s.max_abs
    );
}
