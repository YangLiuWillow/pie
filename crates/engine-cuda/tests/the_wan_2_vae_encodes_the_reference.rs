//! **THE WAN 2.2 VAE ENCODER, LOADED OUT OF THE SERVING ARTIFACT, TURNS A
//! `17 x 480 x 832` PIXEL CLIP INTO THE REFERENCE'S `5 x 30 x 52`
//! DiT-SPACE LATENT — ONE FIRE PER CHUNK, DOWN ONE SLOT, THE HEAD ARM
//! FIRST.** (design D8/D11, milestone M3)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test the_wan_2_vae_encodes_the_reference -- --nocapture
//! ```
//!
//! The golden is `scripts/imagegen/wan22_golden.py --vae`, which after
//! decoding the run's final latent puts the decode BACK IN through
//! `AutoencoderKLWan._encode` in fp32 and dumps, under
//! `$PIE_IMAGEGEN_GOLDEN/wan22/wan22_vae_encode/`: `pixels.f32`
//! (`[F·H·W, 3]` in `[-1, 1]`, the encoder's input), `mean.f32`
//! (`quant_conv`'s first 48 channels — `DiagonalGaussianDistribution`'s
//! MEAN) and `latent.f32` (the same rows normalised into the denoiser's
//! space, `(mean − latents_mean)/latents_std`), rows of voxels in
//! `(t, h, w)` order. The weights come out of
//! `$PIE_IMAGEGEN_ARTIFACTS/wan22-ti2v-5b.zt` through `own_contract` over
//! the ENCODE plan's params alone, so the load is the encoder's planes and
//! not the row's 22.5 GB. It is the artifact and not the snapshot for the
//! same reason the decode gate gives: `vae.enc.norm_scale` /
//! `vae.enc.norm_bias` are STATED rows (a `Concat` of filled cells) that
//! only `pie model import` materializes.
//!
//! **WHAT IS CLAIMED, AND WHAT IS NOT.**
//!
//! 1. *The chunked encode is the reference's encode.* Pixel frame 0 goes
//!    through `vae.encode.head` and lands ONE latent frame; every later
//!    group of FOUR frames goes through `vae.encode` and lands one more.
//!    The two arms share one slot, so each causal conv's `CacheRow::State`
//!    slab carries its last frames forward — and the two `downsample3d`
//!    time convolutions, which the reference does not run on the first
//!    chunk at all, are seeded there by `spatial.cache_store`. Gate:
//!    `cos >= 0.999` over the whole clip and per chunk.
//! 2. *The frame caches MATTER.* A later chunk is fired a second time on a
//!    FRESH slot — same pixels, every slab zeroed — and the latent must
//!    move by more than the gate's tolerance. If it does not, the state
//!    rows are not reaching the convolutions.
//! 3. *The arm answers the DENOISER's space.* The same rows are scored
//!    against the RAW posterior mean, which they must fit materially
//!    worse: the arm's own `(z − mean)/std` is the exact inverse of what
//!    `vae.decode` undoes, so a guest hands this straight to `denoise`.
//!
//! Nothing here claims a sampled posterior (the logvar's 48 channels are
//! never computed), a tiled encode, or a clip longer than the golden's.
//! Skipped by name without a device, the artifact or the golden.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use model_compiler::{Budget, VoxelLadder};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Stream, Value,
    trace_hybrid,
};
use models::wan_2::forward::Facts;
use models::wan_2::model::Model;

/// The two VAE ENCODER arms as a plan of their own: the reading bits still
/// select between them, so the trace keeps the family's own split and the
/// state slabs are the family's own `caches()`.
struct EncodeOnly {
    model: Model,
}

impl ForwardHybrid for EncodeOnly {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        self.model.caches()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let vae = self
            .model
            .vae
            .as_ref()
            .expect("the flagship carries the VAE");
        let codes = self.model.readings();
        let (top, bot) = inputs.split(&Facts::reading_top());
        let (t_hi, t_lo) = top.split(&Facts::reading_hi());
        let (b_hi, b_lo) = bot.split(&Facts::reading_hi());
        let (c7, c6) = t_hi.split(&Facts::reading_lo());
        let (c5, c4) = t_lo.split(&Facts::reading_lo());
        let (c3, c2) = b_hi.split(&Facts::reading_lo());
        let (c1, c0) = b_lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3, c4, c5, c6, c7];
        let head = codes.vae_encode_head.expect("the head arm's code");
        let rest = codes.vae_encode.expect("the later-chunks arm's code");
        let _ = models::wan_2::forward::vae_encode(&arms[usize::from(head)], &vae.enc, true);
        // A hybrid plan answers one value; the arms plant their own seams
        // and this is only what the trace hands back.
        models::wan_2::forward::vae_encode(&arms[usize::from(rest)], &vae.enc, false)
    }
}

fn artifact() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen"));
    let file = root.join("wan22-ti2v-5b.zt");
    file.is_file().then_some(file)
}

fn golden() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let dir = root.join("wan22/wan22_vae_encode");
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
        mean_abs: sum_abs / want.len().max(1) as f64,
    }
}

/// The box `shapes.json` states for one plane.
fn boxed(shapes: &serde_json::Value, key: &str) -> ([u32; 3], usize) {
    let at = &shapes[key];
    let get = |name: &str| at[name].as_u64().expect("a box extent") as u32;
    (
        [get("t"), get("h"), get("w")],
        at["channels"].as_u64().expect("channels") as usize,
    )
}

/// The word one lane of `reading` on the video stream carries.
fn word(reading: u8) -> u64 {
    Facts::of(
        &Request::new(1, false)
            .on_stream(Stream::Video)
            .in_reading(reading),
    )
    .word()
}

/// The loaded encoder, held across the fires that share its caches.
struct Encoder {
    shell: Shell,
    head: u8,
    rest: u8,
}

impl Encoder {
    /// One pixel chunk in, its latent rows and their box out.
    fn chunk(&mut self, first: bool, clip: [u32; 3], payload: &[f32]) -> (Vec<f32>, [u32; 3]) {
        let reading = if first { self.head } else { self.rest };
        let tokens = [0u32];
        let lanes = [Seated::of(Lane {
            slot: 0,
            word: word(reading),
            tokens: &tokens,
        })];
        let bytes = bf16_bytes(payload);
        let clips = [Clips {
            lane: 0,
            clips: &[clip],
            payload: &bytes,
        }];
        let mut answered = self
            .shell
            .fire_voxels(&lanes, &clips)
            .expect("the encode fire");
        assert_eq!(answered.len(), 1);
        let (values, boxes) = answered.remove(0);
        assert_eq!(boxes.len(), 1, "one clip in, one clip out");
        (values, boxes[0])
    }

    /// Zero every frame cache: `Shell::open` is what clears a slot's state
    /// rows, and it is what starts a clip.
    fn rewind(&mut self) {
        self.shell.open(0).expect("slot 0 opens");
    }
}

fn load(artifact: &PathBuf, max_voxels: u32) -> (Encoder, f64) {
    let model = Model::ti2v_5b(Dtype::Bf16, 1);
    let codes = model.readings();
    let (head, rest) = (
        codes.vae_encode_head.expect("the head arm"),
        codes.vae_encode.expect("the later arm"),
    );
    let arm = EncodeOnly { model };
    let trace = trace_hybrid("wan22-vae-encode", &arm, Platform::Cuda);
    let src = ztensor::Source::open(artifact)
        .unwrap_or_else(|why| panic!("{}: {why}", artifact.display()));
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| {
            panic!(
                "{} does not hold every plane of the VAE encoder plan: {why}",
                artifact.display()
            )
        });
    drop(src);
    let started = Instant::now();
    let mut shell = Shell::load(Boot {
        classify: |request| Facts::of(request).word(),
        trace,
        contract: &contract,
        checkpoint: artifact,
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
        deferred_tier: true,
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("the VAE encoder does not load: {why}"));
    let load_s = started.elapsed().as_secs_f64();
    shell.open(0).expect("slot 0 opens");
    (Encoder { shell, head, rest }, load_s)
}

#[test]
fn the_encoder_answers_the_reference_chunk_by_chunk() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE encode gate: no CUDA device");
        return;
    }
    let Some(root) = artifact() else {
        eprintln!("skipping the VAE encode gate: no wan22-ti2v-5b.zt artifact");
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE encode gate: no wan22_golden.py --vae encode dump");
        return;
    };
    let shapes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("shapes.json")).expect("shapes.json"))
            .expect("shapes.json parses");
    let (pixel_box, pixel_c) = boxed(&shapes, "pixels");
    let (latent_box, latent_c) = boxed(&shapes, "latent");
    let pixels = f32s(&gold.join("pixels.f32"));
    let latent = f32s(&gold.join("latent.f32"));
    let raw_mean = f32s(&gold.join("mean.f32"));
    let [frames, hp, wp] = pixel_box;
    let [t_lat, hl, wl] = latent_box;
    assert_eq!(
        frames,
        4 * t_lat - 3,
        "a Wan clip is 4T - 3 frames; the golden says otherwise"
    );
    let in_plane = (hp * wp) as usize;
    let out_plane = (hl * wl) as usize;
    assert_eq!(pixels.len(), in_plane * frames as usize * pixel_c);
    assert_eq!(latent.len(), out_plane * t_lat as usize * latent_c);
    assert_eq!(raw_mean.len(), latent.len());
    let chunks: Vec<u32> = shapes["chunks"]
        .as_array()
        .expect("the chunk boundaries")
        .iter()
        .map(|v| v.as_u64().expect("a frame index") as u32)
        .collect();
    assert_eq!(chunks.len(), t_lat as usize + 1);

    // The ladder is the INPUT clip's ceiling: the widest chunk is four
    // pixel frames, and every rectangle downstream shrinks from it.
    let widest = chunks
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize * in_plane)
        .max()
        .expect("at least one chunk");
    let (mut vae, load_s) = load(&root, widest as u32 + 8);
    eprintln!("wan vae encode: load {load_s:.1} s, {frames} pixel frames of {hp}x{wp}");

    // ---- the clip, chunk by chunk ----------------------------------------
    let mut got: Vec<f32> = Vec::with_capacity(latent.len());
    let mut per_chunk: Vec<(u32, Score)> = Vec::new();
    for k in 0..t_lat as usize {
        let first = k == 0;
        let (from, to) = (chunks[k] as usize, chunks[k + 1] as usize);
        let rows = &pixels[from * in_plane * pixel_c..to * in_plane * pixel_c];
        let started = Instant::now();
        let (out, out_box) = vae.chunk(first, [(to - from) as u32, hp, wp], rows);
        let fire_s = started.elapsed().as_secs_f64();
        assert_eq!(
            out_box,
            [1, hl, wl],
            "chunk {k} ({} pixel frames) lands ONE latent frame",
            to - from
        );
        let len = out_plane * latent_c;
        assert_eq!(out.len(), len);
        let s = score(&out, &latent[k * len..(k + 1) * len]);
        eprintln!(
            "  chunk {k} ({} arm, {} frames, {fire_s:.3} s): cos {:.6}, mean |err| {:.5}, \
             max |err| {:.4}",
            if first { "head" } else { "later" },
            to - from,
            s.cos,
            s.mean_abs,
            s.max_abs
        );
        per_chunk.push((k as u32, s));
        got.extend_from_slice(&out);
    }
    assert_eq!(got.len(), latent.len(), "the fires cover the golden's frames");

    let whole = score(&got, &latent);
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "encode {frames}x{hp}x{wp} -> {t_lat}x{hl}x{wl}: cos {:.6}, mean |err| {:.5}, \
         max |err| {:.4}, range [{lo:.3}, {hi:.3}]",
        whole.cos, whole.mean_abs, whole.max_abs
    );
    for (k, s) in &per_chunk {
        assert!(
            s.cos >= 0.999 && s.mean_abs <= 0.05,
            "chunk {k} drifts from the reference: cos {}, mean |err| {} — a chunk that alone \
             is wrong is a frame cache that did not carry",
            s.cos,
            s.mean_abs
        );
    }
    assert!(
        whole.cos >= 0.999 && whole.mean_abs <= 0.05,
        "the clip drifts from the reference: cos {}, mean |err| {}",
        whole.cos,
        whole.mean_abs
    );

    // ---- claim 3: the answer is the DENOISER's space ----------------------
    // The same rows against the RAW posterior mean. `(z − mean)/std` is a
    // per-channel affine with means as large as 0.5 and stds from 0.35 to
    // 1.7, so a port that forgot it would sit visibly further from these.
    let unnormalised = score(&got, &raw_mean);
    eprintln!(
        "against the RAW posterior mean: cos {:.6}, mean |err| {:.5}",
        unnormalised.cos, unnormalised.mean_abs
    );
    assert!(
        unnormalised.cos < whole.cos - 1e-3,
        "the arm's rows fit the raw posterior mean as well as the normalised latent \
         (cos {} against {}): `vae.encode` is meant to answer the DENOISER's space, and a \
         guest that hands this to `denoise` would be handing it the wrong numbers",
        unnormalised.cos,
        whole.cos
    );

    // ---- claim 2: the frame caches matter --------------------------------
    assert!(t_lat >= 2, "the caches can only be claimed past chunk 0");
    let k = (t_lat - 1) as usize;
    let (from, to) = (chunks[k] as usize, chunks[k + 1] as usize);
    let rows = &pixels[from * in_plane * pixel_c..to * in_plane * pixel_c];
    vae.rewind();
    let (cold, _) = vae.chunk(false, [(to - from) as u32, hp, wp], rows);
    let len = out_plane * latent_c;
    let cacheless = score(&cold, &latent[k * len..(k + 1) * len]);
    eprintln!(
        "cacheless chunk {k}: cos {:.6}, mean |err| {:.5} (the cached fire read {:.6})",
        cacheless.cos, cacheless.mean_abs, per_chunk[k].1.cos
    );
    assert!(
        cacheless.mean_abs > 0.05,
        "zeroing the frame caches moved the latent by mean |err| {}, inside the gate's own \
         tolerance: the `CacheRow::State` slabs are not reaching the causal convolutions and \
         this gate is measuring a cacheless encoder",
        cacheless.mean_abs
    );
}
