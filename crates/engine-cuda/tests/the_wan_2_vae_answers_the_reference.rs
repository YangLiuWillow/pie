//! **THE WAN 2.2 VAE DECODER, LOADED OUT OF THE SERVING ARTIFACT, TURNS A
//! `T x 30 x 52` DiT-SPACE LATENT INTO THE
//! REFERENCE'S `(4T - 3) x 480 x 832` PIXELS — ONE FIRE PER LATENT FRAME,
//! DOWN ONE SLOT, THE HEAD ARM FIRST.** (design D8/D11, milestone M3)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test the_wan_2_vae_answers_the_reference -- --nocapture
//! ```
//!
//! The golden is `scripts/imagegen/wan22_golden.py --vae`: the fp32
//! `AutoencoderKLWan` over `latent.final` of the full 480x832x17 run —
//! `latent.f32` (`[T*30*52, 48]`, the DENOISER's space), `denorm.f32` (the
//! same times `latents_std` plus `latents_mean`, which is what the
//! reference hands its decoder) and `pixels.f32` (`[(4T-3)*480*832, 3]` in
//! `[-1, 1]`), rows of voxels in `(t, h, w)` order under
//! `$PIE_IMAGEGEN_GOLDEN/wan22/wan22_vae/`. The weights come out of
//! `$PIE_IMAGEGEN_ARTIFACTS/wan22-ti2v-5b.zt` — the artifact serving
//! reads — through `own_contract` over the VAE plan's params alone, so the
//! load is the decoder's ~1.4 GB and not the row's 22.5. It is the artifact
//! and not the snapshot on purpose: the denormalisation rows this arm
//! carries are STATED (a `Concat` of filled cells) and materialize on the
//! host at `pie model import`; a CUDA serving plan cannot lower one.
//!
//! **WHAT IS CLAIMED, AND WHAT IS NOT.**
//!
//! 1. *The chunked decode is the reference's decode.* Latent frame 0 goes
//!    through `vae.decode.head` and lands ONE output frame; every later
//!    frame goes through `vae.decode` and lands FOUR; the two arms share
//!    one slot so each causal conv's `CacheRow::State` slab carries its
//!    last frames forward, which is what `feat_cache` does on the other
//!    side. Gate: `cos >= 0.999`, `mean |err| <= 0.02` over the whole clip,
//!    and the SAME gate per chunk, so a fire that read a stale cache says
//!    which one it was. MEASURED, 5 latent frames of 30x52 into 17 frames
//!    of 480x832: cos 0.999986, mean |err| 0.00244, max |err| 0.256, with
//!    every chunk between 0.999984 and 0.999988 — full pixel parity, mid
//!    block included (`spatial::attention_over(.., VoxelSegment::Frames(1),
//!    ..)`), not a decoder-minus-attention.
//! 2. *The frame caches MATTER.* The later-frames arm is fired a second
//!    time on a FRESH slot — the same latent frame, the same everything,
//!    but with every slab zeroed instead of carrying frame `k-1` — and the
//!    pixels must move by more than the gate's tolerance. If they do not,
//!    the state rows are not reaching the convolutions and a green cosine
//!    would be measuring a cacheless decoder that happens to be close.
//!    MEASURED: cos 0.9807, mean |err| 0.1250 — fifty times the gate.
//! 3. *The head arm is not the later arm.* Firing the LATER arm on frame 0
//!    lands four frames, not one; that is not a parity claim, it is the
//!    claim that the two arms are actually two arms.
//!
//! Nothing here claims the ENCODER (untraced, `AvgDown3D` has no `Spatial`
//! member) or a T > 5 clip (the golden's own length). Skipped by name
//! without a device, the artifact or the golden.

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

/// The two VAE arms as a plan of their own: the reading bits still select
/// between them, so the trace keeps the family's own split and the state
/// slabs are the family's own `caches()`.
struct VaeOnly {
    model: Model,
}

impl ForwardHybrid for VaeOnly {
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
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let head = codes.vae_decode_head.expect("the head arm's code");
        let rest = codes.vae_decode.expect("the later-frames arm's code");
        let _ = models::wan_2::forward::vae_decode(&arms[usize::from(head)], vae, true);
        // A hybrid plan answers one value; the arms plant their own seams
        // and this is only what the trace hands back.
        models::wan_2::forward::vae_decode(&arms[usize::from(rest)], vae, false)
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
    let dir = root.join("wan22/wan22_vae");
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

/// The loaded decoder, held across the fires that share its caches.
struct Decoder {
    shell: Shell,
    head: u8,
    rest: u8,
}

impl Decoder {
    /// One latent frame in, its pixels and their box out.
    fn frame(&mut self, first: bool, clip: [u32; 3], payload: &[f32]) -> (Vec<f32>, [u32; 3]) {
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
            .expect("the decode fire");
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

fn load(artifact: &PathBuf, max_voxels: u32) -> (Decoder, f64) {
    let model = Model::ti2v_5b(Dtype::Bf16, 1);
    let codes = model.readings();
    let (head, rest) = (
        codes.vae_decode_head.expect("the head arm"),
        codes.vae_decode.expect("the later arm"),
    );
    let arm = VaeOnly { model };
    let trace = trace_hybrid("wan22-vae-decode", &arm, Platform::Cuda);
    // **THE ARTIFACT, NOT THE SNAPSHOT.** This row's `vae.denorm_scale` /
    // `vae.denorm_bias` are STATED rows — a config vector reaches a plan as
    // 48 filled cells concatenated — and a `Concat` over computed buffers
    // lowers to a `Reblock` tile map, which a CUDA serving plan does not
    // carry (`checkpoint::plan::passes::tile::CUDA_TILE_MAP_MASK`). Those
    // rows are materialized once, on the host, by `pie model import`; what
    // serving reads is the `.zt`'s own plane of that name, which is exactly
    // what `own_contract` asks for. So this gate loads what serving loads,
    // and takes the VAE's planes out of it by naming only the VAE plan's
    // params.
    let src = ztensor::Source::open(artifact)
        .unwrap_or_else(|why| panic!("{}: {why}", artifact.display()));
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Cuda)
        .unwrap_or_else(|why| {
            panic!(
                "{} does not hold every plane of the VAE plan: {why}",
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
    .unwrap_or_else(|why| panic!("the VAE decoder does not load: {why}"));
    let load_s = started.elapsed().as_secs_f64();
    shell.open(0).expect("slot 0 opens");
    (Decoder { shell, head, rest }, load_s)
}

#[test]
fn the_decoder_answers_the_reference_frame_by_frame() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the VAE parity gate: no CUDA device");
        return;
    }
    let Some(root) = artifact() else {
        eprintln!("skipping the VAE parity gate: no wan22-ti2v-5b.zt artifact");
        return;
    };
    let Some(gold) = golden() else {
        eprintln!("skipping the VAE parity gate: no wan22_golden.py --vae dump");
        return;
    };
    let shapes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("shapes.json")).expect("shapes.json"))
            .expect("shapes.json parses");
    let (latent_box, latent_c) = boxed(&shapes, "latent");
    let (pixel_box, pixel_c) = boxed(&shapes, "pixels");
    let latent = f32s(&gold.join("latent.f32"));
    let pixels = f32s(&gold.join("pixels.f32"));
    let [frames, hp, wp] = pixel_box;
    let [t_lat, hl, wl] = latent_box;
    assert_eq!(
        frames,
        4 * t_lat - 3,
        "a Wan clip is 4T - 3 frames; the golden says otherwise"
    );
    let plane = (hl * wl) as usize;
    let out_plane = (hp * wp) as usize;
    assert_eq!(latent.len(), plane * t_lat as usize * latent_c);
    assert_eq!(pixels.len(), out_plane * frames as usize * pixel_c);

    // The ladder is the INPUT clip's ceiling, not the output's: the
    // compiler walks each `Spatial` op's grid rule to size the values a
    // decode grows into. One latent frame is `hl * wl` voxels.
    let (mut vae, load_s) = load(&root, plane as u32 + 8);
    eprintln!("wan vae: load {load_s:.1} s, {t_lat} latent frames of {hl}x{wl}");

    // ---- the clip, fire by fire ------------------------------------------
    let mut got: Vec<f32> = Vec::with_capacity(pixels.len());
    let mut per_chunk: Vec<(u32, Score)> = Vec::new();
    let mut at = 0usize;
    for k in 0..t_lat {
        let first = k == 0;
        let rows = &latent[(k as usize) * plane * latent_c..(k as usize + 1) * plane * latent_c];
        let started = Instant::now();
        let (out, out_box) = vae.frame(first, [1, hl, wl], rows);
        let fire_s = started.elapsed().as_secs_f64();
        let want_frames = if first { 1 } else { 4 };
        assert_eq!(
            out_box,
            [want_frames, hp, wp],
            "latent frame {k} lands {want_frames} output frames"
        );
        let len = want_frames as usize * out_plane * pixel_c;
        assert_eq!(out.len(), len);
        let s = score(&out, &pixels[at..at + len]);
        eprintln!(
            "  frame {k} ({} arm, {fire_s:.3} s): cos {:.6}, mean |err| {:.5}, max |err| {:.4}",
            if first { "head" } else { "later" },
            s.cos,
            s.mean_abs,
            s.max_abs
        );
        per_chunk.push((k, s));
        got.extend_from_slice(&out);
        at += len;
    }
    assert_eq!(at, pixels.len(), "the fires cover the reference's frames");

    let whole = score(&got, &pixels);
    let (lo, hi) = got
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    eprintln!(
        "decode {t_lat}x{hl}x{wl} -> {frames}x{hp}x{wp}: cos {:.6}, mean |err| {:.5}, \
         max |err| {:.4}, range [{lo:.3}, {hi:.3}]",
        whole.cos, whole.mean_abs, whole.max_abs
    );
    for (k, s) in &per_chunk {
        assert!(
            s.cos >= 0.999 && s.mean_abs <= 0.02,
            "latent frame {k} drifts from the reference: cos {}, mean |err| {} — a chunk \
             that alone is wrong is a frame cache that did not carry",
            s.cos,
            s.mean_abs
        );
    }
    assert!(
        whole.cos >= 0.999 && whole.mean_abs <= 0.02,
        "the clip drifts from the reference: cos {}, mean |err| {}",
        whole.cos,
        whole.mean_abs
    );

    // ---- claim 2: the frame caches matter --------------------------------
    // The same later-frames arm on the same latent frame, but from a slot
    // whose slabs were just zeroed. If that lands the same pixels, the
    // state rows are not reaching the convolutions.
    assert!(t_lat >= 2, "the caches can only be claimed past frame 0");
    let k = (t_lat - 1) as usize;
    let rows = &latent[k * plane * latent_c..(k + 1) * plane * latent_c];
    vae.rewind();
    let (cold, _) = vae.frame(false, [1, hl, wl], rows);
    let warm_at = 1 + 4 * (k - 1);
    let warm = &pixels[warm_at * out_plane * pixel_c..(warm_at + 4) * out_plane * pixel_c];
    let cacheless = score(&cold, warm);
    eprintln!(
        "cacheless frame {k}: cos {:.6}, mean |err| {:.5} (the cached fire read {:.6})",
        cacheless.cos, cacheless.mean_abs, per_chunk[k].1.cos
    );
    assert!(
        cacheless.mean_abs > 0.02,
        "zeroing the frame caches moved the pixels by mean |err| {}, inside the gate's own \
         tolerance: the `CacheRow::State` slabs are not reaching the causal convolutions and \
         this gate is measuring a cacheless decoder",
        cacheless.mean_abs
    );

    // ---- claim 3: the two arms are two arms ------------------------------
    vae.rewind();
    let (_, later_on_zero) = vae.frame(false, [1, hl, wl], &latent[..plane * latent_c]);
    assert_eq!(
        later_on_zero,
        [4, hp, wp],
        "the later-frames arm lands four frames whatever it is fed; only the head arm lands one"
    );
}
