//! **THE Z-IMAGE VAE'S TWO READINGS TRACE AND BAKE ON THE VOXEL AXIS, AND
//! THE IMPORT READS EVERY `vae.` TENSOR OF THE REAL SNAPSHOT ONCE.**
//! (design D8, milestone M1)
//!
//! ```text
//! cargo test -p models --test the_z_image_vae_bakes
//! ```
//!
//! ```text
//! (a) the flagship declares `vae.decode` and `vae.encode` after its three
//!     token readings, each a token-less, kv-less `Image` lane with one
//!     `Voxels` port (16 wide in, 3 wide in) and a `pixels` readout (3
//!     wide out, 16 wide out); the miniature declares neither
//! (b) the trace reads exactly those two voxel ports and plants `pixels`
//!     twice — the decoder's `[VoxelsTimes(64), 3]` plane (three nearest
//!     ×2 upsamples) and the encoder's `[Voxels, 16]` mean — each beside
//!     its `[Clips, 4]` grid
//! (c) the shapes of the FLUX VAE: 2 whole-row attentions at 512, 3
//!     upsamples, 3 stride-2 convolutions padded `(0, 1, 0, 1)` behind the
//!     box, every conv weight interned tap-major at its own `c_in`, every
//!     GroupNorm 32 groups at 1e-6 (SiLU fused on all but the attention's)
//! (d) each VAE lane classifies into its own class, apart from the token
//!     readings'
//! (e) the plan bakes on CUDA against a voxel ladder sized for one 64x64
//!     latent (its 512x512 image), the voxel regions their own units
//! (f) over the real `Tongyi-MAI/Z-Image-Turbo` snapshot (skipped by name
//!     when the HuggingFace cache holds none): every one of the 244 `vae.`
//!     tensors is read exactly once, every conv kernel as a transmute of
//!     its own bytes, and every plane type-checks
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{Expr, Partition, TensorType};
use model_dsl::{Classify, Def, Dim, Dtype, Operation, Platform, Request, Stream, Trace, Ty, seam};
use model_ir::{GridRule, ParamLayout, Seam, Spatial};
use models::z_image::forward::Facts;
use models::z_image::{model, vae};
use models::{PortKind, ReadoutKind};

const TURBO: &str = "z-image-turbo-bf16-kv-bf16";
const MINI: &str = "z-image-mini-bf16-kv-bf16";

fn row(sku: &str) -> &'static models::Sku {
    models::sku(sku).unwrap_or_else(|| panic!("this build ships no `{sku}`"))
}

fn trace(sku: &str) -> Trace {
    (row(sku).trace)(Platform::Cuda)
}

fn reading<'a>(facts: &'a models::Generative, name: &str) -> &'a models::ReadingFact {
    facts
        .readings
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no reading `{name}`"))
}

/// (a)
#[test]
fn the_flagship_declares_the_two_vae_readings_and_the_miniature_neither() {
    let facts = row(TURBO).generative.as_ref().expect("facts");
    let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
    assert_eq!(
        names,
        vec!["text", "refine", "denoise", "vae.decode", "vae.encode"]
    );
    let decode = reading(facts, "vae.decode");
    let encode = reading(facts, "vae.encode");
    for r in [decode, encode] {
        assert!(!r.has_kv && !r.takes_tokens, "a voxel lane");
        assert_eq!(r.streams, vec![Stream::Image]);
        assert_eq!(r.readout, ReadoutKind::Pixels);
        assert_eq!(r.ports.len(), 1);
        assert_eq!(r.ports[0].kind, PortKind::Voxels);
        assert_eq!(r.ports[0].streams, vec![Stream::Image]);
    }
    assert_eq!(
        (
            decode.ports[0].name,
            decode.ports[0].width,
            decode.readout_width
        ),
        ("latent", model::CHANNELS, vae::RGB)
    );
    assert_eq!(
        (
            encode.ports[0].name,
            encode.ports[0].width,
            encode.readout_width
        ),
        ("pixels", vae::RGB, model::CHANNELS)
    );
    assert_eq!(decode.port("latent").map(|(index, _)| index), Some(0));
    // Its STATED index, not its position: `vae.encode`'s only voxel
    // port is read at index 1 so `vae.decode`'s 16-wide latent keeps 0.
    assert_eq!(
        encode.port("pixels").map(|(index, _)| index),
        Some(model::port::PIXEL_VOXELS)
    );
    assert_eq!(decode.index, 3);
    assert_eq!(encode.index, 4);

    let mini = row(MINI).generative.as_ref().expect("facts");
    assert!(
        !mini.readings.iter().any(|r| r.name.starts_with("vae.")),
        "the miniature has no VAE"
    );
    assert!(
        !trace(MINI)
            .values
            .iter()
            .any(|v| matches!(v.def, Def::Input(model_dsl::RuntimeInput::Voxels { .. }))),
        "and reads no voxel port"
    );
}

/// (b)
#[test]
fn the_trace_reads_two_voxel_ports_and_plants_pixels_twice() {
    let plan = trace(TURBO);
    let mut ports: Vec<(u8, u32, String)> = plan
        .values
        .iter()
        .filter_map(|v| match (&v.def, &v.ty) {
            (
                Def::Input(model_dsl::RuntimeInput::Voxels { port, channels }),
                Ty::Tensor { dtype, .. },
            ) => Some((*port, *channels, format!("{dtype:?}"))),
            _ => None,
        })
        .collect();
    ports.sort_unstable();
    // Two voxel INDICES, not one: the engine seats one rectangle per
    // `(kind, index)` for the whole plan, so the 16-wide latent clip and the
    // 3-wide pixel clip cannot share index 0.
    assert_eq!(
        ports,
        vec![
            (model::port::VOXELS, model::CHANNELS, "Bf16".to_string()),
            (model::port::PIXEL_VOXELS, vae::RGB, "Bf16".to_string())
        ]
    );
    let pixels: Vec<&Seam> = plan
        .seams
        .iter()
        .filter(|s| s.seam == seam::PIXELS.name)
        .collect();
    assert_eq!(pixels.len(), 2, "one pixels seam per VAE reading");
    let mut planes: Vec<Ty> = pixels
        .iter()
        .map(|s| {
            assert_eq!(s.values.len(), 2, "the plane and its grid");
            assert_eq!(
                plan.values[s.values[1].0 as usize].ty,
                Ty::Tensor {
                    shape: vec![Dim::Clips, Dim::Const(4)],
                    dtype: Dtype::I32
                }
            );
            plan.values[s.values[0].0 as usize].ty.clone()
        })
        .collect();
    planes.sort_by_key(|ty| format!("{ty:?}"));
    let plane = |rows: Dim, width: u64| Ty::Tensor {
        shape: vec![rows, Dim::Const(width)],
        dtype: Dtype::Bf16,
    };
    let mut want = vec![
        plane(Dim::VoxelsTimes(64), u64::from(vae::RGB)),
        plane(Dim::Voxels, u64::from(model::CHANNELS)),
    ];
    want.sort_by_key(|ty| format!("{ty:?}"));
    assert_eq!(planes, want);
    assert!(
        plan.seams
            .iter()
            .filter(|s| s.seam == seam::VELOCITY.name)
            .count()
            == 1
            && !plan.seams.iter().any(|s| s.seam == seam::OUT.name),
        "the token readings' seams are untouched"
    );
}

/// (c)
#[test]
fn the_shapes_are_the_flux_vaes() {
    let plan = trace(TURBO);
    let (mut attentions, mut upsamples, mut strided, mut convs, mut norms) = (0, 0, 0, 0, 0);
    let mut silu_off = 0;
    for node in &plan.nodes {
        let Operation::Spatial(op) = &node.op else {
            continue;
        };
        match op {
            Spatial::Attention { sm_scale, q, .. } => {
                attentions += 1;
                assert_eq!(
                    plan.values[q.0 as usize].ty.clone(),
                    Ty::Tensor {
                        shape: vec![Dim::Voxels, Dim::Const(512)],
                        dtype: Dtype::Bf16
                    }
                );
                assert!((sm_scale - 512f32.sqrt().recip()).abs() < 1e-9);
            }
            Spatial::UpsampleNearest {
                factor,
                keep_first_frame,
                ..
            } => {
                upsamples += 1;
                assert_eq!((*factor, *keep_first_frame), ([1, 2, 2], false));
            }
            Spatial::Conv3d {
                x,
                w,
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                cache,
                ..
            } => {
                convs += 1;
                assert!(!causal_t && cache.is_none() && k[0] == 1 && stride[0] == 1);
                let c_in = match &plan.values[x.0 as usize].ty {
                    Ty::Tensor { shape, .. } => match shape[1] {
                        Dim::Const(c) => c as u32,
                        other => panic!("{other:?}"),
                    },
                    other => panic!("{other:?}"),
                };
                let Def::Weight(at) = plan.values[w.0 as usize].def else {
                    panic!("a conv reads a weight")
                };
                let param = &plan.params[at as usize];
                assert_eq!(
                    param.layout,
                    ParamLayout::ConvTapsMajor {
                        c_in,
                        taps: k[1] * k[2]
                    },
                    "`{}`",
                    param.name
                );
                if stride[1] == 2 {
                    strided += 1;
                    assert_eq!((*k, *pad, *pad_back), ([1, 3, 3], [0; 3], [0, 1, 1]));
                } else {
                    assert_eq!(pad, pad_back);
                    assert!(matches!((k[1], pad[1]), (3, 1) | (1, 0)));
                }
            }
            Spatial::GroupNorm {
                groups, eps, silu, ..
            } => {
                norms += 1;
                assert_eq!((*groups, *eps), (vae::GN_GROUPS, vae::GN_EPS));
                if !silu {
                    silu_off += 1;
                }
            }
            Spatial::Grid { rule, .. } => {
                if let GridRule::Conv { pad, pad_back, .. } = rule {
                    assert!(pad == pad_back || *pad_back == [0, 1, 1]);
                }
            }
            _ => {}
        }
    }
    assert_eq!(attentions, 2, "one mid-block attention per reading");
    assert_eq!(upsamples, 3);
    assert_eq!(strided, 3);
    // Decoder: conv_in, 2 mid resnets x2, 12 resnets x2 + 2 shortcuts, 3
    // upsamplers, conv_out = 1+4+24+2+3+1 = 35. Encoder: conv_in, 8 resnets
    // x2 + 2 shortcuts, 3 downsamplers, 2 mid resnets x2, conv_out =
    // 1+16+2+3+4+1 = 27.
    assert_eq!(convs, 62);
    // Two norms per resnet (14 + 10 resnets), one per attention, one out
    // norm per side: 48 + 2 + 2.
    assert_eq!(norms, 52);
    assert_eq!(silu_off, 2, "only the attention's norm has no SiLU");
}

/// (d)
#[test]
fn each_vae_lane_has_a_class_of_its_own() {
    let plan = trace(TURBO);
    let classes = model_dsl::resolve_classes(&plan).expect("every merge resolves");
    let facts = row(TURBO).generative.as_ref().expect("facts");
    let class_of = |name: &str, stream: Stream| {
        let r = reading(facts, name);
        let request = Request::new(4, false).on_stream(stream).in_reading(r.index);
        let word = Facts::of(&request).word();
        classes
            .class_of(word & classes.mask)
            .unwrap_or_else(|| panic!("a {name} lane has no class"))
    };
    let seen: BTreeSet<usize> = [
        class_of("text", Stream::Text),
        class_of("refine", Stream::Context),
        class_of("denoise", Stream::Image),
        class_of("denoise", Stream::Context),
        class_of("vae.decode", Stream::Image),
        class_of("vae.encode", Stream::Image),
    ]
    .into_iter()
    .collect();
    assert_eq!(seen.len(), 6, "six lanes, six classes");
}

/// (e)
#[test]
fn the_plan_bakes_against_a_voxel_ladder() {
    let plan = trace(TURBO);
    let budget = model_compiler::Budget {
        max_lanes: 16,
        max_tokens: 4096,
        buckets: vec![1024, 4096],
        max_adapters: 0,
    };
    // One 64x64 latent (a 512x512 image) per fire.
    let budgets = model_compiler::Budgets::of(budget)
        .with_voxels(model_compiler::VoxelLadder::new(64 * 64, 1));
    let compiled =
        model_compiler::compile_axes(&plan, &budgets, &model_compiler::DeviceProfile::default())
            .unwrap_or_else(|why| panic!("does not bake: {why}"));
    assert!(compiled.voxels.is_some(), "the voxel axis has its own plan");
    let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
    assert_eq!(tiled, plan.nodes.len(), "the regions tile the nodes once");
    assert!(
        !model_compiler::compile(
            &plan,
            &budgets.tokens,
            &model_compiler::DeviceProfile::default()
        )
        .is_ok(),
        "and against no voxel ladder the plan is refused, not sized at zero"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// (f) the real snapshot
// ─────────────────────────────────────────────────────────────────────────

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("vae/config.json").is_file())
}

struct Types<'a>(&'a ztensor::Source);

impl CheckpointTypes for Types<'_> {
    fn tensor_type(&self, name: &str) -> Option<TensorType> {
        let tensor = self.0.get(name)?;
        let encoding = checkpoint::file::encoding_of(&tensor).ok()?;
        Some(TensorType {
            shape: tensor.shape().iter().map(|&n| n as i64).collect(),
            encoding,
        })
    }
}

#[test]
fn the_import_reads_every_vae_tensor_of_the_real_snapshot_once() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Tongyi-MAI/Z-Image-Turbo snapshot in the HuggingFace cache");
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let contract = row(TURBO)
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the flagship does not read this snapshot: {why}"));

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for source in tensor.expr.sources() {
            *counts.entry(source.to_string()).or_default() += 1;
        }
    }
    let index: BTreeSet<String> = src
        .names()
        .filter(|n| n.starts_with("vae."))
        .map(str::to_string)
        .collect();
    assert_eq!(index.len(), 244);
    let read: BTreeSet<String> = counts
        .keys()
        .filter(|n| n.starts_with("vae."))
        .cloned()
        .collect();
    assert_eq!(read, index, "every `vae.` tensor, and only those");
    assert!(
        counts
            .iter()
            .filter(|(n, _)| n.starts_with("vae."))
            .all(|(_, c)| *c == 1),
        "each exactly once"
    );

    // Every conv kernel is a transmute of its own bytes into the natural
    // `[C_out, C_in·k·k]` rectangle; every plane types to its extents.
    let types = Types(&src);
    let mut resolver = Resolver::new(&types, Partition::WHOLE);
    let mut kernels = 0;
    for tensor in &contract.tensors {
        if tensor.name.starts_with("vae.") && tensor.name.ends_with("conv_in") {
            kernels += 1;
        }
        if let Expr::Transmute { .. } = &tensor.expr {
            assert!(tensor.name.starts_with("vae."), "`{}`", tensor.name);
        }
        // The encoder's `conv_out` declares the mean's 16 output rows of the
        // stored 32: a slice of the transmuted kernel, and of its bias.
        // (the bias through an internal `.head` step under its root cast).
        if tensor.name == "vae.enc.conv_out" || tensor.name == "vae.enc.conv_out.bias.head" {
            assert!(
                format!("{:?}", tensor.expr).contains("Slice"),
                "`{}` is sliced down to the mean's rows",
                tensor.name
            );
        }
        if tensor.name.starts_with("vae.enc.conv_out") {
            assert_eq!(
                tensor.shape.as_ref().map(|s| s[0]),
                Some(16),
                "`{}`",
                tensor.name
            );
        }
        let ty = resolver
            .infer(&tensor.expr, &tensor.name)
            .unwrap_or_else(|why| panic!("`{}` does not type: {why}", tensor.name));
        if let Some(shape) = &tensor.shape {
            assert_eq!(&ty.shape, shape, "`{}`", tensor.name);
        }
        assert_eq!(ty.encoding, tensor.encoding, "`{}`", tensor.name);
        resolver.publish(&tensor.name, ty);
    }
    assert_eq!(kernels, 2, "one conv_in per side");
    let transmuted = contract
        .tensors
        .iter()
        .filter(|t| matches!(&t.expr, Expr::Transmute { .. }))
        .count();
    assert_eq!(
        transmuted, 61,
        "every conv kernel but the sliced encoder head, and nothing else"
    );
}
