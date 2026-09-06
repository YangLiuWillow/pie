//! **THE FLUX.2 IMPORT READS EVERY `dit.` TENSOR OF THE KLEIN-4B SNAPSHOT,
//! THE 27 ENCODER LAYERS IT RUNS, THE VAE'S DECODER SIDE AND NOTHING OF ITS
//! ENCODER — AND EVERY PLANE IT DECLARES TYPES TO THE EXTENTS THE PLAN
//! STATES.**
//!
//! ```text
//! cargo test -p models --test the_flux_2_import_reads_the_klein_snapshot
//! ```
//!
//! A diffusers pipeline opens as one prefixed name space
//! (`checkpoint::file::diffusers`): `dit.<transformer>`, `te.<text_encoder>`,
//! `vae.<vae>`. This test builds the two rows' contracts and checks them
//! from both ends — what they read against the index, and what they declare
//! against the contract type checker and the load-plan compiler:
//!
//! ```text
//! (a) over a SYNTHETIC source shaped like `flux2_golden.py --mini`'s
//!     state_dict (fp32, no bytes): the miniature reads every tensor, each
//!     once per slice it is cut into; every plane types
//!     to its extents and lowers to a load plan; the same under `dit.`
//! (b) over the REAL `black-forest-labs/FLUX.2-klein-4B` snapshot, when the
//!     HuggingFace cache holds one: every `dit.` tensor is read;
//!     `te.model.layers.{0..27}` and the embedding are read, the last nine
//!     layers and the final norm are not; `vae.decoder.*`,
//!     `vae.post_quant_conv.*` and `vae.bn.running_*` are read and
//!     `vae.encoder.*`, `vae.quant_conv.*` are not; the contract types and
//!     lowers; identification lands on the flagship
//! (c) over the golden's own `flux2_mini.safetensors`, when present: the
//!     miniature reads it, and the BatchNorm-free, guidance-carrying
//!     transformer types
//! ```
//!
//! (b) and (c) are skipped by name where their files are absent.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{ModelContract, Partition, TensorType};
use checkpoint::plan::StorageTarget;
use model_dsl::Platform;
use models::flux_2::model::{self, Dims};
use ztensor::Leaf;
use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

const KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";
const MINI: &str = "flux2-mini-bf16-kv-bf16";

/// One tensor of a checkpoint: its name, its shape, its element.
type Named = (String, Vec<u64>, Leaf);

/// The transformer's `state_dict`, as `flux2_golden.py --mini` and the
/// snapshot's header both spell it, at one [`Dims`], one element type.
fn transformer(d: &Dims, leaf: Leaf) -> Vec<Named> {
    let dim = u64::from(d.dim);
    let hd = u64::from(model::HEAD_DIM);
    let inter = u64::from(d.inter);
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: &str, shape: Vec<u64>| out.push((name.to_string(), shape, leaf));
    push(
        "x_embedder.weight",
        vec![dim, u64::from(model::IN_CHANNELS)],
    );
    push(
        "context_embedder.weight",
        vec![dim, u64::from(d.context_in)],
    );
    push(
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        vec![dim, u64::from(model::T_FREQ_DIM)],
    );
    push(
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        vec![dim, dim],
    );
    if d.guidance_embeds {
        push(
            "time_guidance_embed.guidance_embedder.linear_1.weight",
            vec![dim, u64::from(model::T_FREQ_DIM)],
        );
        push(
            "time_guidance_embed.guidance_embedder.linear_2.weight",
            vec![dim, dim],
        );
    }
    push(
        "double_stream_modulation_img.linear.weight",
        vec![6 * dim, dim],
    );
    push(
        "double_stream_modulation_txt.linear.weight",
        vec![6 * dim, dim],
    );
    push("single_stream_modulation.linear.weight", vec![3 * dim, dim]);
    push("norm_out.linear.weight", vec![2 * dim, dim]);
    push("proj_out.weight", vec![u64::from(model::IN_CHANNELS), dim]);
    for i in 0..d.double_blocks {
        let n = |s: &str| format!("transformer_blocks.{i}.{s}");
        for proj in [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            push(&n(&format!("attn.{proj}.weight")), vec![dim, dim]);
        }
        for norm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            push(&n(&format!("attn.{norm}.weight")), vec![hd]);
        }
        for ff in ["ff", "ff_context"] {
            push(&n(&format!("{ff}.linear_in.weight")), vec![2 * inter, dim]);
            push(&n(&format!("{ff}.linear_out.weight")), vec![dim, inter]);
        }
    }
    for i in 0..d.single_blocks {
        let n = |s: &str| format!("single_transformer_blocks.{i}.attn.{s}");
        push(&n("to_qkv_mlp_proj.weight"), vec![3 * dim + 2 * inter, dim]);
        push(&n("to_out.weight"), vec![dim, dim + inter]);
        push(&n("norm_q.weight"), vec![hd]);
        push(&n("norm_k.weight"), vec![hd]);
    }
    out
}

fn prefixed(prefix: &str, tensors: Vec<Named>) -> Vec<Named> {
    tensors
        .into_iter()
        .map(|(name, shape, leaf)| (format!("{prefix}{name}"), shape, leaf))
        .collect()
}

fn bytes_of(leaf: Leaf) -> u64 {
    match leaf {
        Leaf::F32 => 4,
        Leaf::BF16 => 2,
        other => panic!("no synthetic tensor here is {other:?}"),
    }
}

/// A source of these names and shapes over a sparse file holding no bytes:
/// enough for a contract to build and type-check, which never reads a
/// value.
fn synthetic(dir: &Path, tensors: &[Named]) -> ztensor::Source {
    let path = dir.join("synthetic.bin");
    let mut catalog = Catalog::new();
    let mut offset = 0u64;
    for (name, shape, leaf) in tensors {
        let len = shape.iter().product::<u64>() * bytes_of(*leaf);
        catalog.insert(
            name.clone(),
            Entry::leaf(
                shape.clone(),
                *leaf,
                Location {
                    store: StoreId(0),
                    offset,
                    len,
                },
            ),
        );
        offset += len;
    }
    let file =
        std::fs::File::create(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    file.set_len(offset.max(1))
        .unwrap_or_else(|why| panic!("{}: a sparse file of {offset} bytes: {why}", path.display()));
    drop(file);
    let store = Store::index(&path, "safetensors").unwrap_or_else(|why| panic!("{why}"));
    ztensor::Source::from_parts(vec![store], catalog).unwrap_or_else(|why| panic!("{why}"))
}

fn scratch() -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "flux_2_import_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

/// The checkpoint's types, for the contract type checker.
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

/// How many times each checkpoint tensor is named by the contract.
fn reads(contract: &ModelContract) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for source in tensor.expr.sources() {
            *counts.entry(source.to_string()).or_default() += 1;
        }
    }
    counts
}

/// Every declared plane, in contract order, types to the extents it
/// declares — `Out` references resolving against what came before — and the
/// whole contract lowers to a load plan.
fn type_checks(contract: &ModelContract, src: &ztensor::Source) {
    let types = Types(src);
    let mut resolver = Resolver::new(&types, Partition::WHOLE);
    for tensor in &contract.tensors {
        let ty = resolver
            .infer(&tensor.expr, &tensor.name)
            .unwrap_or_else(|why| panic!("`{}` does not type: {why}", tensor.name));
        if let Some(shape) = &tensor.shape {
            assert_eq!(
                &ty.shape, shape,
                "`{}` declares {shape:?} and its expression yields {:?}",
                tensor.name, ty.shape
            );
        }
        assert_eq!(
            ty.encoding, tensor.encoding,
            "`{}` declares {:?} and its expression yields {:?}",
            tensor.name, tensor.encoding, ty.encoding
        );
        resolver.publish(&tensor.name, ty);
    }
    let metadata = checkpoint::file::zt::describe(src)
        .unwrap_or_else(|why| panic!("the source does not describe: {why}"));
    let plan = checkpoint::plan::compile(&metadata, contract, StorageTarget::default())
        .unwrap_or_else(|why| panic!("the contract does not lower to a load plan: {why}"));
    assert!(!plan.instrs.is_empty());
}

/// The read counts a transformer's tensors get: one each, but the single
/// blocks' `to_out` (two column blocks) and, on the flagship, the
/// `context_embedder` (three).
fn expected_dit_reads(prefix: &str, d: &Dims, flagship: bool) -> BTreeMap<String, usize> {
    transformer(d, Leaf::F32)
        .into_iter()
        .map(|(name, ..)| {
            // A cut names its source once per slice: the two column blocks
            // of `to_out`, the three of the flagship's `context_embedder`,
            // the six (three) reordered slices of a modulation linear.
            let count = if name.ends_with(".attn.to_out.weight") {
                2
            } else if flagship && name == "context_embedder.weight" {
                3
            } else if name.starts_with("double_stream_modulation") {
                6
            } else if name.starts_with("single_stream_modulation") {
                3
            } else {
                1
            };
            (format!("{prefix}{name}"), count)
        })
        .collect()
}

/// The miniature's contract over `src` (bare names or `dit.`-prefixed).
fn check_mini(src: &ztensor::Source, prefix: &str) {
    let row = models::sku(MINI).expect("the catalog ships the miniature");
    let contract = row
        .contract(src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its checkpoint: {why}"));
    assert_eq!(
        reads(&contract),
        expected_dit_reads(prefix, &Dims::mini(), false),
        "the miniature reads its whole state_dict, once each but `to_out`"
    );
    type_checks(&contract, src);
}

/// (a)
#[test]
fn the_miniature_reads_a_synthetic_state_dict_bare_and_prefixed() {
    for prefix in ["", "dit."] {
        let dir = scratch();
        let tensors = prefixed(prefix, transformer(&Dims::mini(), Leaf::F32));
        assert_eq!(tensors.len(), 51);
        let src = synthetic(&dir, &tensors);
        check_mini(&src, prefix);
        drop(src);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A raw checkpoint is read by name and typed by shape: the flagship over the
/// bare miniature refuses at build, having no encoder or VAE to read; the
/// miniature over a flagship-shaped transformer builds (every name it wants
/// is there, `guidance_embedder` aside — which it wants and does not find).
#[test]
fn neither_row_serves_the_other_rows_checkpoint() {
    let dir = scratch();
    let bare = transformer(&Dims::mini(), Leaf::F32);
    let src = synthetic(&dir, &bare);
    let flagship = models::sku(KLEIN).unwrap();
    assert!(
        flagship.contract(&src, Platform::Cuda).is_err(),
        "the flagship read a bare 256-wide transformer with no encoder"
    );
    let klein = prefixed("dit.", transformer(&Dims::klein_4b(), Leaf::BF16));
    let src = synthetic(&dir, &klein);
    let mini = models::sku(MINI).unwrap();
    assert!(
        mini.contract(&src, Platform::Cuda).is_err(),
        "the miniature wants a guidance embedder the flagship's transformer lacks"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The repo directory in the HuggingFace cache, honoring the same
/// precedence `huggingface_hub` uses.
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
    let snapshots = hub().join("models--black-forest-labs--FLUX.2-klein-4B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("model_index.json").is_file())
}

/// (b)
#[test]
fn the_flagship_reads_the_real_snapshot() {
    let Some(root) = snapshot() else {
        eprintln!(
            "skipping: no black-forest-labs/FLUX.2-klein-4B snapshot in the HuggingFace cache"
        );
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let index: BTreeSet<String> = src.names().map(str::to_string).collect();

    // The synthetic transformer of (a) is the snapshot's, name for name and
    // shape for shape.
    let synthesized = prefixed("dit.", transformer(&Dims::klein_4b(), Leaf::BF16));
    assert_eq!(synthesized.len(), 169);
    for (name, shape, _) in &synthesized {
        let real = src
            .get(name)
            .unwrap_or_else(|| panic!("the snapshot holds no `{name}`"));
        assert_eq!(real.shape(), shape.as_slice(), "`{name}`");
    }
    let synthesized: BTreeSet<&String> = synthesized.iter().map(|(name, ..)| name).collect();
    let real: BTreeSet<&String> = index.iter().filter(|n| n.starts_with("dit.")).collect();
    assert_eq!(
        synthesized, real,
        "the synthetic transformer and the snapshot's are one list"
    );

    let row = models::sku(KLEIN).expect("the catalog ships the flagship");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the flagship does not read this checkpoint: {why}"));
    let counts = reads(&contract);

    // Every `dit.` tensor, at the counts the cuts imply.
    let dit: BTreeMap<String, usize> = counts
        .iter()
        .filter(|(n, _)| n.starts_with("dit."))
        .map(|(n, c)| (n.clone(), *c))
        .collect();
    assert_eq!(dit, expected_dit_reads("dit.", &Dims::klein_4b(), true));

    // The encoder: the 27 layers this plan runs and the embedding; not the
    // last nine layers, not the final norm, no head.
    let te_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("te.")).collect();
    let te_want: BTreeSet<&String> = index
        .iter()
        .filter(|n| n.starts_with("te."))
        .filter(|n| {
            n.as_str() == "te.model.embed_tokens.weight"
                || n.strip_prefix("te.model.layers.")
                    .and_then(|rest| rest.split('.').next())
                    .and_then(|l| l.parse::<u32>().ok())
                    .is_some_and(|l| l < model::TE_LAYERS)
        })
        .collect();
    assert_eq!(
        te_read, te_want,
        "the encoder planes read are the layers the plan runs"
    );
    assert_eq!(te_want.len(), 1 + 11 * model::TE_LAYERS as usize);

    // The VAE: the whole autoencoder but the frozen BatchNorm's step
    // counter (`tests/the_flux_2_vae_bakes.rs` states the shape of those
    // reads; here only that this import performs them).
    let vae_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("vae.")).collect();
    let vae_want: BTreeSet<&String> = index
        .iter()
        .filter(|n| n.starts_with("vae.") && n.as_str() != "vae.bn.num_batches_tracked")
        .collect();
    assert_eq!(
        vae_read, vae_want,
        "the VAE planes read are every one but the BatchNorm's step counter"
    );

    // Every other tensor exactly once.
    let odd: BTreeSet<&String> = counts
        .iter()
        .filter(|(n, c)| **c != 1 && !n.starts_with("dit."))
        .map(|(n, _)| n)
        .collect();
    assert!(odd.is_empty(), "read more than once: {odd:?}");

    type_checks(&contract, &src);

    // The identification sweep lands on the flagship and no earlier row.
    let identified = models::identify(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the snapshot identifies as nothing: {why}"));
    assert_eq!(identified, KLEIN);
}

fn golden_mini() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let file = root.join("flux2/flux2_mini.safetensors");
    file.is_file().then_some(file)
}

/// (c)
#[test]
fn the_miniature_reads_the_golden_fixture() {
    let Some(file) = golden_mini() else {
        eprintln!("skipping: no flux2_mini.safetensors under $PIE_IMAGEGEN_GOLDEN/flux2");
        return;
    };
    let src = ztensor_compat::open(&file).unwrap_or_else(|why| panic!("{}: {why}", file.display()));
    let index: BTreeSet<String> = src.names().map(str::to_string).collect();
    let want: BTreeSet<String> = transformer(&Dims::mini(), Leaf::F32)
        .into_iter()
        .map(|(name, ..)| name)
        .collect();
    assert_eq!(index, want, "the fixture is the miniature's state_dict");
    check_mini(&src, "");
    let identified = models::identify(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the fixture identifies as nothing: {why}"));
    assert_eq!(identified, MINI);
}
