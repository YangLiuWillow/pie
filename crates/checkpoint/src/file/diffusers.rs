//! A diffusers-style pipeline folder read as ONE tensor name space.
//!
//! A generative checkpoint is not a checkpoint the way a language model's is.
//! It is a directory with a `model_index.json` at the top naming components,
//! and one subdirectory per component — `transformer/`, `text_encoder/`,
//! `vae/` — each with its own `config.json` and its own safetensors set, each
//! of which spells its tensors from its own root (`norm.weight` appears in
//! three of them). The flat discovery in [`super::read`] wants
//! `model.safetensors` beside the config and sees none of this.
//!
//! This module is the component-aware half of discovery. It answers three
//! questions and nothing else:
//!
//! - **which components does this snapshot hold** ([`components`]),
//! - **what does one name space over all of them look like** ([`open`]),
//! - **which JSON does each component say about itself** ([`configs`]).
//!
//! # The prefix vocabulary is fixed here
//!
//! A component's tensors enter the shared name space under a prefix, so
//! `transformer/`'s `norm.weight` is `dit.norm.weight` and `vae/`'s is
//! `vae.norm.weight`. The prefix is a ROLE, not a folder name, because a
//! family's `import.rs` spells its checkpoint paths against the role: every
//! diffusion transformer is `dit.`, whether the pipeline called its folder
//! `transformer` or something else. [`prefix_of`] is the whole vocabulary:
//!
//! | folder | prefix | what it is |
//! |---|---|---|
//! | `transformer` | `dit.` | the denoising backbone |
//! | `transformer_2` | `dit2.` | a second backbone (Wan's low-noise half) |
//! | `text_encoder` | `te.` | the conditioning encoder |
//! | `text_encoder_2` | `te2.` | a second encoder |
//! | `vae` | `vae.` | the latent autoencoder |
//! | `video_vae` | `vae.` | the same role under a pipeline that also
//!   ships a second, non-latent codec (MiniMax H3 pairs a `video_vae/`
//!   with an `audio_vae/`; the video one IS its latent autoencoder) |
//! | `image_encoder` | `ie.` | a reference-image encoder |
//! | `audio_vae` | `avae.` | an audio autoencoder |
//! | `vocoder` | `voc.` | an audio decoder |
//! | anything else | `<folder>.` | the folder's own name |
//!
//! This is the same staging `pie model import --aux` does for a drafter head
//! (`aux.`), minus the byte copy: prefixing renames a catalog, so no pipeline
//! is rewritten to be read.
//!
//! # A top-level bundle beside a pipeline is IGNORED
//!
//! FLUX.2 ships `flux-2-klein-4b.safetensors` at the top of the snapshot
//! beside `transformer/`, `text_encoder/` and `vae/`: one file holding the
//! same weights again, packed the way ComfyUI wants them. Reading both would
//! double the artifact and collide on nothing (the bundle's names are its
//! own), so **the components win**: when `model_index.json` is present, the
//! only weights this module reads are the ones under a component folder.
//! `model_index.json` is the statement that the subfolders are the checkpoint.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::Error;

/// The file whose presence makes a directory a pipeline rather than a
/// checkpoint.
pub const PIPELINE_INDEX: &str = "model_index.json";

/// The weight-file stems a component may use, in the order they are tried.
/// `diffusers` writes `diffusion_pytorch_model*`, `transformers` writes
/// `model*`; a component is one library's or the other's, never both.
const STEMS: [&str; 2] = ["diffusion_pytorch_model", "model"];

/// Components that are never weights: a scheduler is a formula in a JSON
/// file, a tokenizer is a vocabulary. Both are carried as config
/// ([`configs`]) and neither contributes a tensor.
fn is_weightless(folder: &str) -> bool {
    folder == "scheduler" || folder.starts_with("tokenizer") || folder.starts_with("feature_")
}

/// One component of a pipeline: where it is, what it is, and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// The subdirectory's name, as `model_index.json` spells it —
    /// `transformer`, `text_encoder_2`.
    pub folder: String,
    /// The name space prefix its tensors take, ending in `.`. See
    /// [`prefix_of`].
    pub prefix: String,
    /// The `[library, class]` `model_index.json` records, e.g.
    /// `("diffusers", "ZImageTransformer2DModel")`.
    pub library: String,
    /// The class within that library.
    pub class: String,
    /// The component's directory.
    pub dir: PathBuf,
    /// Its safetensors set, in the order a shard index names them.
    pub weights: Vec<PathBuf>,
    /// Its `config.json`, when it has one.
    pub config: Option<PathBuf>,
}

/// The prefix a component's tensors take in the shared name space.
///
/// A ROLE, not a folder name — see the module header for the table and for
/// why a family's `import.rs` spells `dit.` rather than `transformer.`.
/// A folder this vocabulary does not know keeps its own name, so an
/// unrecognised component is readable rather than refused.
#[must_use]
pub fn prefix_of(folder: &str) -> String {
    let role = match folder {
        "transformer" => "dit",
        "transformer_2" => "dit2",
        "text_encoder" => "te",
        "text_encoder_2" => "te2",
        "vae" | "video_vae" => "vae",
        "image_encoder" => "ie",
        "audio_vae" => "avae",
        "vocoder" => "voc",
        other => other,
    };
    format!("{role}.")
}

/// Whether `dir` is a diffusers pipeline: a directory with a
/// `model_index.json` in it.
///
/// The cheap question, asked before the expensive one — every door that
/// opens a checkpoint asks this first to decide whether the flat discovery
/// or [`components`] applies.
#[must_use]
pub fn is_pipeline(dir: &Path) -> bool {
    dir.is_dir() && dir.join(PIPELINE_INDEX).is_file()
}

/// Every weight-bearing component of the pipeline at `dir`, in the order
/// `model_index.json` lists them (which is alphabetical, since it is a JSON
/// object read into a map).
///
/// A key is a component when its value is a two-element array of strings —
/// `"transformer": ["diffusers", "ZImageTransformer2DModel"]`. Everything
/// else in that file is pipeline configuration: `_class_name`,
/// `_diffusers_version`, `boundary_ratio`, and the `[null, null]` Wan 2.2
/// writes for a `transformer_2` its TI2V variant does not ship. Schedulers
/// and tokenizers are components but hold no tensors, so they are dropped
/// here and carried by [`configs`] instead.
///
/// A component the index names but the snapshot does not hold (a partial
/// download, an `image_encoder` the repo lists and omits) is skipped, not
/// refused: the set is what is on this disk.
///
/// # Errors
///
/// `dir` holds no `model_index.json`, the file is not JSON, or a component
/// folder holds a shard index naming files that are not beside it.
pub fn components(dir: &Path) -> Result<Vec<Component>, Error> {
    let index = dir.join(PIPELINE_INDEX);
    let text = std::fs::read_to_string(&index)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Checkpoint(format!("{} is not a JSON object", index.display())))?;

    let mut found = Vec::new();
    for (folder, entry) in object {
        let Some((library, class)) = pair(entry) else {
            continue;
        };
        if folder.starts_with('_') || is_weightless(folder) {
            continue;
        }
        let component_dir = dir.join(folder);
        if !component_dir.is_dir() {
            continue;
        }
        let weights = weight_files(&component_dir)?;
        if weights.is_empty() {
            continue;
        }
        let config = component_dir.join("config.json");
        found.push(Component {
            folder: folder.clone(),
            prefix: prefix_of(folder),
            library,
            class,
            dir: component_dir,
            weights,
            config: config.is_file().then_some(config),
        });
    }
    Ok(found)
}

/// `["diffusers", "AutoencoderKL"]` as a pair; `None` for anything else,
/// including Wan's `[null, null]` placeholder for an absent second backbone.
fn pair(entry: &serde_json::Value) -> Option<(String, String)> {
    let array = entry.as_array()?;
    let [library, class] = array.as_slice() else {
        return None;
    };
    Some((library.as_str()?.to_string(), class.as_str()?.to_string()))
}

/// The safetensors set one component directory holds.
///
/// Each stem is tried whole before the next, so a directory holding both a
/// `model.safetensors` and a `diffusion_pytorch_model.safetensors` reads as
/// the diffusers one it is: the stems are two libraries' conventions, and a
/// component belongs to one library.
///
/// Within a stem: a shard index names the set when there is one, else the
/// lone unsharded file, else the numbered shards on the disk. The index is
/// preferred because it is the checkpoint's own statement of which files
/// belong together, and a directory can hold a stale extra shard.
fn weight_files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    for stem in STEMS {
        let index = dir.join(format!("{stem}.safetensors.index.json"));
        if index.is_file() {
            return shards_from_index(dir, &index);
        }
        let single = dir.join(format!("{stem}.safetensors"));
        if single.is_file() {
            return Ok(vec![single]);
        }
        let loose = loose_shards(dir, stem);
        if !loose.is_empty() {
            return Ok(loose);
        }
    }
    Ok(Vec::new())
}

/// The shard files a `*.safetensors.index.json` weight map names, unique and
/// sorted — the same dedup [`super::read::discover_safetensors_files`] does,
/// against the same shape of file.
fn shards_from_index(dir: &Path, index: &Path) -> Result<Vec<PathBuf>, Error> {
    let text = std::fs::read_to_string(index)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
    })?;
    let weight_map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::Checkpoint(format!("{} missing 'weight_map'", index.display())))?;
    let mut names = BTreeSet::new();
    for shard in weight_map.values() {
        let shard = shard.as_str().ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} weight_map has a non-string shard",
                index.display()
            ))
        })?;
        names.insert(shard.to_string());
    }
    let mut paths = Vec::with_capacity(names.len());
    for name in names {
        let path = dir.join(&name);
        if !path.is_file() {
            return Err(Error::Checkpoint(format!(
                "{} names the shard {name}, which is not beside it; a sharded \
                 component is one checkpoint, and reading the shards that happen \
                 to be present would import a model with holes",
                index.display()
            )));
        }
        paths.push(path);
    }
    Ok(paths)
}

/// `<stem>-00001-of-00003.safetensors` and its siblings, sorted, for a
/// component whose index file did not come down with the rest.
fn loose_shards(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with(&format!("{stem}-")) && name.ends_with(".safetensors")
                    })
        })
        .collect();
    found.sort();
    found
}

/// The pipeline at `dir` as one tensor name space, every component's tensors
/// under its [`prefix_of`] prefix.
///
/// # Errors
///
/// `dir` is not a pipeline, a component's files do not open as safetensors,
/// or two components' prefixed names collide (which the fixed vocabulary
/// makes impossible, and which is checked rather than assumed).
pub fn open(dir: &Path) -> Result<ztensor::Source, Error> {
    open_from(&components(dir)?, dir)
}

/// [`open`], over a component set the caller already has.
pub fn open_from(components: &[Component], dir: &Path) -> Result<ztensor::Source, Error> {
    if components.is_empty() {
        return Err(Error::Checkpoint(format!(
            "{} has a {PIPELINE_INDEX} but no component holds any safetensors; \
             a pipeline's weights live under its component folders, and a \
             bundle beside them is not read",
            dir.display()
        )));
    }
    let mut parts = Vec::with_capacity(components.len());
    for component in components {
        let source = ztensor_compat::index_all(&component.weights).map_err(|err| {
            Error::Checkpoint(format!(
                "cannot read the {} component of {}: {err}",
                component.folder,
                dir.display()
            ))
        })?;
        parts.push(source.under(&component.prefix).map_err(Error::from)?);
    }
    ztensor::Source::merge(parts).map_err(Error::from)
}

/// Every JSON a pipeline says about itself, as `(component, bytes)` — the
/// pairs an import carries into the artifact's config name space.
///
/// The empty component name is the pipeline's own `model_index.json`, which
/// is what a pipeline has instead of a top-level `config.json`; every other
/// entry is a component folder's `config.json`, and `scheduler` is its
/// `scheduler_config.json`. Callers land these at `model/config` and
/// `model/<component>/config` respectively.
///
/// Keyed by FOLDER, not by prefix: the config is diffusers' own statement
/// about diffusers' own component, and renaming it would make the carried
/// JSON disagree with the file it came from. The prefix is pie's name for
/// the weights; the folder is the checkpoint's name for the component.
///
/// # Errors
///
/// A file that is present does not parse as JSON — carrying a config that is
/// not one would put a broken descriptor in the artifact.
pub fn configs(dir: &Path) -> Result<Vec<(String, Vec<u8>)>, Error> {
    let mut carried = Vec::new();
    let index = dir.join(PIPELINE_INDEX);
    if let Some(bytes) = read_json(&index)? {
        carried.push((String::new(), bytes));
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(carried);
    };
    let mut folders: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    folders.sort();
    for folder in folders {
        let Some(name) = folder.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        for file in ["config.json", "scheduler_config.json"] {
            if let Some(bytes) = read_json(&folder.join(file))? {
                carried.push((name.to_string(), bytes));
                break;
            }
        }
    }
    Ok(carried)
}

/// A JSON file's bytes, verbatim, checked to be JSON; `None` if absent.
fn read_json(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read(path)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", path.display())))?;
    serde_json::from_slice::<serde_json::Value>(&raw)
        .map_err(|err| Error::Checkpoint(format!("cannot parse {}: {err}", path.display())))?;
    Ok(Some(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary is a role table, and a stranger keeps its own name.
    #[test]
    fn every_known_component_takes_its_role_as_a_prefix() {
        assert_eq!(prefix_of("transformer"), "dit.");
        assert_eq!(prefix_of("transformer_2"), "dit2.");
        assert_eq!(prefix_of("text_encoder"), "te.");
        assert_eq!(prefix_of("text_encoder_2"), "te2.");
        assert_eq!(prefix_of("vae"), "vae.");
        assert_eq!(prefix_of("image_encoder"), "ie.");
        assert_eq!(prefix_of("audio_vae"), "avae.");
        assert_eq!(prefix_of("vocoder"), "voc.");
        assert_eq!(prefix_of("connector"), "connector.");
    }

    /// Wan 2.2's `"transformer_2": [null, null]` is an absent component, not
    /// a component named null.
    #[test]
    fn a_null_pair_is_not_a_component() {
        assert_eq!(
            pair(&serde_json::json!(["diffusers", "AutoencoderKL"])),
            Some(("diffusers".to_string(), "AutoencoderKL".to_string()))
        );
        assert_eq!(pair(&serde_json::json!([null, null])), None);
        assert_eq!(pair(&serde_json::json!("0.36.0")), None);
        assert_eq!(pair(&serde_json::json!(["one"])), None);
    }
}
