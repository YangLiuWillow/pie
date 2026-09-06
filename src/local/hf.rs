//! HuggingFace download helpers for `pie model pull`. The weight-download IO
//! lives only in `pie`, never the worker daemon.
//!
//! The worker lib resolves already-present snapshots (`weights::resolve`); this
//! crate owns the *fetch*. Downloads are runtime-artifact selective: safetensors
//! weights plus config/tokenizer files, excluding alternate checkpoint formats
//! the engines cannot load (`.pt`/`.bin`/`.gguf`/`consolidated.safetensors`).

pub mod download;

pub use download::{Progress, snapshot_download};

/// Where the HuggingFace CLI and libraries keep their blob cache.
///
/// Same precedence the `huggingface_hub` python package uses, so a snapshot
/// pulled by `huggingface-cli` and one pulled by pie land in one place:
/// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`,
/// else `~/.cache/huggingface/hub`.
pub fn resolve_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("huggingface").join("hub")
}

/// Files required by Pie's runtime loaders. Broad for small metadata + tokenizer
/// artifacts, but narrow for weights: the CUDA/Metal loaders consume
/// `model.safetensors` / `model-*.safetensors` shards, not duplicate `.pt`,
/// `.bin`, `.gguf`, or `consolidated.safetensors` artifacts. Used by
/// `pie model pull` to restrict the HF snapshot download.
///
/// # A diffusers pipeline is a repo of subfolders
///
/// A generative repo keeps `model_index.json` at the top and its weights one
/// level down: `transformer/diffusion_pytorch_model-00001-of-00003.safetensors`,
/// `text_encoder/model-00001-of-00003.safetensors`, `vae/config.json`,
/// `tokenizer/tokenizer.json`, `scheduler/scheduler_config.json`. The `**/`
/// half of this list already reaches every JSON and every `model*` shard at
/// any depth (`**` matches zero segments too, which is why the bare `*.json`
/// row is redundant but kept for readability); what it did not reach is
/// diffusers' own weight name, so `**/diffusion_pytorch_model*.safetensors`
/// is here.
///
/// **THE BUNDLE IS STILL NOT FETCHED.** FLUX.2 ships a single-file
/// `flux-2-klein-4b.safetensors` at the top of the snapshot holding the same
/// weights again for ComfyUI; it matches no pattern here, and the discovery
/// that reads a pipeline ignores it even when a hand-fetched snapshot has one
/// (`checkpoint::file::diffusers`). Images and READMEs match nothing either.
pub fn runtime_snapshot_allow_patterns() -> Vec<String> {
    [
        "*.json",
        "*.model",
        "*.txt",
        "*.tiktoken",
        "*.jinja",
        "model*.safetensors",
        "**/*.json",
        "**/*.model",
        "**/*.txt",
        "**/*.tiktoken",
        "**/*.jinja",
        "**/model*.safetensors",
        "**/diffusion_pytorch_model*.safetensors",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
