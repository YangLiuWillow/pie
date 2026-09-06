//! **THE REAL MINIMAX H3 SNAPSHOT ANSWERS ITS `video_vae` — WHOSE WEIGHTS
//! ARE A FOLDER DOWN, IN `video_vae/source/model.safetensors`.**
//!
//! The synthetic sibling
//! (`a_component_that_keeps_its_weights_a_folder_down_is_still_read`) proves
//! the rule; this one proves the rule was written against the real thing.
//! `MiniMaxAI/MiniMax-H3` ships three pipelines in one snapshot — a modular
//! one at the top, and the two task pipelines `FL2VA/` and `Ref2VA/`, whose
//! `video_vae/` is a folder of Python modules with a `source/` subdirectory
//! under it. Before the descent each task pipeline came back with its video
//! VAE MISSING — three components, no `vae.` prefix at all, a decoder
//! silently absent from the name space; after it, four.
//!
//! Skipped by name when the snapshot is not in the HuggingFace cache.
//!
//!     cargo test -p checkpoint --test the_minimax_h3_snapshot_finds_its_video_vae_a_folder_down

use std::path::{Path, PathBuf};

use checkpoint::file::diffusers;

/// The repo directory in the HuggingFace cache, honoring the same precedence
/// `huggingface_hub` uses.
fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

/// The one snapshot directory of `models--MiniMaxAI--MiniMax-H3`, or `None`
/// when this machine has not pulled it.
fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--MiniMaxAI--MiniMax-H3/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("FL2VA/model_index.json").is_file())
}

/// The component set as `(folder, prefix, files)`, SORTED: whether
/// `components` answers in `model_index.json`'s insertion order or in its
/// keys' alphabetical order depends on whether something in the build turned
/// on `serde_json/preserve_order`, and this claim is about the set.
fn roster(dir: &Path) -> Vec<(String, String, usize)> {
    let mut seen: Vec<(String, String, usize)> = diffusers::components(dir)
        .unwrap()
        .into_iter()
        .map(|c| (c.folder, c.prefix, c.weights.len()))
        .collect();
    seen.sort();
    seen
}

#[test]
fn the_minimax_h3_snapshot_finds_its_video_vae_a_folder_down() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no MiniMaxAI/MiniMax-H3 snapshot in the HuggingFace cache");
        return;
    };

    // The two task pipelines, each with the video VAE the descent recovers.
    for partition in ["FL2VA", "Ref2VA"] {
        let dir = root.join(partition);
        let seen = roster(&dir);
        let named: Vec<(&str, &str, usize)> = seen
            .iter()
            .map(|(f, p, n)| (f.as_str(), p.as_str(), *n))
            .collect();
        assert_eq!(
            named,
            vec![
                ("audio_vae", "avae.", 1),
                ("text_encoder", "te.", 14),
                ("transformer", "dit.", 13),
                ("video_vae", "vae.", 1),
            ],
            "{partition}: the video VAE is the component the flat discovery missed"
        );

        // Its one file is the one under `source/`, and the component still
        // points at the folder `model_index.json` named.
        let components = diffusers::components(&dir).unwrap();
        let vae = components
            .iter()
            .find(|c| c.folder == "video_vae")
            .expect("the video VAE is in the set");
        assert_eq!(
            vae.weights,
            vec![dir.join("video_vae/source/model.safetensors")]
        );
        assert_eq!(vae.dir, dir.join("video_vae"));
        assert!(
            !dir.join("video_vae/model.safetensors").exists()
                && !dir
                    .join("video_vae/diffusion_pytorch_model.safetensors")
                    .exists(),
            "{partition}: the component folder itself holds no set — which is \
             why it was skipped before the descent"
        );
    }

    // The pipeline at the TOP of the snapshot answers nothing at all, before
    // the descent and after it — and not for want of weights (its `vae/`
    // holds three shards beside its config). Its `model_index.json` is
    // diffusers' MODULAR form, whose every entry is a THREE-element array
    // `[library, class, {spec}]`, and `components` reads only the two-element
    // one. That is a separate gap, named here so the next reader does not
    // mistake it for this one; the two task pipelines above are the
    // snapshot's real checkpoints and they read.
    assert!(
        roster(&root).is_empty(),
        "the modular top-level index is not read by component discovery"
    );
    assert!(
        root.join("vae/diffusion_pytorch_model.safetensors.index.json")
            .is_file(),
        "and not because the modular pipeline holds no weights"
    );

    // And the recovered VAE opens: its tensors land under `vae.`.
    let source = diffusers::open(&root.join("FL2VA")).unwrap();
    let vae_names = source.names().filter(|n| n.starts_with("vae.")).count();
    assert!(
        vae_names > 0,
        "the video VAE contributes tensors to the merged name space"
    );
    eprintln!("FL2VA: {vae_names} tensors under `vae.`");
}
