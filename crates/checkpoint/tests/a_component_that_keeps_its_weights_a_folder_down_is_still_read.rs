//! **A COMPONENT'S WEIGHTS ARE READ ONE FOLDER DOWN WHEN THEY ARE NOT
//! BESIDE ITS CONFIG — ONE LEVEL, AND NEVER OUT OF THE COMPONENT.**
//!
//! MiniMax H3 ships `video_vae/` as a folder of Python modules with the
//! weights in a `source/` subdirectory, so component discovery has to look
//! below a component that holds none itself. It must look no further and no
//! wider than that: a component whose own directory holds a set never
//! descends (so a subfolder cannot shadow it), two levels down is not
//! searched, and the bundle at the TOP of the snapshot stays unread — the
//! descent goes into a component, never up out of one.
//!
//!     cargo test -p checkpoint --test a_component_that_keeps_its_weights_a_folder_down_is_still_read

use std::path::{Path, PathBuf};

use checkpoint::file::diffusers;

/// Builds a one-tensor safetensors file at `path`.
fn safetensors(path: &Path, name: &str) {
    let bytes = half::bf16::from_f32(1.0).to_bits().to_le_bytes();
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,{}]}}}}",
        bytes.len()
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A pipeline with the four shapes this rule has to tell apart.
fn snapshot(dir: &Path) {
    write(
        &dir.join("model_index.json"),
        r#"{
            "_class_name": "APipeline",
            "audio_vae": ["diffusers", "AnAudioVAE"],
            "image_encoder": ["diffusers", "AnImageEncoder"],
            "transformer": ["diffusers", "ATransformer"],
            "video_vae": ["diffusers", "AVideoVAE"]
        }"#,
    );

    // transformer/: weights beside the config, and a subfolder that also
    // holds a set. The component's own weights win; the subfolder is never
    // looked at, so it cannot shadow or join them.
    safetensors(
        &dir.join("transformer/diffusion_pytorch_model.safetensors"),
        "beside.weight",
    );
    safetensors(
        &dir.join("transformer/original/model.safetensors"),
        "below.weight",
    );
    write(&dir.join("transformer/config.json"), r#"{"layers": 2}"#);

    // video_vae/: modules and a config at the component level, the weights
    // one folder down. MiniMax H3's shape.
    write(&dir.join("video_vae/vae_cnn.py"), "# a module\n");
    write(&dir.join("video_vae/config.json"), r#"{"z": 32}"#);
    write(&dir.join("video_vae/source/config.json"), r#"{"z": 32}"#);
    safetensors(
        &dir.join("video_vae/source/model.safetensors"),
        "cnn.weight",
    );

    // image_encoder/: two levels down is NOT a component's weights.
    write(&dir.join("image_encoder/config.json"), r#"{"dim": 8}"#);
    safetensors(
        &dir.join("image_encoder/checkpoints/step-100/model.safetensors"),
        "deep.weight",
    );

    // audio_vae/: nothing anywhere, and a hidden folder that holds a file
    // with the right name — a cache, not a component's weights.
    write(&dir.join("audio_vae/config.json"), r#"{"bins": 64}"#);
    safetensors(&dir.join("audio_vae/.cache/model.safetensors"), "cached");

    // The bundle at the top of the snapshot, still not read.
    safetensors(&dir.join("a-pipeline-bundle.safetensors"), "bundle.weight");
}

fn built() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    snapshot(&root);
    (dir, root)
}

#[test]
fn a_component_that_keeps_its_weights_a_folder_down_is_still_read() {
    let (_guard, root) = built();

    let components = diffusers::components(&root).unwrap();
    // Sorted: the order `components` answers in follows `model_index.json`'s
    // key order or its insertion order depending on whether the build turned
    // on `serde_json/preserve_order`, and this claim is about the SET.
    let mut seen: Vec<(&str, &str, usize)> = components
        .iter()
        .map(|c| (c.folder.as_str(), c.prefix.as_str(), c.weights.len()))
        .collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![("transformer", "dit.", 1), ("video_vae", "vae.", 1)],
        "the transformer's own file and the video VAE's file a folder down; \
         two levels down and a hidden folder are not a component's weights, \
         and a component with neither is skipped"
    );

    // The video VAE's set is the one under `source/`, named from there.
    let named = |folder: &str| {
        components
            .iter()
            .find(|c| c.folder == folder)
            .unwrap_or_else(|| panic!("{folder} is in the set"))
    };
    let vae = named("video_vae");
    assert_eq!(
        vae.weights,
        vec![root.join("video_vae/source/model.safetensors")]
    );
    // Its config is still the component's own, not the subfolder's: the
    // descent finds weights, it does not move the component.
    assert_eq!(vae.dir, root.join("video_vae"));
    assert_eq!(vae.config, Some(root.join("video_vae/config.json")));

    // The transformer never descended.
    assert_eq!(
        named("transformer").weights,
        vec![root.join("transformer/diffusion_pytorch_model.safetensors")]
    );

    // And the merged name space holds exactly the two components' tensors:
    // nothing from `original/`, `checkpoints/`, `.cache/` or the bundle.
    let source = diffusers::open(&root).unwrap();
    let mut names: Vec<&str> = source.names().collect();
    names.sort_unstable();
    assert_eq!(names, vec!["dit.beside.weight", "vae.cnn.weight"]);
}
