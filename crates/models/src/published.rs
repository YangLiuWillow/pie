//! **WHICH PUBLISHED HEAD GOES WITH WHICH PUBLISHED TARGET** — the catalog's
//! pairings, kept apart from both the drafter text (which knows no model) and
//! the families (which know their own trunk and no repository).

/// **A PUBLISHED HEAD FOR A PUBLISHED TARGET** — the pairing an operator
/// would otherwise look up by hand: which repository carries the drafter
/// trained against which checkpoint, and which catalog row reads the two
/// together. `pie model import <target> --drafter <name>` resolves through
/// this table; so does `[model] drafter = "<name>"` when `[model] model`
/// names the target repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Published {
    /// The target checkpoint's repository id, as `pie model import` takes it.
    pub target: &'static str,
    /// The drafter's repository id — the `--aux` overlay.
    pub head: &'static str,
    /// The drafter's short name, what `--drafter` and `[model] drafter` say.
    pub drafter: &'static str,
    /// The catalog row that reads target and head together.
    pub sku: &'static str,
}

/// Every head this build knows how to overlay, by target.
pub const PUBLISHED: &[Published] = &[
    Published {
        target: "mlx-community/Qwen3.6-27B-4bit",
        head: "z-lab/Qwen3.6-27B-DFlash",
        drafter: "dflash",
        sku: "qwen36-27b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.8-27B-4bit",
        head: "z-lab/Qwen3.8-27B-DFlash2",
        drafter: "dflash2",
        sku: "qwen38-27b-dflash2-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.8-27B-4bit",
        head: "DimInfer/Qwen3.8-27B-Dspark-v1",
        drafter: "dspark",
        sku: "qwen38-27b-dspark-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.6-35B-A3B-4bit",
        head: "z-lab/Qwen3.6-35B-A3B-DFlash",
        drafter: "dflash",
        sku: "qwen36-35b-a3b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.5-9B-4bit",
        head: "z-lab/Qwen3.5-9B-DFlash",
        drafter: "dflash",
        sku: "qwen35-d9b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/gpt-oss-20b-MXFP4-Q4",
        head: "z-lab/gpt-oss-20b-DFlash",
        drafter: "dflash",
        sku: "gptoss-20b-dflash-u4g64-mxfp4-kv-bf16",
    },
    Published {
        target: "mlx-community/gemma-4-26b-a4b-it-4bit",
        head: "z-lab/gemma-4-26B-A4B-it-DFlash",
        drafter: "dflash",
        sku: "gemma4-26b-a4b-dflash-u4g64-kv-bf16",
    },
];

/// The published head named `drafter` for `target`, if this build knows one.
/// `target` is matched as a repository id, case-insensitively, with or
/// without the `--` spelling a store directory uses for `/`.
#[must_use]
pub fn lookup(target: &str, drafter: &str) -> Option<&'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED.iter().find(|p| {
        p.drafter.eq_ignore_ascii_case(drafter) && p.target.to_ascii_lowercase() == wanted
    })
}

/// Every published head this build knows for `target`.
pub fn for_target(target: &str) -> impl Iterator<Item = &'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED
        .iter()
        .filter(move |p| p.target.to_ascii_lowercase() == wanted)
}
