//! Opening a Metal device from a boot config: the shell reads its own
//! tables directly out of the shared boot document.

use crate::api::{ContractFor, DeviceBoot, Metal};

/// Open the system's default Metal device from a boot document.
///
/// # Errors
///
/// A boot document that is not UTF-8 or not TOML, as a `String` (not
/// [`Fault`](crate::Fault): the caller is the runtime, whose errors are `anyhow`).
pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Metal, String> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the metal boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the metal boot config is not TOML: {error}"))?;
    tuning(&doc);
    crate::diag::publish(&diagnostics(&doc)?);
    Ok(Metal::new(
        DeviceBoot {
            gpu_mem_utilization: gpu_mem_utilization(&doc),
            adapter_dir: adapter_dir(&doc),
        },
        contract_for,
    ))
}

/// `[model] adapter_dir`: where this deployment keeps its shared adapters, one
/// subdirectory per adapter. Absent or empty is `None` (feature off).
fn adapter_dir(doc: &toml::Table) -> Option<std::path::PathBuf> {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("adapter_dir"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
}

/// `[metal] diagnostics`: the one word list a person debugging types
/// (`"tier-trace,kernel-profile=2"`), written here by `pie serve`/`pie run`'s
/// `--diag` or by `[engine] diagnostics` in the operator's own file.
///
/// **A word this shell does not speak refuses the open**, unlike the advisory
/// keys around it: an ignored `gpu_mem_utilization = "most"` leaves a good
/// default standing, but an ignored `teir-trace` leaves a person staring at a
/// silent log wondering what else is broken.
fn diagnostics(doc: &toml::Table) -> Result<crate::diag::Diagnostics, String> {
    let Some(words) = doc
        .get("metal")
        .and_then(toml::Value::as_table)
        .and_then(|metal| metal.get("diagnostics"))
    else {
        return Ok(crate::diag::Diagnostics::default());
    };
    let words = words.as_str().ok_or_else(|| {
        format!("[metal] diagnostics is a comma-separated word list, not {words}")
    })?;
    words
        .parse::<crate::diag::Diagnostics>()
        .map_err(|error| format!("[metal] diagnostics: {error}"))
}

/// `[metal] gpu_mem_utilization`: the fraction of `recommendedMaxWorkingSetSize`
/// this device may hold resident. Advisory: a missing key, wrong type, or a
/// fraction outside `(0, 1]` leaves the default standing rather than refusing.
fn gpu_mem_utilization(doc: &toml::Table) -> f64 {
    doc.get("metal")
        .and_then(toml::Value::as_table)
        .and_then(|metal| metal.get("gpu_mem_utilization"))
        .and_then(toml::Value::as_float)
        .filter(|fraction| fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0)
        .unwrap_or(crate::store::accounting::DEFAULT_GPU_MEM_UTILIZATION)
}

/// `[metal.tuning]`: kernel-selection crossovers, swept via the boot document
/// rather than environment variables (which would need a rebuild per arm). A
/// key not named here keeps the device's measured default; a key named to
/// zero means zero (e.g. `moe_batch_min_per_expert = 0`, not "absent"). A
/// wrong-typed value is dropped, not a refusal.
fn tuning(doc: &toml::Table) {
    let Some(table) = doc
        .get("metal")
        .and_then(toml::Value::as_table)
        .and_then(|metal| metal.get("tuning"))
        .and_then(toml::Value::as_table)
    else {
        return;
    };
    let int = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
    };
    let flag = |key: &str| table.get(key).and_then(toml::Value::as_bool);

    // The device row this table lays over, stated by the document until the
    // shell that binds the device states it itself.
    let described = kernels_metal::DeviceInfo {
        apple_family: int("apple_family").unwrap_or_default(),
        gpu_core_count: int("gpu_core_count").unwrap_or_default(),
    };
    if described != kernels_metal::DeviceInfo::default() {
        kernels_metal::tuning::describe(described);
    }

    kernels_metal::tuning::override_with(kernels_metal::tuning::Overrides {
        qmm_min_batch: int("qmm_min_batch"),
        qmm_min_batch_moe: int("qmm_min_batch_moe"),
        qmm_min_batch_emulated: int("qmm_min_batch_emulated"),
        qmm_bn_crossover_tg: int("qmm_bn_crossover_tg"),
        moe_tile_mid_per: int("moe_tile_mid_per"),
        moe_tile_wide_per: int("moe_tile_wide_per"),
        fp16_qmm: flag("fp16_qmm"),
        sdpa_tile_min_rows_per_request: int("sdpa_tile_min_rows_per_request"),
        sdpa_mma: flag("sdpa_mma"),
        gdn_scan_lanes: int("gdn_scan_lanes"),
        gdn_scan_rows: int("gdn_scan_rows"),
        moe_batch_min_per_expert: int("moe_batch_min_per_expert"),
        moe_batch_min_pairs: int("moe_batch_min_pairs"),
        qmv_rows_max: int("qmv_rows_max"),
        qmv_rows_packs: int("qmv_rows_packs"),
        stream_rows_per_cut: int("stream_rows_per_cut"),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Never called: every test here fails or succeeds at the door.
    fn nothing(
        _trace: &model_ir::Trace,
        _path: &std::path::Path,
    ) -> Result<checkpoint::contract::ModelContract, String> {
        Err("this door never loads".to_string())
    }

    #[test]
    fn a_boot_document_that_says_nothing_about_this_engine_still_opens() {
        assert!(open(b"", nothing).is_ok());
        assert!(open(b"[model]\nid = \"qwen35-d0.8b\"\n", nothing).is_ok());
    }

    #[test]
    fn a_boot_document_that_is_not_toml_is_refused_at_the_door() {
        assert!(open(b"this is not = = toml", nothing).is_err());
    }

    /// The word list is read where the shell reads everything else it is
    /// told, and — unlike the advisory keys beside it — a word this shell
    /// does not speak refuses the open rather than tracing nothing.
    #[test]
    fn the_diagnostics_word_list_is_read_or_refused_by_name() {
        let of = |src: &str| super::diagnostics(&src.parse::<toml::Table>().unwrap());
        assert_eq!(
            of("").expect("silence is no diagnostics"),
            crate::diag::Diagnostics::default()
        );
        let stated = of("[metal]\ndiagnostics = \"tier-trace,kernel-profile=2\"\n")
            .expect("two words this shell speaks");
        assert!(stated.tier_trace);
        assert_eq!(stated.kernel_profile, crate::diag::Profile::Shaped);

        let why = of("[metal]\ndiagnostics = \"teir-trace\"\n").expect_err("a misspelling");
        assert!(why.contains("teir-trace"), "the refusal names it: {why}");
        let why = of("[metal]\ndiagnostics = 3\n").expect_err("a number is not a word list");
        assert!(why.contains("word list"), "{why}");
        assert!(open(b"[metal]\ndiagnostics = \"teir-trace\"\n", nothing).is_err());
    }

    #[test]
    fn gpu_mem_utilization_reads_the_fraction_or_keeps_the_default() {
        let of = |src: &str| super::gpu_mem_utilization(&src.parse::<toml::Table>().unwrap());
        let default = crate::store::accounting::DEFAULT_GPU_MEM_UTILIZATION;
        assert_eq!(of(""), default, "silence keeps the CUDA-matching default");
        assert_eq!(of("[metal]\ngpu_mem_utilization = 0.75\n"), 0.75);
        assert_eq!(
            of("[metal]\ngpu_mem_utilization = 1.5\n"),
            default,
            "a fraction over one is dropped, not clamped to it"
        );
        assert_eq!(
            of("[metal]\ngpu_mem_utilization = \"most\"\n"),
            default,
            "a value of the wrong type leaves the default standing"
        );
    }

    #[test]
    fn the_shared_adapter_directory_is_read_and_an_empty_one_is_off() {
        let read = |text: &str| adapter_dir(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(
            read("[model]\nadapter_dir = \"/srv/pie/shared\""),
            Some(std::path::PathBuf::from("/srv/pie/shared"))
        );
        assert_eq!(read("[model]\nadapter_dir = \"  \""), None, "empty is off");
        assert_eq!(read("[model]"), None, "and so is absent");
        assert_eq!(read(""), None);
    }

    #[test]
    fn a_tuning_table_is_advisory_and_never_a_refusal() {
        assert!(open(b"[metal.tuning]\nsdpa_mma = false\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nqmm_min_batch = \"eight\"\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nqmm_min_batch = -3\n", nothing).is_ok());
        assert!(open(b"[metal.tuning]\nno_such_knob = 1\n", nothing).is_ok());
    }
}
