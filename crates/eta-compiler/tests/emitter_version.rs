//! Pins each backend's emitter version against the compiled-in constant, and
//! a fingerprint of everything `emit_program` produces against a hash, so an
//! emitter change that moves the output can't leave a stale cache key behind
//! unnoticed. Not a golden: the hash says only *that* the output changed.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use msl_corpus::{corpus_bound, corpus_stages, extended_stages};
use eta_compiler::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use eta_compiler::codegen::metal::METAL_M1_EMITTER_VERSION;
use eta_compiler::codegen::program::{Backend, emit_program};

/// `(backend, version, fingerprint)`, both numbers written out as literals
/// (not read off the constant) so each row is a self-contained claim, and a
/// fingerprint change that doesn't move the version fails
/// `the_pinned_versions_are_the_compiled_ones` until someone updates it here.
const PINNED: &[(&str, u16, u64)] = &[
    // 28 -> 29: `fused_block0.cuh` (spliced into every emitted kernel) grew
    // `ptir_fast_gumbel_argmax_intrinsic`, and a Gumbel-max head emits a call
    // to it instead of its four launches. 29 -> 32 (merged with 30/31):
    // row-parallel streams through registers, the reshaped-row-vector
    // broadcast fold and the pooled `top_k` select. 32 -> 33: a stream's
    // vector loop (16-byte loads and stores, four elements a thread) beside
    // the scalar one, chosen per block by alignment. 33 -> 34: a later pass
    // recomputes an intrinsic-derived value instead of loading it, and a
    // `top_k` of the (scaled) logits ranks the intrinsic plane straight.
    // 34 -> 35: `fused_block0.cuh`'s `gather_row` reads a row block's
    // rank-1 view as one row (it zero-filled a row's width past its one
    // output element).
    // 35 -> 36: a row-parallel region's nodes are emitted in reduction-depth
    // order, so an op that waits on no reduction joins the first pass.
    // 36 -> 37: `sin`/`cos`/`sqrt`/`rsqrt` and `RngKind::Normal` joined the
    // op set, so the RNG preamble gained `ptir_rng_hash_normal` and both
    // spliced runtimes (`fused_block0.cuh`, `ptir_m1_runtime_body.cuh`)
    // gained arms -- which moves every emitted kernel's bytes, not only
    // those of a program using them.
    // 37 -> 38: `IntrinsicId::Pixels` joined the intrinsic set (design D8),
    // and the emitted preamble's per-intrinsic side tables are pitched by
    // `IntrinsicId::SLOTS` -- so every emitted kernel's bytes move, not only
    // those of a program that reads the new one.
    // 38 -> 39: `IntrinsicId::PeerVelocity` joined the intrinsic set (the
    // guidance read: another lane of one's own attention group's velocity),
    // so `IntrinsicId::SLOTS` moved 10 -> 11 and every emitted kernel's
    // per-intrinsic side tables re-pitch -- for the same reason 37 -> 38 did.
    ("cuda", 39, 0xf839_541b_764d_fe20),
    // 44 -> 45 -> 46 -> 47: `ptir_m1_runtime.metal` (spliced into every
    // emitted kernel) grew the threadgroup-partitioned op walk, then the
    // partitioned selections, then the streamed form's level reductions and
    // its own `ptir_m4` kernels joined the table. 49 -> 50 (merged with 45):
    // the normalization fold reaches the metal output too. 50 -> 51: a
    // gather is a direct op, a scalar runs mid-dispatch, a scatter splits.
    // 51 -> 52: the same op-set growth as cuda 36 -> 37, reaching
    // `ptir_m1_runtime.metal` and the generated RNG preamble.
    ("metal", 52, 0xdb5b_4877_cc81_27a7),
];

/// Everything an engine receives for both corpora, hashed. Includes the
/// extended corpus (reaches ops/intrinsics the base one doesn't) and
/// refusals alongside sources, since a region that stops being emittable
/// changes engine behavior just as much as a changed source would.
fn fingerprint(backend: Backend) -> u64 {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .map(|stage| stage.plan)
        .collect();
    let mut bytes = Vec::new();
    for kernel in emit_program(backend, &stages, &corpus_bound()) {
        bytes.extend_from_slice(&(kernel.kind as u32).to_le_bytes());
        bytes.extend_from_slice(&kernel.stage_index.to_le_bytes());
        bytes.extend_from_slice(&kernel.region_index.to_le_bytes());
        for text in [&kernel.entry_name, &kernel.source, &kernel.error] {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
    }
    eta_ir::fnv1a64(&bytes)
}

fn backend_of(name: &str) -> Backend {
    match name {
        "cuda" => Backend::Cuda,
        "metal" => Backend::Metal,
        other => panic!("no backend named `{other}`"),
    }
}

/// The pinned version literals are the constants the engines are compiled
/// against, so `PINNED` describes this compiler rather than a past one.
#[test]
fn the_pinned_versions_are_the_compiled_ones() {
    for (name, version, _) in PINNED {
        let constant = match backend_of(name) {
            Backend::Cuda => CUDA_GENERATED_EMITTER_VERSION,
            Backend::Metal => METAL_M1_EMITTER_VERSION,
        };
        assert_eq!(
            *version, constant,
            "the {name} emitter version constant is {constant}, but PINNED still says {version}. \
             If the emitted bytes changed, update both; if they did not, the constant moved for \
             nothing and every engine's cache was discarded."
        );
    }
}

/// The emitted bytes are still the ones the pinned version was written for.
/// A failure means the repair is to bump the backend's constant and re-pin.
#[test]
fn each_emitter_version_still_describes_its_output() {
    let mut moved = Vec::new();
    for (name, version, pinned) in PINNED {
        let actual = fingerprint(backend_of(name));
        if actual != *pinned {
            moved.push(format!(
                "{name}: emitter version {version} was pinned to {pinned:#018x} but now emits \
                 {actual:#018x}"
            ));
        }
    }
    assert!(
        moved.is_empty(),
        "the emitted bytes changed, so the engines' compile caches must be \
         invalidated:\n  {}\n\nBump the backend's emitter version constant and \
         put the new fingerprint in PINNED, in the same commit. If the change \
         was meant to be output-neutral, this is the bug report.",
        moved.join("\n  ")
    );
}

