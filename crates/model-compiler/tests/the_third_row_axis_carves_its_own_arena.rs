//! **THE THIRD ROW AXIS BAKES AS ITS OWN UNIT, CARVES ITS OWN COLUMNS, AND
//! THE PATCHIFY PAIR SITS AT THE UNIT BOUNDARY.** (design D8)
//!
//! ```text
//! cargo test -p model-compiler --test the_third_row_axis_carves_its_own_arena
//! ```
//!
//! Two plans traced through the real DSL:
//!
//! (a) an ENCODER — a voxel port convolved then patchified into tokens a
//!     matmul reads — bakes to `units == [Voxels, Tokens]`: every voxel
//!     region first, the patchify (writes tokens) opening the token unit,
//!     the fold stood down, a voxel seriation beside the token one;
//! (b) a DECODER — tokens unpatchified into voxels, convolved, upsampled,
//!     shuffled into pixels — bakes to `units == [Tokens, Voxels]`, the
//!     unpatchify (writes voxels) opening the voxel unit; its pixel plane is
//!     carved at `RowExpr::VoxelsTimes(16)` — sixteen times the voxel
//!     ceiling, three channels — and its grids at `RowExpr::Clips`, and no
//!     voxel rectangle shares a column with a token one;
//! (c) the same decoder against budgets that size no voxel ceiling is
//!     refused by the axis's name, not carved at zero rows; a voxel ladder
//!     is checked on its own terms (a rung past `max_voxels` is refused
//!     where it would be legal against `max_tokens`).

use model_compiler::{
    Budget, Budgets, DeviceProfile, Error, Placement, RowAxis, RowExpr, VoxelLadder, compile,
    compile_axes,
};
use model_dsl::ops::spatial::{self, Conv};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops, seam,
    trace_hybrid,
};
use model_ir::{Operands, Trace};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

const C: u32 = 8;

fn conv(name: &str, c_in: u32, c_out: u32) -> Weight {
    Weight::sym(name, [u64::from(c_out), u64::from(c_in) * 27], Dtype::Bf16)
        .conv_taps_major(c_in, 27)
}

fn plane(name: &str, c: u32) -> Weight {
    Weight::sym(name, [u64::from(c)], Dtype::F32)
}

/// Voxels in, tokens out.
struct Encoder;

impl ForwardHybrid for Encoder {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C, Dtype::Bf16);
        let (h, g1) = spatial::conv3d(&x, &g, &conv("conv", C, C), None, Conv::same3(), None);
        let h = spatial::group_norm(
            &h,
            &g1,
            2,
            &plane("norm.w", C),
            &plane("norm.b", C),
            1e-6,
            true,
        );
        let tg = inputs.token_grid([1, 2, 2]);
        let tokens = spatial::patchify(&h, &g1, [1, 2, 2], &tg);
        let w = Weight::sym("proj", [64, u64::from(C) * 4], Dtype::Bf16);
        ops::linear::matmul(&tokens, &w)
    }
}

/// Tokens in, pixels out.
struct Decoder;

impl ForwardHybrid for Decoder {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let latents = inputs.latents(0, C * 4, Dtype::Bf16);
        let w = Weight::sym("proj", [u64::from(C) * 4, u64::from(C) * 4], Dtype::Bf16);
        let tokens = ops::linear::matmul(&latents, &w);
        let g = inputs.grid();
        let tg = inputs.token_grid([1, 2, 2]);
        let x = spatial::unpatchify(&tokens, &tg, [1, 2, 2], &g);
        let (h, g1) = spatial::conv3d(&x, &g, &conv("conv1", C, C), None, Conv::same3(), None);
        let (h, g2) = spatial::upsample_nearest(&h, &g1, [1, 2, 2], false);
        let (h, g3) = spatial::conv3d(&h, &g2, &conv("conv2", C, 12), None, Conv::same3(), None);
        let (y, g4) = spatial::pixel_shuffle(&h, &g3, [1, 2, 2]);
        seam::at(seam::PIXELS, &[&y, &g4]);
        y
    }
}

fn budgets() -> Budgets {
    Budgets::of(Budget::new(4, 64)).with_voxels(VoxelLadder {
        max_voxels: 256,
        buckets: vec![64, 256],
        max_clips: 4,
    })
}

fn unit_axes(trace: &Trace, compiled: &model_compiler::CompiledModel) -> Vec<(String, RowAxis)> {
    compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(_, region)| region.phase == model_compiler::Phase::Capture)
        .flat_map(|(at, region)| {
            let axis = compiled.axis_of(at);
            region
                .nodes
                .clone()
                .map(move |node| (trace.nodes[node as usize].op.name().to_string(), axis))
        })
        .collect()
}

#[test]
fn an_encoder_runs_its_voxel_unit_first_and_the_patchify_opens_the_token_one() {
    let trace = trace_hybrid("encoder", &Encoder, Platform::Cuda);
    let compiled = compile_axes(&trace, &budgets(), &DeviceProfile::default()).expect("bakes");
    assert_eq!(compiled.units, vec![RowAxis::Voxels, RowAxis::Tokens]);
    assert!(compiled.fold_refused, "two units, no single graph to fold");
    assert!(
        compiled.voxels.is_some(),
        "the voxel axis has its own seriation"
    );
    assert!(compiled.order_for(RowAxis::Voxels).is_some());
    assert!(compiled.patches.is_none(), "and no patch axis was stated");

    let placed = unit_axes(&trace, &compiled);
    let boundary = placed
        .iter()
        .position(|(name, _)| name == "spatial.patchify")
        .expect("the patchify is a capture node");
    assert!(
        placed[..boundary]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Voxels),
        "everything before the patchify is the voxel unit's: {placed:?}"
    );
    assert!(
        placed[boundary..]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Tokens),
        "the patchify writes token rows and opens the token unit: {placed:?}"
    );
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
}

#[test]
fn a_decoder_carves_its_pixels_at_sixteen_voxel_ceilings_and_shares_no_column() {
    let trace = trace_hybrid("decoder", &Decoder, Platform::Cuda);
    let compiled = compile_axes(&trace, &budgets(), &DeviceProfile::default()).expect("bakes");
    assert_eq!(compiled.units, vec![RowAxis::Tokens, RowAxis::Voxels]);

    let placed = unit_axes(&trace, &compiled);
    let boundary = placed
        .iter()
        .position(|(name, _)| name == "spatial.unpatchify")
        .expect("the unpatchify is a capture node");
    assert!(
        placed[..boundary]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Tokens)
    );
    assert!(
        placed[boundary..]
            .iter()
            .all(|(_, axis)| *axis == RowAxis::Voxels),
        "the unpatchify writes voxel rows and opens the voxel unit: {placed:?}"
    );

    // The pixel plane and its grid, off the seam.
    let pixels = trace
        .seams
        .iter()
        .find(|s| s.seam == "pixels")
        .expect("planted");
    let (plane, grid) = (pixels.values[0], pixels.values[1]);
    let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[plane.0 as usize] else {
        panic!("the pixel plane is a rectangle of the arena")
    };
    assert_eq!(*rows, RowExpr::VoxelsTimes(16));
    // 16 x 256 voxels x 3 channels x 2 bytes.
    assert_eq!(*bytes, 16 * 256 * 3 * 2);
    let Placement::Arena { rows, bytes, .. } = &compiled.arena.placements[grid.0 as usize] else {
        panic!("the grid is a rectangle of the arena")
    };
    assert_eq!(*rows, RowExpr::Clips);
    assert_eq!(*bytes, 4 * 4 * 4, "four clips of four i32");

    // No voxel rectangle shares a column with a token one: `co_tenants`
    // demands equal row symbols, and these are different axes.
    let token_values: Vec<_> = (0..trace.values.len() as u32)
        .map(model_ir::ValueId)
        .filter(|id| {
            matches!(&compiled.arena.placements[id.0 as usize], Placement::Arena { rows, .. }
                if rows.axis() == Some(RowAxis::Tokens))
        })
        .collect();
    assert!(!token_values.is_empty());
    for token in token_values {
        assert!(!compiled.arena.co_tenants(plane, token));
        assert!(!compiled.arena.co_tenants(grid, token));
    }
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
}

#[test]
fn a_voxel_plan_against_no_voxel_ceiling_is_refused_by_name() {
    let trace = trace_hybrid("decoder", &Decoder, Platform::Cuda);
    let refusal = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
        .expect_err("no voxel ceiling, no load");
    assert_eq!(
        refusal,
        Error::Unsized {
            axis: RowAxis::Voxels
        }
    );
    assert!(refusal.to_string().contains("voxels"), "{refusal}");

    // The voxel ladder is its own ladder: 48 is under `max_tokens` and over
    // `max_voxels`.
    let past = Budgets::of(Budget::new(4, 64)).with_voxels(VoxelLadder {
        max_voxels: 32,
        buckets: vec![16, 48],
        max_clips: 2,
    });
    assert!(matches!(
        compile_axes(&trace, &past, &DeviceProfile::default()),
        Err(Error::Budget { .. })
    ));
}
