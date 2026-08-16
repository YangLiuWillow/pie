// What does `moe_route_sort` cost ON ITS OWN, and what does that cost scale with?
//
// ## The question this exists to settle
//
// Ablation prices the sort at 2.04 ms of a 17.58 ms decode fire -- 11.6%, about
// 42 us a layer (`docs/NEXT-decode-dispatch-count.md`). Its body at decode is
// tiny: `PIE_METAL_MOE_TRACE=1` reports `n=8 experts=128 tile_rows=1 padded=8
// tiles=8 lanes=128`, i.e. roughly 150 memory operations and five barriers in
// ONE threadgroup. Nothing in that should take 42 us.
//
// In situ that number is unavoidably a mixture of two different things:
//
//   1. the kernel's own execution, and
//   2. what having that dispatch in the chain does to its neighbours -- a
//      pipeline drain, a dependency stall, a flush.
//
// **They want opposite fixes.** If it is (1) the kernel is rewritten; if it is
// (2) the kernel is fused away or moved, and rewriting it would buy nothing.
// Dispatching it alone, with nothing before or after to stall against,
// separates them: whatever it costs here is (1), and the remainder of the
// 42 us is (2).
//
// ## And what it scales with
//
// The sweep varies ONE parameter at a time with every result still observable,
// which is the correction to the trap recorded in the doc: bisecting the body
// with early `return`s measured nothing but dead-code elimination, because the
// counting, the prefix scan and the cursors have no consumer once the scatter is
// unreachable. Varying inputs cannot be optimized away.
//
//   experts   the clear loop, the span, the prefix scan  -- all per-expert
//   n         the counting loop and the scatter          -- per (row, slot)
//   lanes     the threadgroup width
//
// A cost flat in all three is fixed overhead and not the body at all.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"
#include "mtl4_context.hpp"

using namespace pie::metal;

namespace {

// `moe_params.h`'s MoeRouteParams, which the kernel reads as a constant buffer.
struct MoeRouteParamsHost {
    unsigned int n;
    unsigned int n_experts;
    unsigned int experts_per_token;
    unsigned int tile_rows;
    unsigned int padded;
    unsigned int width;
    unsigned int x_pitch;
};

// `decode_abi.hpp`, `enum class MoeRouteSort`.
constexpr int kExpertIds = 0, kPerm = 1, kRowExpert = 2, kTileExpert = 3,
              kParams = 4, kInv = 5;

std::uint32_t lane_width(int n_experts) {
    const int lanes = n_experts < 1 ? 1 : (n_experts > 1024 ? 1024 : n_experts);
    return (std::uint32_t(lanes) + 31u) / 32u * 32u;
}

struct Shape {
    int rows;      // query rows in the fire
    int k;         // experts per token
    int experts;
    int tile_rows;
};

/// One configuration, timed alone. Returns microseconds per dispatch.
double run(RawMetalContext& ctx, Pso pso, const Shape& s, int ordinal,
           bool verify) {
    const int n = s.rows * s.k;
    // What `llama_moe_sorted_rows` computes: per TOUCHED expert, its run
    // rounded up to a whole tile. With distinct experts and tile_rows 1 that is
    // just `n`; the probe mirrors the shape rather than the function so a change
    // to one does not silently redefine the other.
    const int touched = std::min(n, s.experts);
    const int padded = ((n + s.tile_rows - 1) / s.tile_rows) * s.tile_rows +
                       (touched - 1) * 0;  // tile_rows==1 at decode
    const int tiles = padded / (s.tile_rows < 1 ? 1 : s.tile_rows);

    SlotHandle ids = ctx.heap_alloc(std::size_t(n) * sizeof(int));
    SlotHandle perm = ctx.heap_alloc(std::size_t(padded) * sizeof(int));
    SlotHandle rowe = ctx.heap_alloc(std::size_t(padded) * sizeof(int));
    SlotHandle tile = ctx.heap_alloc(std::size_t(tiles) * sizeof(int));
    SlotHandle inv = ctx.heap_alloc(std::size_t(n) * sizeof(int));
    SlotHandle par = ctx.heap_alloc(sizeof(MoeRouteParamsHost));

    // Spread the pairs over the experts the way a real router does, so the
    // per-expert spans are 1 rather than all landing on one expert.
    auto* idp = static_cast<int*>(ids.contents());
    for (int i = 0; i < n; ++i) idp[i] = (i * 7 + 3) % s.experts;
    std::memset(perm.contents(), 0, std::size_t(padded) * sizeof(int));
    std::memset(rowe.contents(), 0, std::size_t(padded) * sizeof(int));
    std::memset(tile.contents(), 0, std::size_t(tiles) * sizeof(int));
    std::memset(inv.contents(), 0, std::size_t(n) * sizeof(int));
    *static_cast<MoeRouteParamsHost*>(par.contents()) = MoeRouteParamsHost{
        std::uint32_t(n),    std::uint32_t(s.experts), std::uint32_t(s.k),
        std::uint32_t(s.tile_rows), std::uint32_t(padded), 2048u, 0u};

    const Kernel kind = Kernel::LlMoeSort;
    ctx.arg_bind(kind, ordinal, kExpertIds, ids);
    ctx.arg_bind(kind, ordinal, kPerm, perm);
    ctx.arg_bind(kind, ordinal, kRowExpert, rowe);
    ctx.arg_bind(kind, ordinal, kTileExpert, tile);
    ctx.arg_bind(kind, ordinal, kParams, par);
    ctx.arg_bind(kind, ordinal, kInv, inv);
    ctx.make_resident();

    const std::uint32_t w = lane_width(s.experts);
    // Barriered, because that is how the real fire runs it -- back to back with
    // nothing between would let the GPU overlap dispatches the driver never
    // allows to overlap.
    constexpr int kReps = 48;  // one model's worth of layers, so the units match
    LatencyHarness h(ctx);
    auto enc = [&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable(kind, ordinal);
        for (int i = 0; i < kReps; ++i) {
            se.dispatch(Grid{w, 1, 1}, Threadgroup{w, 1, 1});
            se.barrier();
        }
    };
    BenchResult r = h.time_step("moe-sort", enc, 30, 8);
    const double us_each = r.median.gpu_exec_ms * 1000.0 / kReps;

    if (verify) {
        // Every pair must land somewhere, exactly once. A sort that silently
        // dropped rows would look FAST, which is the failure mode a timing
        // cannot see.
        const int* invp = static_cast<const int*>(inv.contents());
        const int* permp = static_cast<const int*>(perm.contents());
        std::vector<int> seen(std::size_t(padded), 0);
        int placed = 0, bad = 0;
        for (int i = 0; i < n; ++i) {
            const int at = invp[i];
            if (at < 0 || at >= padded) { ++bad; continue; }
            ++placed;
            ++seen[std::size_t(at)];
            if (permp[at] != i) ++bad;
        }
        for (int t = 0; t < padded; ++t) if (seen[std::size_t(t)] > 1) ++bad;
        std::printf("      verify: %d of %d pairs placed, %d inconsistent%s\n",
                    placed, n, bad, (placed == n && bad == 0) ? "  — OK" : "  <-- WRONG");
    }
    return us_each;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = PIE_METAL_TOOL_KERNELS_DIR;
    if (argc > 1) dir = argv[1];

    printf("moe_route_sort, dispatched ALONE (48 barriered reps, us per dispatch)\n");
    printf("in situ it costs ~42 us a layer; whatever it costs here is the\n");
    printf("kernel itself, and the difference is what the chain does to it.\n\n");

    auto ctx = RawMetalContext::create(/*heap_bytes=*/256ull << 20);
    if (!ctx) { printf("FAIL: no Metal context\n"); return 1; }
    std::string err;
    Pso pso = ctx->compile_pso_from_file(dir + "/moe_route.metal",
                                         "moe_route_sort", &err);
    if (!pso.valid()) { printf("FAIL compile: %s\n", err.c_str()); return 1; }

    // The decode shape, exactly as PIE_METAL_MOE_TRACE reports it.
    const Shape decode{1, 8, 128, 1};
    printf("  decode shape (rows=1 k=8 experts=128 tile=1):\n");
    const double base = run(*ctx, pso, decode, 10, /*verify=*/true);
    printf("      %.2f us per dispatch\n\n", base);

    printf("  scaling with EXPERTS (rows=1 k=8 tile=1):\n");
    for (const int e : {8, 32, 64, 128, 256, 512}) {
        const double t = run(*ctx, pso, Shape{1, 8, e, 1}, 20 + e % 17, false);
        printf("      experts=%-4d lanes=%-4u  %7.2f us\n", e, lane_width(e), t);
    }

    printf("\n  scaling with N = rows*k (experts=128 tile=1):\n");
    for (const int rows : {1, 8, 32, 128}) {
        const double t = run(*ctx, pso, Shape{rows, 8, 128, 1}, 50 + rows % 23, false);
        printf("      rows=%-4d n=%-5d  %7.2f us\n", rows, rows * 8, t);
    }

    // The decode shape AGAIN, last. The first version of this probe reported
    // 14.07 / 9.72 / 6.16 us for three IDENTICAL decode-shape configurations at
    // different points in the run -- a 2.3x spread that looked like a parameter
    // effect and was warm-up. Repeating the first configuration last is what
    // separates the two, exactly as `four_way.sh` does for thermal drift.
    const double again = run(*ctx, pso, decode, 90, /*verify=*/true);
    printf("\n  decode shape REPEATED last: %.2f us (first: %.2f us)\n", again, base);
    if (base > again * 1.25 || again > base * 1.25)
        printf("  >>> the two disagree by more than 25%%: this run is WARM-UP\n"
               "      DRIFT, not parameter scaling. Read the repeat, not the first.\n");
    printf("\n  A cost flat in both sweeps is fixed overhead, not the body.\n");
    return 0;
}
