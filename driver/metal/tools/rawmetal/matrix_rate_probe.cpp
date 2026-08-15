// matrix_rate_probe.cpp — which matrix unit, and how much faster?
//
// The last unmeasured explanation for pie's attention deficit. Everything else
// has been ruled out by measurement: paging (+1.37 ms, 21% of the gap), page
// size (flat), addressing (landed, -0.62 ms), memory layout (padding lost),
// staging (2.71 of 6.90 ms, and the multiply half ALONE is already 2.2x MLX's
// whole kernel), tile shape (pie's IS MLX's), accumulator type and chunking
// (5-6% slower when changed to MLX's form).
//
// What remains is the instruction. pie issues `simdgroup_multiply_accumulate`
// on 8x8 fragments. MLX 0.31.3 also ships kernels on `mpp::tensor_ops::
// matmul2d` with 16x16 fragments -- the M5 neural accelerators -- and this
// machine is an M5 Pro. `nax_probe.metal` already established that pie's
// runtime shader compiler can build for that unit. This prices it.
//
// See `kernels/matrix_rate.metal` for why each kernel runs in its native
// configuration and why both carry two independent accumulator chains.

#include <cstdio>
#include <algorithm>
#include <string>

#include "harness.hpp"
#include "mtl4_context.hpp"

using namespace pie::metal;

namespace {

constexpr int kThreadsPerGroup = 128;   // 4 simdgroups, as pie's attention uses
constexpr int kGroups = 2048;           // enough to fill the machine
constexpr int kIters = 4096;

// Per-simdgroup FLOPs per iteration of each kernel's inner loop.
constexpr double kNaxFlopsPerIter = 2.0 * 2.0 * 16.0 * 32.0 * 16.0;   // 32768
constexpr double kSimdFlopsPerIter = 8.0 * 2.0 * 8.0 * 8.0 * 8.0;     // 8192

double run(RawMetalContext& ctx, Pso pso, const char* label, int ordinal,
           double flops_per_iter, int iters = kIters) {
    const int threads = kGroups * kThreadsPerGroup;
    SlotHandle out = ctx.heap_alloc(size_t(threads) * sizeof(float));
    SlotHandle it = ctx.heap_alloc(sizeof(int));
    *static_cast<int*>(it.contents()) = iters;

    const Kernel kind = Kernel::Argmax;  // a distinct arg table, nothing more
    ctx.arg_bind(kind, ordinal, 0, out);
    ctx.arg_bind(kind, ordinal, 1, it);
    ctx.make_resident();

    Grid grid{uint32_t(threads), 1, 1};
    Threadgroup tg{uint32_t(kThreadsPerGroup), 1, 1};

    LatencyHarness h(ctx);
    auto encode = [&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable(kind, ordinal);
        se.dispatch(grid, tg);
    };
    BenchResult r = h.time_step(label, encode, /*iters=*/20, /*warmup=*/5);

    const double simdgroups = double(kGroups) * (kThreadsPerGroup / 32);
    const double flops = simdgroups * double(iters) * flops_per_iter;
    const double tflops = flops / (r.median.gpu_exec_ms / 1000.0) / 1e12;
    printf("  %-34s  %8.3f ms   %7.2f TFLOP/s\n", label, r.median.gpu_exec_ms, tflops);
    return tflops;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = PIE_METAL_TOOL_LOCAL_KERNELS_DIR;
    if (argc > 1) dir = argv[1];

    printf("Matrix-unit throughput on this device (registers only, no memory)\n\n");

    auto ctx = RawMetalContext::create(/*heap_bytes=*/256ull << 20);
    if (!ctx) {
        printf("FAIL: no Metal context\n");
        return 1;
    }

    std::string err;
    Pso simd = ctx->compile_pso_from_file(dir + "/matrix_rate.metal",
                                          "simdgroup_rate", &err);
    if (!simd.valid()) {
        printf("FAIL simdgroup_rate: %s\n", err.c_str());
        return 1;
    }
    Pso nax = ctx->compile_pso_from_file(dir + "/matrix_rate.metal",
                                         "nax_rate", &err);

    double s = run(*ctx, simd, "simdgroup 8x8 half, 8 chains (pie today)", 0,
                   kSimdFlopsPerIter);
    // Two variants that test WHY this arm reads below what pie's real kernel
    // achieves. More chains answers "latency-bound?"; fp32 answers "is half
    // actually the fast path on this hardware?". The best of the three is the
    // honest simdgroup number, since the question is what the unit can do.
    Pso simd16 = ctx->compile_pso_from_file(dir + "/matrix_rate.metal",
                                            "simdgroup_rate16", &err);
    if (simd16.valid()) {
        s = std::max(s, run(*ctx, simd16, "simdgroup 8x8 half, 16 chains", 2,
                            2.0 * kSimdFlopsPerIter));
    }
    Pso simdf = ctx->compile_pso_from_file(dir + "/matrix_rate.metal",
                                           "simdgroup_rate_f32", &err);
    if (simdf.valid()) {
        s = std::max(s, run(*ctx, simdf, "simdgroup 8x8 FLOAT, 8 chains", 3,
                            kSimdFlopsPerIter));
    }
    if (!nax.valid()) {
        printf("\n  nax_rate did not compile: %s\n", err.c_str());
        printf("  The neural-accelerator path is NOT reachable here.\n");
        return 0;
    }
    const double n = run(*ctx, nax, "mpp::tensor_ops::matmul2d 16x32x16", 1,
                         kNaxFlopsPerIter);

    // PLAUSIBILITY CHECK, and it fails in a way worth printing. pie's shipped
    // attention reaches ~6.9 TFLOP/s on its multiply half (23.4 GFLOP in
    // 3.38 ms, measured by sdpa_paged_probe's no-staging ablation). A real
    // kernel cannot beat the unit's peak, so this microbenchmark UNDERSTATES
    // simdgroup throughput -- it is a floor, not a ceiling. The ratio below is
    // therefore an upper bound; the conservative one uses pie's own achieved
    // rate as the reference instead.
    constexpr double kPieAchieved = 6.9;
    printf("\n  neural accelerators are %.2fx this simdgroup measurement\n", n / s);
    printf("  ...but pie's own kernel achieves %.1f TFLOP/s, ABOVE this %.2f,\n",
           kPieAchieved, s);
    printf("     so the simdgroup figure is a floor. Conservative ratio: %.2fx\n",
           n / kPieAchieved);
    // The gap this would have to close, stated so the answer is judged against
    // the goal rather than against zero -- the mistake this plan already made
    // once with de-paging.
    printf("  pie's attention needs %.2fx on the MULTIPLY half (3.38 -> ~1.2 ms)\n",
           3.38 / 1.2);
    printf("  => %s\n", (n / kPieAchieved) >= (3.38 / 1.2)
             ? "sufficient on its own; a NAX attention kernel is the rewrite"
             : "NOT sufficient on its own; necessary but not the whole story");

    // ── STEP 3 STAGE 1: NAX at attention shapes, operand fill included ──
    {
        std::string e2;
        Pso qk = ctx->compile_pso_from_file(dir + "/sdpa_nax_qk.metal",
                                            "sdpa_nax_qk", &e2);
        if (!qk.valid()) {
            printf("\n(sdpa_nax_qk did not compile: %s)\n", e2.c_str());
        } else {
            const int kCtx = 7424, kRows = 184, kHeads = 32, kBQ = 64, kBK = 32, kD = 128;
            const int tiles = (kRows + kBQ - 1) / kBQ;          // 3
            const int tgs = kHeads * tiles;                     // 96
            SlotHandle o = ctx.get()->heap_alloc(size_t(tgs) * 128 * sizeof(float));
            SlotHandle c = ctx.get()->heap_alloc(sizeof(int));
            *static_cast<int*>(c.contents()) = kCtx;
            const Kernel k = Kernel::Sdpa;
            ctx->arg_bind(k, 9, 0, o);
            ctx->arg_bind(k, 9, 1, c);
            ctx->make_resident();
            Grid g{uint32_t(tgs) * 128u, 1, 1};
            Threadgroup t{128, 1, 1};
            LatencyHarness h(*ctx);
            auto enc = [&](StepEncoder& se) {
                se.set_pso(qk);
                se.set_argtable(k, 9);
                se.dispatch(g, t);
            };
            BenchResult r = h.time_step("nax_qk", enc, 40, 10);
            // Q K^T only: rows x heads x ctx x D x 2. Padded rows (3 tiles of
            // 64 = 192) are what the kernel actually issues, so count those.
            const double flops = double(tiles * kBQ) * kHeads * kCtx * kD * 2.0;
            const double tf = flops / (r.median.gpu_exec_ms / 1000.0) / 1e12;
            printf("\nSTAGE 1 — NAX Q.K^T at the serving shape (BQ=64 BK=32 d=128):\n");
            printf("  %8.3f ms   %7.2f TFLOP/s   (%.1f GFLOP)\n",
                   r.median.gpu_exec_ms, tf, flops / 1e9);
            printf("  vs %.2f TFLOP/s simdgroup ceiling  -> %.2fx\n", s, tf / s);
            printf("  vs %.2f TFLOP/s NAX ceiling (fill-free) -> %.0f%% of it\n",
                   n, 100.0 * tf / n);
            printf("  DECISION: %s\n", (tf / s) >= 3.0
                     ? ">= 3x, the plan holds — proceed to stage 2"
                     : "< 3x, the operand fill eats it — STOP and re-plan");
        }
    }

    // ── Does the rate depend on how long the kernel runs? ──
    //
    // The remaining explanation for the arm reading below pie's achieved rate.
    // A ~50 ms burst of unbroken matrix work is a different power regime from
    // an attention kernel that runs a few ms at a time. If the short runs are
    // faster, this benchmark has been measuring SUSTAINED throughput and pie's
    // kernel lives in the burst regime -- both real, different questions.
    printf("\nDuration sweep — is this throttling rather than issue rate?\n");
    for (int it : {128, 512, 2048, 8192}) {
        char lab[64];
        snprintf(lab, sizeof lab, "simdgroup, iters=%d", it);
        run(*ctx, simd, lab, 4, kSimdFlopsPerIter, it);
    }
    for (int it : {128, 512, 2048, 8192}) {
        char lab[64];
        snprintf(lab, sizeof lab, "NAX,       iters=%d", it);
        run(*ctx, nax, lab, 5, kNaxFlopsPerIter, it);
    }
    return 0;
}
