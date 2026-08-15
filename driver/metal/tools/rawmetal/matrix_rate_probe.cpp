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
#include <cstring>
#include <cmath>
#include <vector>
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
        // Three variants of one kernel, each adding exactly one thing, so a
        // regression can be attributed instead of guessed at.
        const struct { const char* file; const char* label; double flop_mul; } vs[] = {
            {"/sdpa_nax_qk.metal",   "stage 1: Q.K^T only",            1.0},
            {"/sdpa_nax_qkpv.metal", "stage 2: + P.V (no softmax)",    2.0},
            {"/sdpa_nax_full.metal", "stage 2: + softmax (full pass)", 2.0},
            {"/sdpa_nax_staged.metal", "stage 2c: + REAL staging",       2.0},
            {"/sdpa_nax_stageonly.metal", "  ...of which: staging alone",  2.0},
            {"/sdpa_nax_straightk.metal", "stage 2c: K straight, instr transpose", 2.0},
        };
        printf("\nSTEP 3 — NAX attention at the serving shape (BQ=64 BK=32 d=128):\n");
        for (const auto& v : vs) {
            std::string ev;
            Pso p = ctx->compile_pso_from_file(dir + v.file, "sdpa_nax_qk", &ev);
            if (!p.valid()) { printf("  %-32s COMPILE FAIL: %s\n", v.label, ev.c_str()); continue; }
            const int kCtx = 7424, kRows = 184, kHeads = 32, kBQ = 64, kD = 128;
            const int tiles = (kRows + kBQ - 1) / kBQ;
            const int tgs = kHeads * tiles;
            SlotHandle o = ctx->heap_alloc(size_t(tgs) * 128 * sizeof(float));
            SlotHandle c = ctx->heap_alloc(sizeof(int));
            *static_cast<int*>(c.contents()) = kCtx;
            static int ord = 20;
            const Kernel kk = Kernel::Sdpa;
            SlotHandle qd = ctx->heap_alloc(size_t(3) * 64 * 128 * sizeof(uint16_t));
            SlotHandle kd = ctx->heap_alloc(size_t(kCtx) * kD * sizeof(uint16_t));
            SlotHandle vd = ctx->heap_alloc(size_t(kCtx) * kD * sizeof(uint16_t));
            std::memset(qd.contents(), 0, size_t(3) * 64 * 128 * 2);
            std::memset(kd.contents(), 0, size_t(kCtx) * kD * 2);
            std::memset(vd.contents(), 0, size_t(kCtx) * kD * 2);
            ctx->arg_bind(kk, ++ord, 0, o);
            ctx->arg_bind(kk, ord, 1, c);
            ctx->arg_bind(kk, ord, 2, qd);
            ctx->arg_bind(kk, ord, 3, kd);
            ctx->arg_bind(kk, ord, 4, vd);
            ctx->make_resident();
            Grid g{uint32_t(tgs) * 128u, 1, 1};
            Threadgroup t{128, 1, 1};
            LatencyHarness h(*ctx);
            const int myord = ord;
            auto enc = [&](StepEncoder& se) {
                se.set_pso(p); se.set_argtable(kk, myord); se.dispatch(g, t);
            };
            BenchResult r = h.time_step(v.label, enc, 40, 10);
            const double fl = double(tiles * kBQ) * kHeads * kCtx * kD * 2.0 * v.flop_mul;
            const double tf = fl / (r.median.gpu_exec_ms / 1000.0) / 1e12;
            printf("  %-32s %8.3f ms  %6.2f TFLOP/s  %5.2fx simdgroup  x48 = %6.1f ms\n",
                   v.label, r.median.gpu_exec_ms, tf, tf / s,
                   r.median.gpu_exec_ms * 48.0);
        }
        printf("  (shipped 8x8 kernel, whole pass: 6.87 ms/layer; MLX 1.31)\n");

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

    // ── CORRECTNESS: does the kernel compute attention, or just FLOPs? ──
    //
    // Every number above is a timing, and a timing cannot tell a correct
    // Q.K^T from one whose B operand is transposed -- both issue the same
    // matmuls at the same rate. This runs ONE threadgroup over ONE key block
    // with deterministic inputs and checks all 2048 scores against a CPU
    // reference, which is what validates the 16x16 fragment lane mapping and
    // the two-fragment column split.
    {
        std::string ec;
        Pso chk = ctx->compile_pso_from_file(dir + "/sdpa_nax_check.metal",
                                             "sdpa_nax_qk", &ec);
        printf("\nCORRECTNESS — NAX Q.K^T against a CPU reference:\n");
        if (!chk.valid()) { printf("  compile failed: %s\n", ec.c_str()); }
        else {
            const int BQ = 64, BK = 32, D = 128;
            SlotHandle o = ctx->heap_alloc(4 * 32 * 16 * sizeof(float));
            SlotHandle c = ctx->heap_alloc(sizeof(int));
            SlotHandle qd = ctx->heap_alloc(size_t(3) * BQ * D * sizeof(uint16_t));
            SlotHandle kd = ctx->heap_alloc(size_t(BK) * D * sizeof(uint16_t));
            SlotHandle vd = ctx->heap_alloc(size_t(BK) * D * sizeof(uint16_t));
            *static_cast<int*>(c.contents()) = BK;   // exactly one pass
            // bfloat16 by truncation: small exact-in-bf16 values, so the
            // reference is exact and a mismatch is a MAPPING error, never
            // rounding.
            auto bf = [](float f) {
                uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16);
            };
            auto* qp = static_cast<uint16_t*>(qd.contents());
            auto* kp = static_cast<uint16_t*>(kd.contents());
            std::memset(qp, 0, size_t(3) * BQ * D * 2);
            std::memset(vd.contents(), 0, size_t(BK) * D * 2);
            std::vector<float> qf(size_t(BQ) * D), kf(size_t(BK) * D);
            for (int r = 0; r < BQ; ++r)
                for (int d = 0; d < D; ++d) {
                    qf[size_t(r) * D + d] = float((r + 2 * d) % 7) * 0.25f;
                    qp[size_t(r) * D + d] = bf(qf[size_t(r) * D + d]);
                }
            for (int k = 0; k < BK; ++k)
                for (int d = 0; d < D; ++d) {
                    kf[size_t(k) * D + d] = float((3 * k + d) % 5) * 0.5f;
                    kp[size_t(k) * D + d] = bf(kf[size_t(k) * D + d]);
                }
            const Kernel kk = Kernel::Sdpa;
            ctx->arg_bind(kk, 40, 0, o); ctx->arg_bind(kk, 40, 1, c);
            ctx->arg_bind(kk, 40, 2, qd); ctx->arg_bind(kk, 40, 3, kd);
            ctx->arg_bind(kk, 40, 4, vd);
            ctx->make_resident();
            LatencyHarness h(*ctx);
            auto enc = [&](StepEncoder& se) {
                se.set_pso(chk); se.set_argtable(kk, 40);
                se.dispatch(Grid{128, 1, 1}, Threadgroup{128, 1, 1});
            };
            h.time_step("check", enc, 1, 0);

            const float* got = static_cast<const float*>(o.contents());
            int bad = 0; double worst = 0; int wr = -1, wc = -1;
            for (int sg = 0; sg < 4; ++sg)
              for (int lane = 0; lane < 32; ++lane) {
                const int qid = lane >> 2;
                const int fm = (qid & 4) | ((lane >> 1) & 3);
                const int fn = ((qid & 2) | (lane & 1)) * 4;
                for (int e = 0; e < 16; ++e) {
                    // Two 16x16 fragments side by side, each 2 rows x 4 cols.
                    // The alternative -- one 2 rows x 8 consecutive columns
                    // layout -- was TESTED and is worse: 1163 of 2048 wrong
                    // against this mapping's 128. See the plan.
                    const int nf = e / 8, i = (e % 8) / 4, j = e % 4;
                    const int row = sg * 16 + fm + i * 8;
                    const int col = nf * 16 + fn + j;
                    double ref = 0;
                    for (int d = 0; d < D; ++d)
                        ref += double(qf[size_t(row) * D + d]) * double(kf[size_t(col) * D + d]);
                    const double g = got[(size_t(sg) * 32 + lane) * 16 + e];
                    const double err = std::abs(g - ref) / (std::abs(ref) + 1e-6);
                    if (err > 1e-2) { ++bad; if (err > worst) { worst = err; wr = row; wc = col; } }
                }
              }
            if (bad == 0) {
                printf("  all 2048 scores match — fragment mapping and operand\n");
                printf("  orientation are CORRECT. The 1.652 ms is attention.\n");
            } else {
                printf("  %d of 2048 scores WRONG (worst rel %.3f at row %d col %d)\n",
                       bad, worst, wr, wc);
                // The PATTERN is the diagnostic, not the count. A wrong operand
                // orientation, a wrong fragment mapping and a wrong column
                // split each produce a different one.
                int by_e[16] = {0}, by_col[32] = {0}, by_rowmod[16] = {0};
                for (int sg = 0; sg < 4; ++sg)
                  for (int lane = 0; lane < 32; ++lane) {
                    const int qid = lane >> 2;
                    const int fm = (qid & 4) | ((lane >> 1) & 3);
                    const int fn = ((qid & 2) | (lane & 1)) * 4;
                    for (int e = 0; e < 16; ++e) {
                        const int nf = e / 8, i = (e % 8) / 4, j = e % 4;
                        const int row = sg * 16 + fm + i * 8, col = nf * 16 + fn + j;
                        double ref = 0;
                        for (int d = 0; d < D; ++d)
                            ref += double(qf[size_t(row) * D + d]) * double(kf[size_t(col) * D + d]);
                        const double g = got[(size_t(sg) * 32 + lane) * 16 + e];
                        if (std::abs(g - ref) / (std::abs(ref) + 1e-6) > 1e-2) {
                            by_e[e]++; by_col[col]++; by_rowmod[row % 16]++;
                        }
                    }
                  }
                printf("  wrong by element index e :");
                for (int i = 0; i < 16; ++i) printf(" %d", by_e[i]);
                printf("\n  wrong by column         :");
                for (int i = 0; i < 32; ++i) printf(" %d", by_col[i]);
                printf("\n  wrong by row%%16         :");
                for (int i = 0; i < 16; ++i) printf(" %d", by_rowmod[i]);
                printf("\n");
            }
        }
    }

    // Does the documented tensor-slice form even compile here?
    {
        std::string es;
        Pso sl = ctx->compile_pso_from_file(dir + "/nax_slice_qk.metal",
                                            "nax_slice_qk", &es);
        printf("\nTENSOR-SLICE API (the documented form): %s\n",
               sl.valid() ? "COMPILES" : "does not compile");
        if (!sl.valid()) printf("  %s\n", es.c_str());
        else {
            // The payoff: S comes back as plain row-major [BQ][BK]. There is no
            // lane mapping to get wrong, which is the entire reason to prefer
            // this form over hand-filled cooperative tensors.
            const int BQ = 64, BK = 32, D = 128;
            auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };
            SlotHandle qh = ctx->heap_alloc(size_t(BQ) * D * 2);
            SlotHandle kh = ctx->heap_alloc(size_t(D) * BK * 2);
            SlotHandle sh = ctx->heap_alloc(size_t(BQ) * BK * sizeof(float));
            auto* qq = static_cast<uint16_t*>(qh.contents());
            auto* kk2 = static_cast<uint16_t*>(kh.contents());
            std::memset(sh.contents(), 0, size_t(BQ) * BK * 4);
            std::vector<float> qf2(size_t(BQ) * D), kf2(size_t(BK) * D);
            for (int r = 0; r < BQ; ++r)
              for (int d = 0; d < D; ++d) {
                qf2[size_t(r) * D + d] = float((r + 2 * d) % 7) * 0.25f;
                qq[size_t(r) * D + d] = bf(qf2[size_t(r) * D + d]);
              }
            for (int cc = 0; cc < BK; ++cc)
              for (int d = 0; d < D; ++d) {
                kf2[size_t(cc) * D + d] = float((3 * cc + d) % 5) * 0.5f;
                kk2[size_t(d) * BK + cc] = bf(kf2[size_t(cc) * D + d]);   // K^T
              }
            const Kernel ks = Kernel::Sdpa;
            ctx->arg_bind(ks, 60, 0, qh); ctx->arg_bind(ks, 60, 1, kh);
            SlotHandle rh = ctx->heap_alloc(sizeof(int));
            *static_cast<int*>(rh.contents()) = 1;
            ctx->arg_bind(ks, 60, 2, sh); ctx->arg_bind(ks, 60, 3, rh);
            ctx->make_resident();
            LatencyHarness hs(*ctx);
            auto es2 = [&](StepEncoder& se) {
                se.set_pso(sl); se.set_argtable(ks, 60);
                se.dispatch(Grid{128, 1, 1}, Threadgroup{128, 1, 1});
            };
            hs.time_step("slice", es2, 1, 0);
            const float* sg = static_cast<const float*>(sh.contents());
            int bad2 = 0; double worst2 = 0;
            for (int r = 0; r < BQ; ++r)
              for (int cc = 0; cc < BK; ++cc) {
                double ref = 0;
                for (int d = 0; d < D; ++d)
                    ref += double(qf2[size_t(r) * D + d]) * double(kf2[size_t(cc) * D + d]);
                const double g = sg[size_t(r) * BK + cc];
                const double e = std::abs(g - ref) / (std::abs(ref) + 1e-6);
                if (e > 1e-2) { ++bad2; worst2 = e > worst2 ? e : worst2; }
              }
            printf("  correctness: %d of %d scores wrong%s\n", bad2, BQ * BK,
                   bad2 == 0 ? "  — CORRECT" : "");
            if (bad2) printf("  worst relative error %.3f (extent order is the first suspect)\n", worst2);
            // Is the CORRECT form also fast? A hollow win otherwise.
            const int kReps = 232;   // one full 7424-key context at BK=32
            const int kTgs = 96;     // 32 heads x 3 row-tiles, as the other arms
            *static_cast<int*>(rh.contents()) = kReps;
            // ONE THREADGROUP LEAVES THE GPU 99% IDLE. The first version of
            // this measurement dispatched 128 threads against the other arms'
            // 96 threadgroups and reported 0.26 TFLOP/s -- 64x slower than the
            // hand-filled form, which would have been a machine-occupancy
            // artifact reported as an API difference.
            auto er = [&](StepEncoder& se) {
                se.set_pso(sl); se.set_argtable(ks, 60);
                se.dispatch(Grid{uint32_t(kTgs) * 128u, 1, 1}, Threadgroup{128, 1, 1});
            };
            BenchResult rr = hs.time_step("slice-rate", er, 40, 10);
            const double flr = double(kTgs) * kReps * 2.0 * BQ * BK * D;
            const double tfr = flr / (rr.median.gpu_exec_ms / 1000.0) / 1e12;
            printf("  rate: %.3f ms, %d tgs x %d reps  ->  %.2f TFLOP/s  (%.2fx simdgroup;\n",
                   rr.median.gpu_exec_ms, kTgs, kReps, tfr, tfr / s);
            printf("        hand-filled cooperative-tensor form got 16.7)\n");
        }
    }

    // Tile sweep on the CORRECT kernel. 2.30x failed the stage-1 rule; this
    // asks whether that verdict is about the API or about one tile choice.
    {
        printf("\nTILE SWEEP (correct slice kernel, matched occupancy):\n");
        const struct { const char* f; int bq, bk; } ts[] = {
            {"/nax_slice_qk.metal",    64, 32}, {"/nax_slice_128x32.metal", 128, 32},
            {"/nax_slice_64x64.metal", 64, 64}, {"/nax_slice_128x64.metal", 128, 64},
        };
        for (const auto& t : ts) {
            std::string e3;
            Pso p3 = ctx->compile_pso_from_file(dir + t.f, "nax_slice_qk", &e3);
            if (!p3.valid()) { printf("  BQ=%-4d BK=%-3d  compile fail\n", t.bq, t.bk); continue; }
            const int D3 = 128, reps = 7424 / t.bk, tg3 = 32 * ((184 + t.bq - 1) / t.bq);
            SlotHandle q3 = ctx->heap_alloc(size_t(t.bq) * D3 * 2);
            SlotHandle k3 = ctx->heap_alloc(size_t(D3) * t.bk * 2);
            SlotHandle s3 = ctx->heap_alloc(size_t(t.bq) * t.bk * 4);
            SlotHandle r3 = ctx->heap_alloc(sizeof(int));
            std::memset(q3.contents(), 0, size_t(t.bq) * D3 * 2);
            std::memset(k3.contents(), 0, size_t(D3) * t.bk * 2);
            *static_cast<int*>(r3.contents()) = reps;
            static int o3 = 70;
            ++o3;
            const Kernel k3k = Kernel::Sdpa;
            ctx->arg_bind(k3k, o3, 0, q3); ctx->arg_bind(k3k, o3, 1, k3);
            ctx->arg_bind(k3k, o3, 2, s3); ctx->arg_bind(k3k, o3, 3, r3);
            ctx->make_resident();
            LatencyHarness h3(*ctx);
            const int myo = o3;
            auto e3f = [&](StepEncoder& se) {
                se.set_pso(p3); se.set_argtable(k3k, myo);
                se.dispatch(Grid{uint32_t(tg3) * 128u, 1, 1}, Threadgroup{128, 1, 1});
            };
            BenchResult r = h3.time_step("tile", e3f, 30, 8);
            const double fl = double(tg3) * reps * 2.0 * t.bq * t.bk * D3;
            const double tf = fl / (r.median.gpu_exec_ms / 1000.0) / 1e12;
            printf("  BQ=%-4d BK=%-3d  %7.3f ms  %6.2f TFLOP/s  %.2fx simdgroup  (%d tgs)\n",
                   t.bq, t.bk, r.median.gpu_exec_ms, tf, tf / s, tg3);
        }
    }

    // ── WHICH lane layout? A one-hot probe that NAMES it. ──
    //
    // K is an identity (K[c][d] = 1 iff d == c), so S[row][col] must equal
    // Q[row][col] exactly. Run it twice -- once with Q[r][d] = r, once with
    // Q[r][d] = d -- and every output element reports the row and the dim it
    // was actually built from. That names the mapping instead of permuting
    // indices until the error count falls, which finds something that fits
    // rather than something that is right.
    {
        std::string ec2;
        Pso chk = ctx->compile_pso_from_file(dir + "/sdpa_nax_check.metal",
                                             "sdpa_nax_qk", &ec2);
        if (chk.valid()) {
            const int BQ = 64, BK = 32, D = 128;
            auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };
            printf("\nLANE LAYOUT — one-hot probe (K = identity, so S == Q):\n");
            for (int mode = 0; mode < 2; ++mode) {
                SlotHandle o = ctx->heap_alloc(4 * 32 * 16 * sizeof(float));
                SlotHandle c = ctx->heap_alloc(sizeof(int));
                SlotHandle qd = ctx->heap_alloc(size_t(3) * BQ * D * sizeof(uint16_t));
                SlotHandle kd = ctx->heap_alloc(size_t(BK) * D * sizeof(uint16_t));
                SlotHandle vd = ctx->heap_alloc(size_t(BK) * D * sizeof(uint16_t));
                *static_cast<int*>(c.contents()) = BK;
                auto* qp = static_cast<uint16_t*>(qd.contents());
                auto* kp = static_cast<uint16_t*>(kd.contents());
                std::memset(qp, 0, size_t(3) * BQ * D * 2);
                std::memset(kp, 0, size_t(BK) * D * 2);
                std::memset(vd.contents(), 0, size_t(BK) * D * 2);
                for (int r = 0; r < BQ; ++r)
                    for (int d = 0; d < D; ++d)
                        qp[size_t(r) * D + d] = bf(float(mode == 0 ? r : d));
                for (int cc = 0; cc < BK; ++cc) kp[size_t(cc) * D + cc] = bf(1.0f);
                const Kernel kk2 = Kernel::Sdpa;
                const int od = 50 + mode;
                ctx->arg_bind(kk2, od, 0, o); ctx->arg_bind(kk2, od, 1, c);
                ctx->arg_bind(kk2, od, 2, qd); ctx->arg_bind(kk2, od, 3, kd);
                ctx->arg_bind(kk2, od, 4, vd);
                ctx->make_resident();
                LatencyHarness h2(*ctx);
                auto enc2 = [&](StepEncoder& se) {
                    se.set_pso(chk); se.set_argtable(kk2, od);
                    se.dispatch(Grid{128, 1, 1}, Threadgroup{128, 1, 1});
                };
                h2.time_step("onehot", enc2, 1, 0);
                const float* g = static_cast<const float*>(o.contents());
                printf("  Q[r][d] = %s   (element -> value; expect %s)\n",
                       mode == 0 ? "r" : "d", mode == 0 ? "row" : "col");
                // simdgroup 0, the four lanes that were WRONG (0..3) and one
                // that was right (8), first four elements each.
                for (int lane : {0, 1, 2, 3, 8}) {
                    const int qid = lane >> 2;
                    const int fm = (qid & 4) | ((lane >> 1) & 3);
                    const int fn = ((qid & 2) | (lane & 1)) * 4;
                    printf("    lane %2d (fm=%d fn=%2d):", lane, fm, fn);
                    for (int e = 0; e < 8; ++e) {
                        const int nf = e / 8, i = (e % 8) / 4, j = e % 4;
                        printf(" e%d[r%d,c%2d]=%.0f", e, fm + i * 8,
                               nf * 16 + fn + j, double(g[(size_t(0) * 32 + lane) * 16 + e]));
                    }
                    printf("\n");
                }
            }
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
