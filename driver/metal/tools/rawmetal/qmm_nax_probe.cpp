// What does the routed MoE GEMM cost, and does the neural accelerator help?
//
// ## Why this exists
//
// After attention moved to the neural accelerators (4.86x, and past MLX), the
// composition of a cold prefill was re-measured and had shifted underneath the
// old plan (`results-prefill-experiments.md`):
//
//     affine_qmm_t_routed   43.6%   <- the routed MoE GEMM, the largest term
//     sdpa_paged_nax        32.2%
//     affine_qmm_t          19.5%
//
// Both GEMMs still run on the simdgroup matrix unit at ~6.9 TFLOP/s, against
// the 32.5 TFLOP/s `matmul2d` measures on the accelerators. And the routed one
// is not weight-bound at prefill widths: its 128 experts' 4-bit weights are
// ~302 MB per layer, 1.02 ms at this machine's ~296 GB/s, against 44.8 ms
// measured. It is compute-bound, so the unit is available to be changed.
//
// `affine_qmm_t_routed_nax` is that change and nothing else -- same
// `QuantizedBlockLoader` dequantizing 4-bit weights into threadgroup memory,
// same `BlockLoader` staging the activations, same two-fence K loop, same
// expert slice. Only `BlockMMA` becomes `NaxBlockMMA`.
//
// ## What this probe insists on
//
// Correctness against a float64 reference BEFORE any timing, on a small shape,
// with distinct data per expert so a kernel that computed the right arithmetic
// against the wrong expert slice cannot pass. That specific failure is what the
// `tile_expert` indirection makes easy to get wrong, and a timing cannot see it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"
#include "mtl4_context.hpp"

using namespace pie::metal;

namespace {

// Qwen3-Coder-30B-A3B's routed FFN, from the served artifact's config.
constexpr int kHidden = 2048;   // K for gate/up
constexpr int kMoeInter = 768;  // N for gate/up
constexpr int kExperts = 128;
constexpr int kGroup = 64;      // affine quantization group
constexpr int kBits = 4;

template <typename T>
SlotHandle scalar(RawMetalContext& ctx, T v) {
    SlotHandle h = ctx.heap_alloc(sizeof(T));
    *static_cast<T*>(h.contents()) = v;
    return h;
}

inline uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t(u >> 16);
}
inline float unbf(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Arm {
    const char* label;
    Pso pso;
    int bm;
    int bn;
};

// One routed GEMM: `rows` sorted rows over `experts` experts, N x K per expert.
//
// The weight layout is the contract the kernel declares: a flat
// [n_experts * N, K] matrix, expert-major, 4-bit packed two to a byte along K,
// with per-group scales and biases at [n_experts * N, K / group].
double routed_run(RawMetalContext& ctx, const Arm& a, int rows, int N, int K,
                  int experts, int ordinal, bool check) {
    const int bm = a.bm, bn = a.bn;
    const int row_tiles = rows / bm;
    const size_t wbytes = size_t(experts) * N * K / 2;
    const size_t gcount = size_t(experts) * N * (K / kGroup);

    SlotHandle w = ctx.heap_alloc(wbytes);
    SlotHandle sc = ctx.heap_alloc(gcount * 2);
    SlotHandle bi = ctx.heap_alloc(gcount * 2);
    SlotHandle x = ctx.heap_alloc(size_t(rows) * K * 2);
    SlotHandle y = ctx.heap_alloc(size_t(rows) * N * 2);
    SlotHandle te = ctx.heap_alloc(size_t(row_tiles) * sizeof(int));
    auto* wz = static_cast<uint8_t*>(w.contents());
    auto* scz = static_cast<uint16_t*>(sc.contents());
    auto* biz = static_cast<uint16_t*>(bi.contents());
    auto* xz = static_cast<uint16_t*>(x.contents());
    std::memset(y.contents(), 0, size_t(rows) * N * 2);

    // Distinct per expert, per output column and per input column, so a wrong
    // expert slice, a wrong column or a transposed operand all show up.
    for (size_t i = 0; i < wbytes; ++i) wz[i] = uint8_t((i * 7 + (i >> 5) * 3) & 0xff);
    for (size_t g = 0; g < gcount; ++g) {
        scz[g] = bf(float((g % 5) + 1) * 0.03125f);
        biz[g] = bf(float(int(g % 3) - 1) * 0.25f);
    }
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k)
            xz[size_t(r) * K + k] = bf(float((r * 3 + k * 2) % 7) * 0.125f);
    // Tile t belongs to expert t % experts, so every expert is exercised and
    // adjacent tiles disagree -- the arrangement most likely to expose an
    // off-by-one in the expert slice.
    auto* tez = static_cast<int*>(te.contents());
    for (int t = 0; t < row_tiles; ++t) tez[t] = t % experts;

    const Kernel kind = Kernel::LlExpertGate;
    ctx.arg_bind(kind, ordinal, 0, w);
    ctx.arg_bind(kind, ordinal, 1, sc);
    ctx.arg_bind(kind, ordinal, 2, bi);
    ctx.arg_bind(kind, ordinal, 3, x);
    ctx.arg_bind(kind, ordinal, 4, y);
    ctx.arg_bind(kind, ordinal, 5, scalar<int>(ctx, K));
    ctx.arg_bind(kind, ordinal, 6, scalar<int>(ctx, N));
    ctx.arg_bind(kind, ordinal, 12, te);
    ctx.make_resident();

    Grid grid{uint32_t(N / bn) * 128u, uint32_t(row_tiles), 1};
    Threadgroup tg{128, 1, 1};
    LatencyHarness h(ctx);
    auto enc = [&](StepEncoder& se) {
        se.set_pso(a.pso);
        se.set_argtable(kind, ordinal);
        se.dispatch(grid, tg);
    };
    BenchResult r = h.time_step("qmm", enc, check ? 1 : 20, check ? 0 : 5);

    if (check) {
        const uint16_t* yg = static_cast<const uint16_t*>(y.contents());
        int bad = 0;
        double worst = 0;
        long long checked = 0;
        // Every row of a few tiles, all columns -- enough to catch an expert
        // or column error without an O(rows*N*K) sweep of the whole shape.
        for (int t = 0; t < row_tiles && t < 6; ++t) {
            const int e = tez[t];
            for (int rr = 0; rr < bm; ++rr) {
                const int row = t * bm + rr;
                for (int n = 0; n < N; n += 37) {
                    double ref = 0;
                    for (int k = 0; k < K; ++k) {
                        const size_t widx = (size_t(e) * N + n) * K + k;
                        const uint8_t byte = wz[widx / 2];
                        const int q = (widx & 1) ? (byte >> 4) : (byte & 0x0f);
                        const size_t g = (size_t(e) * N + n) * (K / kGroup) + k / kGroup;
                        const double wv = double(q) * unbf(scz[g]) + unbf(biz[g]);
                        ref += double(unbf(xz[size_t(row) * K + k])) * wv;
                    }
                    const double got = unbf(yg[size_t(row) * N + n]);
                    const double err = std::abs(got - ref) / (std::abs(ref) + 1e-3);
                    ++checked;
                    if (err > 3e-2) { ++bad; worst = std::max(worst, err); }
                }
            }
        }
        printf("    %-10s correctness: %d of %lld wrong%s", a.label, bad, checked,
               bad == 0 ? "  — CORRECT\n" : "\n");
        if (bad) printf("      worst relative %.4f\n", worst);
    }
    return r.median.gpu_exec_ms;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = PIE_METAL_TOOL_KERNELS_DIR;
    if (argc > 1) dir = argv[1];

    printf("Routed MoE GEMM: simdgroup matrix unit against the neural accelerators\n");
    printf("Qwen3-Coder-30B-A3B routed FFN: K=%d N=%d, %d experts, 4-bit affine g%d\n\n",
           kHidden, kMoeInter, kExperts, kGroup);

    auto ctx = RawMetalContext::create(/*heap_bytes=*/6144ull << 20);
    if (!ctx) { printf("FAIL: no Metal context\n"); return 1; }

    std::string e1, e2;
    Arm shipped{"shipped", ctx->compile_pso_from_file(
                    dir + "/quantized_qmm_t.metal",
                    "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64", &e1), 32, 64};
    Arm nax{"NAX", ctx->compile_pso_from_file(
                dir + "/quantized_qmm_t.metal",
                "affine_qmm_t_routed_nax_bfloat16_gs_64_b_4_bm_32_bn_64", &e2), 32, 64};
    if (!shipped.pso.valid()) { printf("FAIL shipped compile: %s\n", e1.c_str()); return 1; }
    if (!nax.pso.valid()) { printf("FAIL NAX compile: %s\n", e2.c_str()); return 1; }

    // Correctness first, on a shape small enough for a float64 reference.
    printf("correctness (8 experts, 256 rows, N=128, K=256):\n");
    routed_run(*ctx, shipped, 256, 128, 256, 8, 10, /*check=*/true);
    routed_run(*ctx, nax, 256, 128, 256, 8, 11, /*check=*/true);

    // Then the serving shape. A 4096-row prefill fire routes 8 of 128 experts
    // per token, so the sorted stack is 4096*8 rows over 128 experts.
    printf("\nserving shape (4096-row fire -> %d sorted rows over %d experts):\n",
           4096 * 8, kExperts);
    const double ts = routed_run(*ctx, shipped, 4096 * 8, kMoeInter, kHidden,
                                 kExperts, 20, false);
    const double tn = routed_run(*ctx, nax, 4096 * 8, kMoeInter, kHidden,
                                 kExperts, 21, false);
    const double flop = 2.0 * double(4096 * 8) * kMoeInter * kHidden;
    printf("  %-10s %8.3f ms   %6.2f TFLOP/s\n", "shipped", ts, flop / (ts / 1000.0) / 1e12);
    printf("  %-10s %8.3f ms   %6.2f TFLOP/s   %.2fx\n", "NAX", tn,
           flop / (tn / 1000.0) / 1e12, ts / tn);
    // The driver picks bm from the row count (`moe_bm_slot`), so the other
    // tile it can select has to be measured too -- a NAX path that only covers
    // one of them would silently fall back at the other width.
    printf("\n  the other tile the driver can select (bm=64):\n");
    {
        std::string e3, e4;
        Arm s64{"shipped64", ctx->compile_pso_from_file(
                    dir + "/quantized_qmm_t.metal",
                    "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_64", &e3), 64, 64};
        Arm n64{"NAX64", ctx->compile_pso_from_file(
                    dir + "/quantized_qmm_t.metal",
                    "affine_qmm_t_routed_nax_bfloat16_gs_64_b_4_bm_64_bn_64", &e4), 64, 64};
        if (!s64.pso.valid()) printf("    shipped bm=64 compile: %s\n", e3.c_str());
        if (!n64.pso.valid()) printf("    NAX bm=64 compile: %s\n", e4.c_str());
        if (s64.pso.valid() && n64.pso.valid()) {
            routed_run(*ctx, s64, 256, 128, 256, 8, 30, true);
            routed_run(*ctx, n64, 256, 128, 256, 8, 31, true);
            const double a = routed_run(*ctx, s64, 4096 * 8, kMoeInter, kHidden, kExperts, 32, false);
            const double b = routed_run(*ctx, n64, 4096 * 8, kMoeInter, kHidden, kExperts, 33, false);
            printf("  %-10s %8.3f ms   %6.2f TFLOP/s\n", "shipped64", a, flop / (a / 1000.0) / 1e12);
            printf("  %-10s %8.3f ms   %6.2f TFLOP/s   %.2fx\n", "NAX64", b,
                   flop / (b / 1000.0) / 1e12, a / b);
        }
    }
    printf("\n  simdgroup ceiling 5.48 TFLOP/s, matmul2d 32.5 (matrix_rate_probe).\n");
    printf("  This is ONE of the three routed projections; a layer runs gate,\n");
    printf("  up and down, and the trace prices all three together at 43.6%% of\n");
    printf("  a cold prefill.\n");
    return 0;
}
