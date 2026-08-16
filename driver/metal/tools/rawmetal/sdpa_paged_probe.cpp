// sdpa_paged_probe.cpp — what does pie's PAGED attention cost, on its own?
//
// ## Why this exists
//
// A 184-row prefill fire at 7424 context takes 572 ms, and
// `PIE_METAL_DISPATCH_TRACE` attributes 42.70% of it to
// `sdpa_paged_mma_bfloat16_d_128`. That share is the only number we have for
// attention, and it comes from a trace that SERIALIZES the fire — its own
// banner says "shares point at kernels; they do not price changes". So the
// 42.70% is an estimate derived from a distorted run, and every conclusion
// resting on it (including "pie's attention is 3.3x MLX's") inherits that.
//
// This prices it directly: one PSO, one argument table, one dispatch, timed in
// isolation by `LatencyHarness::bench_kernel` — the rig whose own comment says
// it exists "so delta can A/B each ported kernel's exec-ms against MLX's
// kernel". Same shapes MLX was measured on:
//
//     184 queries x 32 heads x 128 dim, against 7424 keys x 4 KV heads,
//     bfloat16, page size 32 — Qwen3-Coder-30B-A3B's geometry.
//
// MLX's `mx.fast.scaled_dot_product_attention` on those shapes: 1.555 ms.
// If this prints something near 5.1 ms (244 ms / 48 layers) the share was
// honest and the 3.3x gap is real. If it prints ~1.5 ms, the share was an
// artifact of serialization and pie's attention is fine — and the prefill
// deficit is somewhere nobody has looked yet.
//
// The comparison is NOT like for like and is not meant to be: MLX reads a
// CONTIGUOUS KV cache and this reads a PAGE TABLE. That difference is the
// thing being priced. Page granularity is already known not to matter (32 /
// 64 / 128 / 256 tokens per page give 572 / 571 / 586 / 572 ms end to end),
// so whatever this costs is the kernel, not the indirection.

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "harness.hpp"
#include "mtl4_context.hpp"
#include "model/qwen3_5/decode_dispatch_mb.hpp"

using namespace pie::metal;

namespace {

// Qwen3-Coder-30B-A3B, read from the served artifact's own config.
constexpr int kQHeads = 32;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 32;
constexpr int kLayers = 48;
// Dispatches fused into one command buffer, to amortize the per-CB sync floor.
constexpr int kRepeats = 48;

struct Shape {
    int rows;  // query rows in the fire
    int ctx;   // keys already in the cache
    const char* label;
};

// A scalar operand is still a bound buffer: the kernel declares
// `constant int& page_size [[buffer(9)]]`, so it needs four bytes at slot 9,
// not an inline constant.
template <typename T>
SlotHandle scalar(RawMetalContext& ctx, T v) {
    SlotHandle h = ctx.heap_alloc(sizeof(T));
    *static_cast<T*>(h.contents()) = v;
    return h;
}

// `Contig` shares the MMA launch shape -- it IS the MMA kernel, recompiled.
enum class Path { Mma, Tiled, Contig };

// ## The de-paging question, and why it is asked THIS way
//
// The obvious experiment -- gather the pages into scratch, then run a
// contiguous kernel -- cannot be run honestly with what pie has, because pie's
// only contiguous attention kernel (`sdpa_vector`) is a DECODE kernel: one
// threadgroup per (head, query row), each re-reading the entire KV. At 184
// rows x 32 heads that is 5888 threadgroups each streaming 3.8 MB, ~22 GB of
// reads for a fire whose KV is 15 MB. It would lose by a mile, and it would
// lose for a reason that has nothing to do with paging.
//
// The second thing that shapes this: the probe binds an IDENTITY page table,
// so the paged kernel is ALREADY reading contiguous, sequential memory. There
// is no locality left for de-paging to recover -- the 7.47 ms is what the
// kernel costs on the best possible layout. So the only thing de-paging can
// remove is the ADDRESSING: an integer divide, a modulo and a dependent load,
// executed once per staged ELEMENT (2048 per 16-key tile, for 16 distinct
// values).
//
// Hence the split below. `Path::Contig` is the same kernel compiled with
// `PIE_SDPA_CONTIG_KV`, which replaces exactly those three operations with
// `slot = kp` and changes nothing else -- same tile shape, same staging, same
// fragments, same bytes touched. Its delta against `Path::Mma` IS the paging
// tax. `depage_cost` prices the gather that would have to be paid to earn it.
// If (tax - gather) is small, "de-page then compute" cannot pay, whatever the
// contiguous kernel is, and the gap is the kernel's shape.

double run_one(RawMetalContext& ctx, Pso pso, const Shape& s, int layer, Path path) {
    const int n = s.rows;
    const int ctx_len = s.ctx;
    // Pages must cover the cache AND the rows this fire writes past it.
    const int pages = (ctx_len + n + kPageSize - 1) / kPageSize;

    const size_t tsz = 2;  // bfloat16
    const size_t q_bytes = size_t(n) * kQHeads * kHeadDim * tsz;
    const size_t page_bytes = size_t(pages) * kPageSize * kKvHeads * kHeadDim * tsz;

    SlotHandle q = ctx.heap_alloc(q_bytes);
    SlotHandle kp = ctx.heap_alloc(page_bytes);
    SlotHandle vp = ctx.heap_alloc(page_bytes);
    SlotHandle out = ctx.heap_alloc(q_bytes);
    std::memset(q.contents(), 0, q_bytes);
    std::memset(kp.contents(), 0, page_bytes);
    std::memset(vp.contents(), 0, page_bytes);

    // Positions: this fire's rows sit at ctx_len .. ctx_len+n, so every row's
    // causal bound covers the whole cache — the expensive case, and the one a
    // prefill actually runs.
    SlotHandle pos = ctx.heap_alloc(size_t(n) * sizeof(int));
    SlotHandle req = ctx.heap_alloc(size_t(n) * sizeof(int));
    for (int i = 0; i < n; ++i) {
        static_cast<int*>(pos.contents())[i] = ctx_len + i;
        static_cast<int*>(req.contents())[i] = 0;  // one request
    }

    // Page CSR for a single request: identity page list, one span.
    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    SlotHandle pindptr = ctx.heap_alloc(2 * sizeof(uint32_t));
    for (int p = 0; p < pages; ++p) static_cast<uint32_t*>(pidx.contents())[p] = uint32_t(p);
    static_cast<uint32_t*>(pindptr.contents())[0] = 0;
    static_cast<uint32_t*>(pindptr.contents())[1] = uint32_t(pages);

    // No user mask: `attention_mask_enabled[row] == 0` makes the kernel take
    // its causal bound from `position_ids` alone.
    SlotHandle mask = ctx.heap_alloc(16);
    SlotHandle mask_on = ctx.heap_alloc(size_t(n));
    std::memset(mask.contents(), 0, 16);
    std::memset(mask_on.contents(), 0, size_t(n));
    SlotHandle sinks = ctx.heap_alloc(16);  // unused in the no-sink variant
    std::memset(sinks.contents(), 0, 16);

    const float scale = 1.0f / 11.3137085f;  // 1/sqrt(128)
    SlotHandle gqa = scalar<int>(ctx, kQHeads / kKvHeads);
    SlotHandle psz = scalar<int>(ctx, kPageSize);
    SlotHandle nkv = scalar<int>(ctx, kKvHeads);
    SlotHandle scl = scalar<float>(ctx, scale);
    SlotHandle mstride = scalar<uint32_t>(ctx, 0u);
    SlotHandle window = scalar<int>(ctx, 0);  // 0 = no sliding window
    SlotHandle nrows = scalar<int>(ctx, n);

    const Kernel kind = Kernel::SdpaPaged;
    ctx.arg_bind(kind, layer, 0, q);
    ctx.arg_bind(kind, layer, 1, kp);
    ctx.arg_bind(kind, layer, 2, vp);
    ctx.arg_bind(kind, layer, 3, out);
    ctx.arg_bind(kind, layer, 4, gqa);
    ctx.arg_bind(kind, layer, 5, pos);
    ctx.arg_bind(kind, layer, 6, req);
    ctx.arg_bind(kind, layer, 7, pidx);
    ctx.arg_bind(kind, layer, 8, pindptr);
    ctx.arg_bind(kind, layer, 9, psz);
    ctx.arg_bind(kind, layer, 10, nkv);
    ctx.arg_bind(kind, layer, 11, scl);
    ctx.arg_bind(kind, layer, 12, mask);
    ctx.arg_bind(kind, layer, 13, mstride);
    ctx.arg_bind(kind, layer, 14, mask_on);
    ctx.arg_bind(kind, layer, 15, window);
    ctx.arg_bind(kind, layer, 16, sinks);
    ctx.arg_bind(kind, layer, 17, nrows);
    ctx.make_resident();

    // The two kernels have DIFFERENT launch shapes -- 128 threads per
    // threadgroup for the matrix path, 1024 for the tiled one. Using the MMA
    // shape for both is how the first run of this probe ranked tiled FASTER
    // than MMA, which the end-to-end A/B (572 ms vs 1240 ms) says is backwards.
    Grid grid;
    Threadgroup tg;
    if (path == Path::Tiled) sdpa_paged_tiled_dispatch(kQHeads, n, grid, tg);
    else                     sdpa_paged_mma_dispatch(kQHeads, n, grid, tg);

    // AMORTIZED, not single-dispatch. `bench_kernel`'s gpu_exec_ms includes
    // launch + sync per dispatch, and at these shapes that floor DOMINATES:
    // it priced a 1-row fire's attention at 3.536 ms/layer, which would make
    // attention alone 170 ms of a decode fire that measures 21 ms end to end,
    // and it ranked the tiled kernel FASTER than the MMA one that is 2.1x
    // faster in a real fire. Both impossibilities are the sync floor, not the
    // kernel. Repeating the dispatch inside ONE command buffer strips it and
    // leaves compute + barrier -- what a fused 48-layer fire actually pays.
    LatencyHarness h(ctx);
    auto encode = [&](StepEncoder& se) {
        se.set_pso(pso);
        for (int i = 0; i < kRepeats; ++i) {
            se.set_argtable(kind, layer);
            se.dispatch(grid, tg);
            se.barrier();
        }
    };
    BenchResult r = h.time_step(s.label, encode, /*iters=*/40, /*warmup=*/10);
    const double per_layer = r.median.gpu_exec_ms / kRepeats;
    // The KV this fire must read, once per 32-row query tile: that is the
    // kernel's own decomposition, and it is the number a bandwidth claim has
    // to be made against.
    const int tiles = (n + 31) / 32;
    const double gb = double(tiles) * double(ctx_len) * kKvHeads * kHeadDim * 2.0 * 2.0 / 1e9;
    printf("  %-26s  %7.3f ms/layer   x%d layers = %7.1f ms   (KV read %5.2f GB -> %6.1f GB/s)\n",
           s.label, per_layer, kLayers, per_layer * kLayers, gb,
           per_layer > 0 ? gb / (per_layer / 1000.0) : 0.0);
    return per_layer;
}

// Prices materializing one request's paged KV into a contiguous run: the
// cost side of the de-paging trade, in the same units as the kernels above.
double depage_cost(RawMetalContext& ctx, Pso pso, int ctx_len, int layer) {
    const int pages = (ctx_len + kPageSize - 1) / kPageSize;
    const int vec_per_key = kKvHeads * kHeadDim * 2 / 16;  // bfloat16 -> uint4s
    const size_t page_bytes = size_t(pages) * kPageSize * kKvHeads * kHeadDim * 2;
    const size_t out_bytes = size_t(ctx_len) * kKvHeads * kHeadDim * 2;

    SlotHandle kp = ctx.heap_alloc(page_bytes);
    SlotHandle vp = ctx.heap_alloc(page_bytes);
    SlotHandle ko = ctx.heap_alloc(out_bytes);
    SlotHandle vo = ctx.heap_alloc(out_bytes);
    std::memset(kp.contents(), 0, page_bytes);
    std::memset(vp.contents(), 0, page_bytes);

    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    for (int p = 0; p < pages; ++p) static_cast<uint32_t*>(pidx.contents())[p] = uint32_t(p);

    const Kernel kind = Kernel::KvAppendPaged;  // a distinct arg table, nothing more
    ctx.arg_bind(kind, layer, 0, kp);
    ctx.arg_bind(kind, layer, 1, vp);
    ctx.arg_bind(kind, layer, 2, ko);
    ctx.arg_bind(kind, layer, 3, vo);
    ctx.arg_bind(kind, layer, 4, pidx);
    ctx.arg_bind(kind, layer, 5, scalar<int>(ctx, 0));
    ctx.arg_bind(kind, layer, 6, scalar<int>(ctx, kPageSize));
    ctx.arg_bind(kind, layer, 7, scalar<int>(ctx, vec_per_key));
    ctx.arg_bind(kind, layer, 8, scalar<int>(ctx, ctx_len));
    ctx.make_resident();

    const uint32_t threads = uint32_t(ctx_len) * uint32_t(vec_per_key);
    Grid grid{((threads + 255u) / 256u) * 256u, 1, 1};
    Threadgroup tg{256, 1, 1};

    LatencyHarness h(ctx);
    auto encode = [&](StepEncoder& se) {
        se.set_pso(pso);
        for (int i = 0; i < kRepeats; ++i) {
            se.set_argtable(kind, layer);
            se.dispatch(grid, tg);
            se.barrier();
        }
    };
    BenchResult r = h.time_step("depage", encode, /*iters=*/40, /*warmup=*/10);
    const double per_layer = r.median.gpu_exec_ms / kRepeats;
    // Read K and V, write K and V: four times the contiguous size.
    const double gb = 4.0 * double(out_bytes) / 1e9;
    printf("  %-26s  %7.3f ms/layer   x%d layers = %7.1f ms   (moved %5.2f GB -> %6.1f GB/s)\n",
           "gather pages -> contig", per_layer, kLayers, per_layer * kLayers, gb,
           per_layer > 0 ? gb / (per_layer / 1000.0) : 0.0);
    return per_layer;
}

}  // namespace

// One k-row decode fire: correctness against a CPU reference, then its cost.
// KROWS=1 is the baseline -- the same kernel doing what `sdpa_paged_decode`
// does -- so the k/1 ratio isolates the KV sharing and nothing else.
double krow_run(RawMetalContext& ctx, Pso pso, int krows, int ctx_len, int ordinal,
                bool check) {
    constexpr int kHeads = 32, kKv = 4, kD = 128, kPage = 32;
    const int pages = (ctx_len + krows + kPage - 1) / kPage;
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };

    SlotHandle q = ctx.heap_alloc(size_t(krows) * kHeads * kD * 2);
    SlotHandle kp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle vp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle o  = ctx.heap_alloc(size_t(krows) * kHeads * kD * 2);
    SlotHandle pos = ctx.heap_alloc(size_t(krows) * sizeof(int));
    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    auto* qz = static_cast<uint16_t*>(q.contents());
    auto* kz = static_cast<uint16_t*>(kp.contents());
    auto* vz = static_cast<uint16_t*>(vp.contents());
    std::memset(kz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(vz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(o.contents(), 0, size_t(krows) * kHeads * kD * 2);
    for (int p2 = 0; p2 < pages; ++p2) static_cast<uint32_t*>(pidx.contents())[p2] = uint32_t(p2);
    for (int r = 0; r < krows; ++r) static_cast<int*>(pos.contents())[r] = ctx_len + r;

    std::vector<float> qf(size_t(krows) * kD), kf(size_t(ctx_len + krows) * kD), vf(kf.size());
    for (int r = 0; r < krows; ++r)
      for (int d = 0; d < kD; ++d) {
        qf[size_t(r)*kD+d] = float((r + d) % 5) * 0.125f;
        for (int h = 0; h < kHeads; ++h) qz[(size_t(r)*kHeads + h)*kD + d] = bf(qf[size_t(r)*kD+d]);
      }
    for (int c = 0; c < ctx_len + krows; ++c)
      for (int d = 0; d < kD; ++d) {
        kf[size_t(c)*kD+d] = float((c + 2*d) % 4) * 0.125f;
        vf[size_t(c)*kD+d] = float((3*c + d) % 6) * 0.25f;
        for (int h = 0; h < kKv; ++h) {
            kz[(size_t(c)*kKv + h)*kD + d] = bf(kf[size_t(c)*kD+d]);
            vz[(size_t(c)*kKv + h)*kD + d] = bf(vf[size_t(c)*kD+d]);
        }
      }

    const float scale = 1.0f / 11.3137085f;
    const Kernel kind = Kernel::Sdpa;
    ctx.arg_bind(kind, ordinal, 0, q);   ctx.arg_bind(kind, ordinal, 1, kp);
    ctx.arg_bind(kind, ordinal, 2, vp);  ctx.arg_bind(kind, ordinal, 3, o);
    ctx.arg_bind(kind, ordinal, 4, scalar<int>(ctx, kHeads / kKv));
    ctx.arg_bind(kind, ordinal, 5, pos); ctx.arg_bind(kind, ordinal, 6, pidx);
    ctx.arg_bind(kind, ordinal, 7, scalar<int>(ctx, kPage));
    ctx.arg_bind(kind, ordinal, 8, scalar<int>(ctx, kKv));
    ctx.arg_bind(kind, ordinal, 9, scalar<float>(ctx, scale));
    ctx.make_resident();

    Grid grid{uint32_t(kHeads) * 1024u, 1, 1};
    Threadgroup tg{1024, 1, 1};
    LatencyHarness h(ctx);
    auto enc = [&](StepEncoder& se) {
        se.set_pso(pso); se.set_argtable(kind, ordinal); se.dispatch(grid, tg);
    };
    BenchResult r = h.time_step("krow", enc, check ? 1 : 30, check ? 0 : 8);

    if (check) {
        const uint16_t* og = static_cast<const uint16_t*>(o.contents());
        int bad = 0; double worst = 0;
        for (int rr = 0; rr < krows; ++rr) {
            const int hi = ctx_len + rr;
            std::vector<double> sc(size_t(hi) + 1); double mx = -1e30;
            for (int c = 0; c <= hi; ++c) {
                double a2 = 0;
                for (int d = 0; d < kD; ++d) a2 += double(qf[size_t(rr)*kD+d]) * double(kf[size_t(c)*kD+d]);
                sc[c] = a2 * double(scale); if (sc[c] > mx) mx = sc[c];
            }
            double sm = 0;
            for (int c = 0; c <= hi; ++c) { sc[c] = std::exp(sc[c] - mx); sm += sc[c]; }
            for (int d = 0; d < kD; ++d) {
                double ref = 0;
                for (int c = 0; c <= hi; ++c) ref += sc[c] * double(vf[size_t(c)*kD+d]);
                ref /= sm;
                uint32_t bits = uint32_t(og[(size_t(rr)*kHeads + 0)*kD + d]) << 16;
                float got; std::memcpy(&got, &bits, 4);
                const double e = std::abs(double(got) - ref) / (std::abs(ref) + 1e-6);
                if (e > 3e-2) { ++bad; worst = e > worst ? e : worst; }
            }
        }
        printf("  KROWS=%d correctness: %d of %d wrong%s", krows, bad, krows * kD,
               bad == 0 ? "  \xe2\x80\x94 CORRECT\n" : "");
        if (bad) printf("   (worst %.4f)\n", worst);
    }
    return r.median.gpu_exec_ms;
}

// One head-sharing decode fire: correctness against a CPU reference, then cost.
//
// The data here is DISTINCT PER HEAD, deliberately, and that is the difference
// between this and `krow_run` above. `krow_run` writes the same q to all 32
// query heads and the same k/v to all 4 kv heads, which is fine for what it
// tests -- but under that data a kernel that computed head 3 and stored it as
// head 5 would be byte-perfect. The whole hazard of a head-sharing kernel is
// exactly that mix-up, so the reference has to be able to see it: every query
// head gets its own q, every kv head its own k/v, and every head is checked.
double hshare_run(RawMetalContext& ctx, Pso pso, int heads, int ctx_len, int ordinal,
                  bool check) {
    constexpr int kHeads = 32, kKv = 4, kD = 128, kPage = 32;
    const int gqa = kHeads / kKv;                 // 8
    const int pages = (ctx_len + kPage) / kPage + 1;
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };

    SlotHandle q = ctx.heap_alloc(size_t(kHeads) * kD * 2);
    SlotHandle kp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle vp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle o  = ctx.heap_alloc(size_t(kHeads) * kD * 2);
    SlotHandle pos = ctx.heap_alloc(sizeof(int));
    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    auto* qz = static_cast<uint16_t*>(q.contents());
    auto* kz = static_cast<uint16_t*>(kp.contents());
    auto* vz = static_cast<uint16_t*>(vp.contents());
    std::memset(kz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(vz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(o.contents(), 0, size_t(kHeads) * kD * 2);
    for (int p2 = 0; p2 < pages; ++p2) static_cast<uint32_t*>(pidx.contents())[p2] = uint32_t(p2);
    *static_cast<int*>(pos.contents()) = ctx_len;

    // qf[h][d] and kf[kvh][c][d] -- head-dependent, so a swap shows up.
    std::vector<float> qf(size_t(kHeads) * kD);
    std::vector<float> kf(size_t(kKv) * (ctx_len + 1) * kD), vf(kf.size());
    for (int h = 0; h < kHeads; ++h)
      for (int d = 0; d < kD; ++d) {
        qf[size_t(h) * kD + d] = float((h * 3 + d * 2) % 7) * 0.125f;
        qz[size_t(h) * kD + d] = bf(qf[size_t(h) * kD + d]);
      }
    for (int c = 0; c <= ctx_len; ++c)
      for (int h = 0; h < kKv; ++h)
        for (int d = 0; d < kD; ++d) {
          const float kk = float((c + 2 * d + 5 * h) % 4) * 0.125f;
          const float vv = float((3 * c + d + 11 * h) % 6) * 0.25f;
          kf[(size_t(h) * (ctx_len + 1) + c) * kD + d] = kk;
          vf[(size_t(h) * (ctx_len + 1) + c) * kD + d] = vv;
          kz[(size_t(c) * kKv + h) * kD + d] = bf(kk);
          vz[(size_t(c) * kKv + h) * kD + d] = bf(vv);
        }

    const float scale = 1.0f / 11.3137085f;
    const Kernel kind = Kernel::Sdpa;
    ctx.arg_bind(kind, ordinal, 0, q);   ctx.arg_bind(kind, ordinal, 1, kp);
    ctx.arg_bind(kind, ordinal, 2, vp);  ctx.arg_bind(kind, ordinal, 3, o);
    ctx.arg_bind(kind, ordinal, 4, scalar<int>(ctx, gqa));
    ctx.arg_bind(kind, ordinal, 5, pos); ctx.arg_bind(kind, ordinal, 6, pidx);
    ctx.arg_bind(kind, ordinal, 7, scalar<int>(ctx, kPage));
    ctx.arg_bind(kind, ordinal, 8, scalar<int>(ctx, kKv));
    ctx.arg_bind(kind, ordinal, 9, scalar<float>(ctx, scale));
    ctx.make_resident();

    // One threadgroup per GROUP of `heads` query heads. The grid shrinks as the
    // sharing widens -- which is the saving, and also the risk: fewer, fatter
    // threadgroups can under-occupy the device. That is why this is measured.
    Grid grid{uint32_t(kHeads / heads) * 1024u, 1, 1};
    Threadgroup tg{1024, 1, 1};
    LatencyHarness h(ctx);
    auto enc = [&](StepEncoder& se) {
        se.set_pso(pso); se.set_argtable(kind, ordinal); se.dispatch(grid, tg);
    };
    BenchResult r = h.time_step("hshare", enc, check ? 1 : 30, check ? 0 : 8);

    if (check) {
        const uint16_t* og = static_cast<const uint16_t*>(o.contents());
        int bad = 0; double worst = 0; int worst_head = -1;
        for (int hd = 0; hd < kHeads; ++hd) {
            const int kvh = hd / gqa;
            const float* kk = &kf[size_t(kvh) * (ctx_len + 1) * kD];
            const float* vv = &vf[size_t(kvh) * (ctx_len + 1) * kD];
            std::vector<double> sc(size_t(ctx_len) + 1); double mx = -1e30;
            for (int c = 0; c <= ctx_len; ++c) {
                double a2 = 0;
                for (int d = 0; d < kD; ++d)
                    a2 += double(qf[size_t(hd) * kD + d]) * double(kk[size_t(c) * kD + d]);
                sc[c] = a2 * double(scale); if (sc[c] > mx) mx = sc[c];
            }
            double sm = 0;
            for (int c = 0; c <= ctx_len; ++c) { sc[c] = std::exp(sc[c] - mx); sm += sc[c]; }
            for (int d = 0; d < kD; ++d) {
                double ref = 0;
                for (int c = 0; c <= ctx_len; ++c) ref += sc[c] * double(vv[size_t(c) * kD + d]);
                ref /= sm;
                uint32_t bits = uint32_t(og[size_t(hd) * kD + d]) << 16;
                float got; std::memcpy(&got, &bits, 4);
                const double e = std::abs(double(got) - ref) / (std::abs(ref) + 1e-6);
                if (e > 3e-2) { ++bad; if (e > worst) { worst = e; worst_head = hd; } }
            }
        }
        printf("  HEADS=%d correctness: %d of %d wrong%s", heads, bad, kHeads * kD,
               bad == 0 ? "  \xe2\x80\x94 CORRECT (all 32 heads)\n" : "");
        if (bad) printf("   (worst %.4f at head %d)\n", worst, worst_head);
    }
    return r.median.gpu_exec_ms;
}

// The fused NAX prefill attention: correctness FIRST, then rate.
//
// That order is not a style preference. `results-*.md` records a NAX kernel
// measured at 1.652 ms/layer and reported three times before a CPU reference
// showed it computed the wrong thing -- a timing cannot tell correct attention
// from a transposed operand, because both do the same FLOPs at the same rate.
//
// Data is distinct per query head AND per KV head, so a kernel that computed
// the right arithmetic against the wrong head cannot pass.
// `nwarps` is NOT optional and NOT defaulted. The kernel's query tile is
// `nwarps * 16` rows and its threadgroup is `nwarps * 32` threads, and the
// launch has to state both. An earlier version hardcoded 64 and 128 here and
// swept the kernel constant anyway -- which is the `pso_for` / `launch_shape`
// disagreement this repo documents, reproduced inside the instrument meant to
// measure it. It reported a BQ=128 kernel as 24576 elements WRONG when the
// kernel was fine and the grid was a different kernel's.
double nax_prefill_run(RawMetalContext& ctx, Pso pso, int rows, int ctx_len,
                       int ordinal, bool check, int nwarps) {
    constexpr int kHeads = 32, kKv = 4, kD = 128, kPage = 32;
    const int kBQ = nwarps * 16;
    const int gqa = kHeads / kKv;
    const int total = ctx_len + rows;
    const int pages = (total + kPage - 1) / kPage + 1;
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };
    auto unbf = [](uint16_t h) { uint32_t u = uint32_t(h) << 16; float f;
                                 std::memcpy(&f, &u, 4); return f; };

    SlotHandle q = ctx.heap_alloc(size_t(rows) * kHeads * kD * 2);
    SlotHandle kp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle vp = ctx.heap_alloc(size_t(pages) * kPage * kKv * kD * 2);
    SlotHandle o = ctx.heap_alloc(size_t(rows) * kHeads * kD * 2);
    SlotHandle pos = ctx.heap_alloc(size_t(rows) * sizeof(int));
    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    auto* qz = static_cast<uint16_t*>(q.contents());
    auto* kz = static_cast<uint16_t*>(kp.contents());
    auto* vz = static_cast<uint16_t*>(vp.contents());
    std::memset(kz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(vz, 0, size_t(pages) * kPage * kKv * kD * 2);
    std::memset(o.contents(), 0, size_t(rows) * kHeads * kD * 2);
    for (int p = 0; p < pages; ++p) static_cast<uint32_t*>(pidx.contents())[p] = uint32_t(p);
    for (int r = 0; r < rows; ++r) static_cast<int*>(pos.contents())[r] = ctx_len + r;

    // Kept for the reference. Only allocated when it will be used: at the rate
    // shape this would be 32 x 7608 x 128 floats and the rate run never reads it.
    std::vector<float> qf, kf, vf;
    if (check) {
        qf.assign(size_t(rows) * kHeads * kD, 0.f);
        kf.assign(size_t(kKv) * total * kD, 0.f);
        vf.assign(kf.size(), 0.f);
    }
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < kHeads; ++h)
        for (int d = 0; d < kD; ++d) {
          const float x = float((r * 2 + h * 3 + d) % 7) * 0.125f;
          qz[(size_t(r) * kHeads + h) * kD + d] = bf(x);
          if (check) qf[(size_t(r) * kHeads + h) * kD + d] = x;
        }
    for (int c = 0; c < total; ++c)
      for (int h = 0; h < kKv; ++h)
        for (int d = 0; d < kD; ++d) {
          const float kk = float((c + 2 * d + 5 * h) % 4) * 0.125f;
          const float vv = float((3 * c + d + 11 * h) % 6) * 0.25f;
          kz[(size_t(c) * kKv + h) * kD + d] = bf(kk);
          vz[(size_t(c) * kKv + h) * kD + d] = bf(vv);
          if (check) {
              kf[(size_t(h) * total + c) * kD + d] = unbf(bf(kk));
              vf[(size_t(h) * total + c) * kD + d] = unbf(bf(vv));
          }
        }

    const float scale = 1.0f / 11.3137085f;
    const Kernel kind = Kernel::Sdpa;
    ctx.arg_bind(kind, ordinal, 0, q);   ctx.arg_bind(kind, ordinal, 1, kp);
    ctx.arg_bind(kind, ordinal, 2, vp);  ctx.arg_bind(kind, ordinal, 3, o);
    ctx.arg_bind(kind, ordinal, 4, scalar<int>(ctx, gqa));
    ctx.arg_bind(kind, ordinal, 5, pos); ctx.arg_bind(kind, ordinal, 6, pidx);
    ctx.arg_bind(kind, ordinal, 7, scalar<int>(ctx, kKv));
    ctx.arg_bind(kind, ordinal, 8, scalar<float>(ctx, scale));
    ctx.arg_bind(kind, ordinal, 9, scalar<int>(ctx, rows));
    ctx.make_resident();

    const uint32_t threads = uint32_t(nwarps) * 32u;
    Grid grid{uint32_t(kHeads) * threads, uint32_t((rows + kBQ - 1) / kBQ), 1};
    Threadgroup tg{threads, 1, 1};
    LatencyHarness h(ctx);
    auto enc = [&](StepEncoder& se) {
        se.set_pso(pso); se.set_argtable(kind, ordinal); se.dispatch(grid, tg);
    };
    BenchResult r = h.time_step("nax-prefill", enc, check ? 1 : 30, check ? 0 : 8);

    if (check) {
        const uint16_t* og = static_cast<const uint16_t*>(o.contents());
        int bad = 0; double worst = 0; int wr = -1, wh = -1, wd = -1;
        for (int rr = 0; rr < rows; ++rr) {
            const int hi = ctx_len + rr;            // causal bound for this row
            for (int hd = 0; hd < kHeads; ++hd) {
                const int kvh = hd / gqa;
                const float* kk = &kf[size_t(kvh) * total * kD];
                const float* vv = &vf[size_t(kvh) * total * kD];
                std::vector<double> sc(size_t(hi) + 1);
                double mx = -1e30;
                for (int c = 0; c <= hi; ++c) {
                    double a2 = 0;
                    for (int d = 0; d < kD; ++d)
                        a2 += double(qf[(size_t(rr) * kHeads + hd) * kD + d]) *
                              double(kk[size_t(c) * kD + d]);
                    sc[c] = a2 * double(scale);
                    if (sc[c] > mx) mx = sc[c];
                }
                double sm = 0;
                for (int c = 0; c <= hi; ++c) { sc[c] = std::exp(sc[c] - mx); sm += sc[c]; }
                for (int d = 0; d < kD; ++d) {
                    double ref = 0;
                    for (int c = 0; c <= hi; ++c) ref += sc[c] * double(vv[size_t(c) * kD + d]);
                    ref /= sm;
                    uint32_t bits = uint32_t(og[(size_t(rr) * kHeads + hd) * kD + d]) << 16;
                    float got; std::memcpy(&got, &bits, 4);
                    const double e = std::abs(double(got) - ref) / (std::abs(ref) + 1e-6);
                    if (e > 3e-2) {
                        ++bad;
                        if (e > worst) { worst = e; wr = rr; wh = hd; wd = d; }
                    }
                }
            }
        }
        printf("  correctness %d rows @ %d ctx: %d of %d wrong%s", rows, ctx_len, bad,
               rows * kHeads * kD, bad == 0 ? "  \xe2\x80\x94 CORRECT\n" : "\n");
        if (bad) printf("    worst %.4f at row %d head %d dim %d\n", worst, wr, wh, wd);
    }
    return r.median.gpu_exec_ms;
}

// The accuracy question this kernel actually has to answer.
//
// The NAX kernel changes serving output: the same 28.4k prompt that produced
// 382 identical tokens with head sharing on and off produced 332 with NAX on.
// A different attention kernel rounds differently and greedy decoding diverges
// at ties, so that is not by itself a defect -- `apc-graft-probe` measured the
// same shape of divergence from re-CHUNKING a prefill, which changes no
// arithmetic at all.
//
// But "not by itself a defect" is a claim, and the way to test it is not to
// compare NAX against exact arithmetic. It is to compare NAX against THE KERNEL
// IT REPLACES, both against the same double-precision reference, on the same
// inputs. If the two are the same order of error, the divergence is the
// ordinary consequence of swapping kernels. If NAX is much worse, it is not.
//
// `relaxed_precision` in the matmul descriptor is the specific thing under
// suspicion, which is why it is a parameter of the sweep below.
struct ErrStats { double mean, p99, worst; long long n; };

ErrStats compare_to_reference(const uint16_t* got, const std::vector<float>& qf,
                              const std::vector<float>& kf, const std::vector<float>& vf,
                              int rows, int ctx_len, int heads, int kv, int d,
                              float scale) {
    std::vector<double> errs;
    errs.reserve(size_t(rows) * heads * d);
    const int gqa = heads / kv;
    const int total = ctx_len + rows;
    for (int r = 0; r < rows; ++r) {
        const int hi = ctx_len + r;
        for (int h = 0; h < heads; ++h) {
            const float* kk = &kf[size_t(h / gqa) * total * d];
            const float* vv = &vf[size_t(h / gqa) * total * d];
            std::vector<double> sc(size_t(hi) + 1);
            double mx = -1e30;
            for (int c = 0; c <= hi; ++c) {
                double a = 0;
                for (int e = 0; e < d; ++e)
                    a += double(qf[(size_t(r) * heads + h) * d + e]) *
                         double(kk[size_t(c) * d + e]);
                sc[c] = a * double(scale);
                if (sc[c] > mx) mx = sc[c];
            }
            double sm = 0;
            for (int c = 0; c <= hi; ++c) { sc[c] = std::exp(sc[c] - mx); sm += sc[c]; }
            for (int e = 0; e < d; ++e) {
                double ref = 0;
                for (int c = 0; c <= hi; ++c) ref += sc[c] * double(vv[size_t(c) * d + e]);
                ref /= sm;
                uint32_t bits = uint32_t(got[(size_t(r) * heads + h) * d + e]) << 16;
                float f; std::memcpy(&f, &bits, 4);
                errs.push_back(std::abs(double(f) - ref) / (std::abs(ref) + 1e-6));
            }
        }
    }
    std::sort(errs.begin(), errs.end());
    double sum = 0;
    for (double e : errs) sum += e;
    return {sum / double(errs.size()), errs[size_t(double(errs.size()) * 0.99)],
            errs.back(), (long long)errs.size()};
}

// Bind ONE set of buffers in the shipped `sdpa_paged_mma` ABI and run both
// kernels over it, so nothing differs between the arms except the pipeline.
// The shipped NAX kernel carries that ABI byte for byte, deliberately, which is
// what makes this comparison a one-line difference rather than a second rig.
void accuracy_ab(RawMetalContext& ctx, Pso mma, Pso nax, int rows, int ctx_len,
                 int ordinal) {
    constexpr int kH = 32, kKv = 4, kD = 128, kPg = 32;
    const int gqa = kH / kKv, total = ctx_len + rows;
    const int pages = (total + kPg - 1) / kPg + 1;
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); };
    auto unbf = [](uint16_t h) { uint32_t u = uint32_t(h) << 16; float f;
                                 std::memcpy(&f, &u, 4); return f; };

    SlotHandle q = ctx.heap_alloc(size_t(rows) * kH * kD * 2);
    SlotHandle kp = ctx.heap_alloc(size_t(pages) * kPg * kKv * kD * 2);
    SlotHandle vp = ctx.heap_alloc(size_t(pages) * kPg * kKv * kD * 2);
    SlotHandle out = ctx.heap_alloc(size_t(rows) * kH * kD * 2);
    SlotHandle pos = ctx.heap_alloc(size_t(rows) * sizeof(int));
    SlotHandle req = ctx.heap_alloc(size_t(rows) * sizeof(int));
    SlotHandle pidx = ctx.heap_alloc(size_t(pages) * sizeof(uint32_t));
    SlotHandle pindptr = ctx.heap_alloc(2 * sizeof(uint32_t));
    SlotHandle mask = ctx.heap_alloc(16), mask_on = ctx.heap_alloc(size_t(rows));
    SlotHandle sinks = ctx.heap_alloc(64);
    auto* qz = static_cast<uint16_t*>(q.contents());
    auto* kz = static_cast<uint16_t*>(kp.contents());
    auto* vz = static_cast<uint16_t*>(vp.contents());
    std::memset(kz, 0, size_t(pages) * kPg * kKv * kD * 2);
    std::memset(vz, 0, size_t(pages) * kPg * kKv * kD * 2);
    std::memset(mask.contents(), 0, 16);
    std::memset(mask_on.contents(), 0, size_t(rows));
    std::memset(sinks.contents(), 0, 64);
    for (int i = 0; i < rows; ++i) {
        static_cast<int*>(pos.contents())[i] = ctx_len + i;
        static_cast<int*>(req.contents())[i] = 0;
    }
    for (int p2 = 0; p2 < pages; ++p2) static_cast<uint32_t*>(pidx.contents())[p2] = uint32_t(p2);
    static_cast<uint32_t*>(pindptr.contents())[0] = 0;
    static_cast<uint32_t*>(pindptr.contents())[1] = uint32_t(pages);

    std::vector<float> qf(size_t(rows) * kH * kD), kf(size_t(kKv) * total * kD), vf(kf.size());
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < kH; ++h)
        for (int d = 0; d < kD; ++d) {
          const float x = float((r * 2 + h * 3 + d) % 7) * 0.125f;
          qz[(size_t(r) * kH + h) * kD + d] = bf(x);
          qf[(size_t(r) * kH + h) * kD + d] = x;
        }
    for (int c = 0; c < total; ++c)
      for (int h = 0; h < kKv; ++h)
        for (int d = 0; d < kD; ++d) {
          const float kk = float((c + 2 * d + 5 * h) % 4) * 0.125f;
          const float vv = float((3 * c + d + 11 * h) % 6) * 0.25f;
          kz[(size_t(c) * kKv + h) * kD + d] = bf(kk);
          vz[(size_t(c) * kKv + h) * kD + d] = bf(vv);
          kf[(size_t(h) * total + c) * kD + d] = unbf(bf(kk));
          vf[(size_t(h) * total + c) * kD + d] = unbf(bf(vv));
        }

    const float scale = 1.0f / 11.3137085f;
    const Kernel kind = Kernel::SdpaPaged;
    ctx.arg_bind(kind, ordinal, 0, q);    ctx.arg_bind(kind, ordinal, 1, kp);
    ctx.arg_bind(kind, ordinal, 2, vp);   ctx.arg_bind(kind, ordinal, 3, out);
    ctx.arg_bind(kind, ordinal, 4, scalar<int>(ctx, gqa));
    ctx.arg_bind(kind, ordinal, 5, pos);  ctx.arg_bind(kind, ordinal, 6, req);
    ctx.arg_bind(kind, ordinal, 7, pidx); ctx.arg_bind(kind, ordinal, 8, pindptr);
    ctx.arg_bind(kind, ordinal, 9, scalar<int>(ctx, kPg));
    ctx.arg_bind(kind, ordinal, 10, scalar<int>(ctx, kKv));
    ctx.arg_bind(kind, ordinal, 11, scalar<float>(ctx, scale));
    ctx.arg_bind(kind, ordinal, 12, mask);
    ctx.arg_bind(kind, ordinal, 13, scalar<uint32_t>(ctx, 0u));
    ctx.arg_bind(kind, ordinal, 14, mask_on);
    ctx.arg_bind(kind, ordinal, 15, scalar<int>(ctx, 0));
    ctx.arg_bind(kind, ordinal, 16, sinks);
    ctx.arg_bind(kind, ordinal, 17, scalar<int>(ctx, rows));
    ctx.make_resident();

    LatencyHarness h(ctx);
    struct Arm { const char* name; Pso pso; int tile; };
    const Arm arms[] = {{"sdpa_paged_mma (shipped)", mma, 32},
                        {"sdpa_paged_nax (new)", nax, 64}};
    for (const Arm& a : arms) {
        if (!a.pso.valid()) { printf("    %-26s  NOT COMPILED\n", a.name); continue; }
        std::memset(out.contents(), 0, size_t(rows) * kH * kD * 2);
        const uint32_t tiles = uint32_t((rows + a.tile - 1) / a.tile);
        auto enc = [&](StepEncoder& se) {
            se.set_pso(a.pso); se.set_argtable(kind, ordinal);
            se.dispatch(Grid{uint32_t(kH) * 128u, tiles, 1}, Threadgroup{128, 1, 1});
        };
        h.time_step("acc", enc, 1, 0);
        const ErrStats e = compare_to_reference(
            static_cast<const uint16_t*>(out.contents()), qf, kf, vf, rows, ctx_len,
            kH, kKv, kD, scale);
        printf("    %-26s  mean %.2e   p99 %.2e   worst %.2e\n",
               a.name, e.mean, e.p99, e.worst);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kernels_dir = PIE_METAL_TOOL_KERNELS_DIR;
    if (argc > 1) kernels_dir = argv[1];

    printf("pie paged attention, priced in isolation (bfloat16, d=128, page=32)\n");
    printf("MLX fast.scaled_dot_product_attention on the same shapes: 1.555 ms/layer\n\n");

    auto ctx = RawMetalContext::create(/*heap_bytes=*/1024ull << 20);
    if (!ctx) {
        printf("FAIL: no Metal context\n");
        return 1;
    }
    std::string err;
    Pso mma = ctx->compile_pso_from_file(kernels_dir + "/sdpa_paged_mma.metal",
                                         "sdpa_paged_mma_bfloat16_d_128", &err);
    if (!mma.valid()) {
        printf("FAIL sdpa_paged_mma compile: %s\n", err.c_str());
        return 1;
    }
    Pso tiled = ctx->compile_pso_from_file(kernels_dir + "/sdpa_paged.metal",
                                           "sdpa_paged_tiled_bfloat16_d_128", &err);

    // 1-row is deliberately absent. `llama_sdpa_mma_this_fire` requires
    // `sdpa_should_tile` (>= 32 rows per request), so a decode step takes
    // `sdpa_paged_decode`, NOT this kernel. Timing MMA at one row measures a
    // matrix kernel doing a vector's work -- a configuration the driver never
    // dispatches, and it priced at 3.347 ms/layer, which would put attention
    // alone at 161 ms inside a decode fire that measures 21 ms end to end.
    const Shape shapes[] = {
        {184, 7424, "184 rows @ 7424 ctx"},   // the serving prefill fire
        {184, 2048, "184 rows @ 2048 ctx"},   // shorter cache, same width
        {512, 7424, "512 rows @ 7424 ctx"},   // wider fire, same cache
    };

    // ── k-row decode: does it read the cache once, or k times? ──
    {
        printf("k-row decode, shared KV read (KROWS=1 is the per-row baseline):\n");
        const std::string ld = PIE_METAL_TOOL_LOCAL_KERNELS_DIR;
        double base = 0;
        for (int k : {1, 2, 3, 4, 5, 6, 8}) {
            std::string ek;
            char f[64]; snprintf(f, sizeof f, "/sdpa_krow_%d.metal", k);
            Pso p = ctx->compile_pso_from_file(ld + f, "sdpa_krow_decode", &ek);
            if (!p.valid()) { printf("  KROWS=%d compile fail: %s\n", k, ek.c_str()); continue; }
            if (k <= 4 || k == 8) krow_run(*ctx, p, k, 2048, 200 + k, /*check=*/true);
            const double t = krow_run(*ctx, p, k, 16384, 220 + k, /*check=*/false);
            if (k == 1) base = t;
            printf("    16k ctx: %7.3f ms   %.2fx the 1-row fire   (per-row kernel: %.2fx)\n",
                   t, base > 0 ? t / base : 1.0,
                   k == 1 ? 1.00 : (k == 2 ? 1.54 : (k == 4 ? 2.39 : (k == 5 ? 2.82 : (k == 8 ? 4.08 : 0.0)))));
        }
        printf("  (measured for the shipped per-row kernel: 1.54x / 2.39x / 4.08x)\n\n");
    }

    // ── head-sharing decode: the axis a single-stream agent decode actually
    // has. k-row sharing needs k>1 and an agent turn decodes one token at a
    // time; the GQA group is 8 wide regardless. HEADS=1 is the shipped shape,
    // so every ratio below is head sharing and nothing else.
    {
        printf("head-sharing decode, one KV read per GQA group (HEADS=1 is the shipped shape):\n");
        const std::string ld = PIE_METAL_TOOL_LOCAL_KERNELS_DIR;
        double base16 = 0, base2 = 0, base8 = 0, base12 = 0;
        for (int hh : {1, 2, 4, 8}) {
            std::string ek;
            char f[64]; snprintf(f, sizeof f, "/sdpa_hshare_%d.metal", hh);
            Pso p = ctx->compile_pso_from_file(ld + f, "sdpa_hshare_decode", &ek);
            if (!p.valid()) { printf("  HEADS=%d compile fail: %s\n", hh, ek.c_str()); continue; }
            hshare_run(*ctx, p, hh, 2047, 300 + hh, /*check=*/true);
            // 8k and 12k are where this workload's turns actually sit
            // (results-e2e-one-instance.md: prompts of 7.4k rising to 12.0k).
            // 16k is kept because the k-row arm above reports there and the two
            // must be readable against each other.
            const double t2 = hshare_run(*ctx, p, hh, 2047, 320 + hh, /*check=*/false);
            const double t8 = hshare_run(*ctx, p, hh, 8191, 360 + hh, /*check=*/false);
            const double t12 = hshare_run(*ctx, p, hh, 12287, 380 + hh, /*check=*/false);
            const double t16 = hshare_run(*ctx, p, hh, 16383, 340 + hh, /*check=*/false);
            if (hh == 1) { base16 = t16; base2 = t2; base8 = t8; base12 = t12; }
            printf("     2k %6.3f %.2fx | 8k %6.3f %.2fx | 12k %6.3f %.2fx | "
                   "16k %6.3f %.2fx   (12k x48 = %5.1f ms)\n",
                   t2, base2 > 0 ? t2 / base2 : 1.0,
                   t8, base8 > 0 ? t8 / base8 : 1.0,
                   t12, base12 > 0 ? t12 / base12 : 1.0,
                   t16, base16 > 0 ? t16 / base16 : 1.0, t12 * 48.0);
        }
        printf("  A decode step's whole attention is the 16k column x48. The\n"
               "  dispatch trace puts attention at 75-77%% of a decode fire at 23k.\n\n");
    }

    // ── the fused NAX prefill: the only path to MLX's number ──
    {
        printf("FUSED NAX PREFILL (paged, O in registers, no staging):\n");
        std::string ek;
        Pso p = ctx->compile_pso_from_file(
            std::string(PIE_METAL_TOOL_LOCAL_KERNELS_DIR) + "/sdpa_nax_prefill.metal",
            "sdpa_nax_prefill", &ek);
        if (!p.valid()) {
            printf("  compile failed: %s\n\n", ek.c_str());
        } else {
            // Small first, and cheap enough for an O(rows x heads x ctx x dim)
            // CPU reference. A kernel wrong at 64 rows is wrong at 184.
            // Deliberately unfriendly shapes, because the friendly ones hide
            // exactly the bugs this kernel can have: a row count that is not a
            // multiple of the 64-row threadgroup tile leaves a partial
            // simdgroup, and a context that is not a multiple of the 32-key
            // page leaves a partial final block. Both are masked rather than
            // branched -- an MMA cannot run under a divergent condition -- so a
            // mistake there is wrong numbers in a corner, not a crash.
            nax_prefill_run(*ctx, p, 64, 96, 500, /*check=*/true, 4);    // both aligned
            nax_prefill_run(*ctx, p, 64, 100, 501, /*check=*/true, 4);   // ctx % 32 = 4
            nax_prefill_run(*ctx, p, 40, 96, 504, /*check=*/true, 4);    // rows < one tile
            nax_prefill_run(*ctx, p, 70, 133, 505, /*check=*/true, 4);   // both ragged
            nax_prefill_run(*ctx, p, 17, 1, 506, /*check=*/true, 4);     // tiny, ctx 1
            nax_prefill_run(*ctx, p, 184, 224, 502, /*check=*/true, 4);
            // The query-tile width is the free parameter, and it is swept
            // rather than assumed: BK is pinned to the page size and D to the
            // head, so this is the only one left. `nax_prefill_run` builds the
            // grid from the kernel's own BQ, so each arm launches its own shape.
            for (const int w : {2, 4, 8}) {
                char f[64]; snprintf(f, sizeof f, "/sdpa_nax_prefill_w%d.metal", w);
                std::string ew;
                Pso pw = ctx->compile_pso_from_file(
                    std::string(PIE_METAL_TOOL_LOCAL_KERNELS_DIR) + f,
                    "sdpa_nax_prefill", &ew);
                if (!pw.valid()) { printf("  BQ=%d compile fail: %s\n", w * 16, ew.c_str()); continue; }
                // Correctness per arm too. A tile width is a different mask and
                // a different tail, so "the 64-row one was right" is not a
                // statement about this one.
                nax_prefill_run(*ctx, pw, 70, 133, 520 + w, /*check=*/true, w);
                const double t = nax_prefill_run(*ctx, pw, 184, 7424, 540 + w, false, w);
                printf("  BQ=%-3d  184 rows @ 7424 ctx  %7.3f ms/layer  x48 = %6.1f ms"
                       "  (%5.2f TFLOP/s)  %.2fx shipped\n",
                       w * 16, t, t * 48.0,
                       184.0 * 32 * 7424 * 128 * 4 / (t / 1000.0) / 1e12,
                       6.975 / t);
            }
            printf("  shipped sdpa_paged_mma: 6.97.  MLX: 1.555.  "
                   "simdgroup floor: 4.1.\n");
            // Accuracy against the kernel it REPLACES, not against exact.
            {
                std::string es;
                Pso shipped_nax = ctx->compile_pso_from_file(
                    std::string(PIE_METAL_TOOL_KERNELS_DIR) + "/sdpa_paged_nax.metal",
                    "sdpa_paged_nax", &es);
                if (!shipped_nax.valid())
                    printf("  shipped NAX compile failed: %s\n", es.c_str());
                printf("  accuracy, both against a float64 reference:\n");
                accuracy_ab(*ctx, mma, shipped_nax, 184, 224, 600);
                accuracy_ab(*ctx, mma, shipped_nax, 184, 1024, 601);
            }
            printf("\n");
        }
    }

    printf("MMA path (what a >=32-row prefill dispatches):\n");
    int layer = 0;
    double paged = 0;
    for (const Shape& s : shapes) {
        const double t = run_one(*ctx, mma, s, layer++, Path::Mma);
        if (s.ctx == 7424 && s.rows == 184) paged = t;
    }

    if (tiled.valid()) {
        printf("\nTiled path (the fallback the MMA kernel replaced):\n");
        run_one(*ctx, tiled, shapes[0], layer++, Path::Tiled);
    } else {
        printf("\n(tiled variant not compiled: %s)\n", err.c_str());
    }

    // ── Does de-paging pay? ──
    const std::string local_dir = PIE_METAL_TOOL_LOCAL_KERNELS_DIR;
    Pso contig = ctx->compile_pso_from_file(local_dir + "/sdpa_contig_mma.metal",
                                            "sdpa_paged_mma_bfloat16_d_128", &err);
    Pso depage = ctx->compile_pso_from_file(local_dir + "/kv_depage.metal",
                                            "kv_depage_16b", &err);
    if (!contig.valid() || !depage.valid()) {
        printf("\n(de-page A/B skipped: %s)\n", err.c_str());
        return 0;
    }

    printf("\nDe-paging A/B — same kernel, same bytes, page walk removed:\n");
    const double contiguous = run_one(*ctx, contig, shapes[0], layer++, Path::Contig);
    const double gather = depage_cost(*ctx, depage, shapes[0].ctx, layer++);

    // Decomposes the tax: how much of it is the integer DIVISION, which the
    // shipped kernel now removes for free via its `_p32` specialization,
    // versus the dependent load, which would cost a scratch-buffer redesign.
    // This compiles the SHIPPED file, not a twin -- it measures what the
    // driver actually dispatches when kv_page_size == 32.
    Pso pow2 = ctx->compile_pso_from_file(kernels_dir + "/sdpa_paged_mma.metal",
                                          "sdpa_paged_mma_bfloat16_d_128_p32", &err);
    double shifted = 0;
    if (pow2.valid()) {
        printf("\nShipped `_p32` path — page walk KEPT, divide -> shift:\n");
        shifted = run_one(*ctx, pow2, shapes[0], layer++, Path::Contig);
    } else {
        printf("\n(_p32 variant not compiled: %s)\n", err.c_str());
    }

    // Is the OTHER matrix path even reachable from here? See nax_probe.metal.
    {
        std::string nerr;
        Pso nax = ctx->compile_pso_from_file(local_dir + "/nax_probe.metal",
                                             "nax_probe", &nerr);
        printf("\nM5 neural-accelerator path (mpp::tensor_ops::matmul2d): %s\n",
               nax.valid() ? "AVAILABLE through pie's runtime compiler"
                           : "NOT available");
        if (!nax.valid()) printf("  %s\n", nerr.c_str());
    }

    // MLX's arithmetic on pie's tile: fp32 accumulators, no DCH chunking,
    // fragments as float pairs. The one experiment aimed at the multiply half.
    Pso f32acc = ctx->compile_pso_from_file(local_dir + "/sdpa_f32acc_mma.metal",
                                            "sdpa_paged_mma_bfloat16_d_128_p32", &err);
    if (f32acc.valid()) {
        printf("\nfp32 accumulators, unchunked, MLX fragment form:\n");
        const double f32 = run_one(*ctx, f32acc, shapes[0], layer++, Path::Contig);
        const double ref = shifted > 0 ? shifted : paged;
        printf("  -> %+.1f%% against the shipped `_p32` (%.3f ms), %.2fx MLX\n",
               100.0 * (f32 - ref) / ref, ref, f32 / 1.555);
    } else {
        printf("\n(f32acc twin not compiled: %s)\n", err.c_str());
    }

    // ── Where does the time actually go? ──
    //
    // Two twins that between them split the kernel in half: one keeps the
    // staging and drops the multiply, the other drops the staging and keeps
    // the multiply. See `sdpa_nomath_mma.metal` for why the halves are a
    // direction rather than an attribution, and why their SUM is the check
    // that keeps them honest.
    Pso nomath = ctx->compile_pso_from_file(local_dir + "/sdpa_nomath_mma.metal",
                                            "sdpa_paged_mma_bfloat16_d_128_p32", &err);
    Pso nostage = ctx->compile_pso_from_file(local_dir + "/sdpa_nostage_mma.metal",
                                             "sdpa_paged_mma_bfloat16_d_128_p32", &err);
    if (nomath.valid() && nostage.valid()) {
        // THE REFERENCE IS MEASURED TWICE, before and after the ablation arms.
    // Arms run sequentially inside one invocation, so a later arm is measured
    // on a hotter machine than an earlier one -- and this device's clock is
    // demonstrably load-dependent (a duration sweep runs 3.96 -> 5.18 TFLOP/s
    // as the kernel lengthens). Comparing an arm that runs last against a
    // baseline that ran first is therefore not a clean A/B. If these two
    // bracketing numbers disagree, the split between them cannot be trusted.
    const double ref_before = run_one(*ctx, pow2, shapes[0], layer++, Path::Contig);
    printf("\nAblation — splitting the kernel into move vs multiply:\n");
        const double move = run_one(*ctx, nomath, shapes[0], layer++, Path::Contig);
        const double mult = run_one(*ctx, nostage, shapes[0], layer++, Path::Contig);
        const double whole = shifted > 0 ? shifted : paged;
        printf("\n  move keys into threadgroup memory : %7.3f ms/layer  (%.0f%%)\n",
               move, 100.0 * move / whole);
        printf("  multiply them                    : %7.3f ms/layer  (%.0f%%)\n",
               mult, 100.0 * mult / whole);
        const double ref_after = run_one(*ctx, pow2, shapes[0], layer++, Path::Contig);
        const double drift = 100.0 * (ref_after - ref_before) / ref_before;
        printf("\n  reference before/after the arms: %.3f / %.3f ms  (drift %+.1f%%)\n",
               ref_before, ref_after, drift);
        if (drift > 5.0 || drift < -5.0) {
            printf("  !! DRIFT EXCEEDS 5%% — the split below is NOT trustworthy.\n");
        }
        printf("  halves sum to %.3f against %.3f unablated — %s\n",
               move + mult, whole,
               (move + mult) > 0.85 * whole && (move + mult) < 1.15 * whole
                   ? "consistent, read the split"
                   : "NOT consistent; the interaction is the finding, not the split");
    } else {
        printf("\n(ablation twins not compiled: %s)\n", err.c_str());
    }

    const double tax = paged - contiguous;
    const double net = tax - gather;
    printf("\n  paging tax   = %7.3f - %7.3f = %+7.3f ms/layer\n", paged, contiguous, tax);
    printf("  gather cost  =                   %+7.3f ms/layer\n", -gather);
    printf("  NET of de-paging               = %+7.3f ms/layer  (x48 = %+.1f ms)\n",
           net, net * kLayers);
    // The verdict is stated against the gap it would have to close, not against
    // zero: a win that is real but 2% of a 4.8x deficit is not a plan.
    printf("\n  contiguous kernel would still be %.2fx MLX's 1.555 ms.\n",
           contiguous / 1.555);
    printf("  de-paging closes %.1f%% of the %.2fx gap.\n",
           paged > 1.555 ? 100.0 * net / (paged - 1.555) : 0.0, paged / 1.555);
    if (shifted > 0) {
        printf("\n  of the %+.3f ms tax, the DIVIDE is %+.3f ms (%.0f%%) and costs\n",
               tax, paged - shifted, tax > 0 ? 100.0 * (paged - shifted) / tax : 0.0);
        printf("  nothing to remove; the dependent LOAD is the remaining %+.3f ms\n",
               shifted - contiguous);
        printf("  and is the only part a scratch buffer buys.\n");
    }
    return 0;
}
