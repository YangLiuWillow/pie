// device_caps.mm — the limits this driver assumes but never asks about.
//
// `sdpa_paged_mma.metal` reasons throughout about "the 32 KB a threadgroup
// gets", and that number appears nowhere except in its comments: nothing in
// this driver queries `maxThreadgroupMemoryLength`. The whole KT ladder
// (d=64 -> 8 KB, d=128 -> 16 KB, d=256 -> 32 KB "exactly the cap") rests on
// it, and so does the tile choice for a neural-accelerator rewrite -- at
// BQ=64, BK=64, D=128 the staged tiles estimate to ~35.8 KB, which does not
// fit 32 KB, yet MLX ships exactly that configuration for this width. Either
// the cap is larger here or the layout is tighter than the estimate, and
// guessing 32 would silently give up half the available pass reduction.
//
// Standalone and framework-only, like `dvfs_probe.mm`: it must be answerable
// without building the driver, and it must not become a driver dependency.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

namespace {

struct FamilyProbe {
    MTLGPUFamily family;
    const char* name;
};

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            std::printf("FAIL: no Metal device\n");
            return 1;
        }

        std::printf("device: %s\n", dev.name.UTF8String);
        std::printf("  unified memory        : %s\n", dev.hasUnifiedMemory ? "yes" : "no");
        std::printf("  recommended workingset: %.1f GB\n",
                    double(dev.recommendedMaxWorkingSetSize) / 1073741824.0);

        // THE NUMBER THIS TOOL EXISTS FOR.
        const NSUInteger tgm = dev.maxThreadgroupMemoryLength;
        std::printf("\n  maxThreadgroupMemoryLength : %lu bytes  (%.1f KB)\n",
                    (unsigned long)tgm, double(tgm) / 1024.0);
        std::printf("  maxThreadsPerThreadgroup   : %lu x %lu x %lu\n",
                    (unsigned long)dev.maxThreadsPerThreadgroup.width,
                    (unsigned long)dev.maxThreadsPerThreadgroup.height,
                    (unsigned long)dev.maxThreadsPerThreadgroup.depth);

        static const FamilyProbe kFamilies[] = {
            {MTLGPUFamilyApple7, "Apple7"}, {MTLGPUFamilyApple8, "Apple8"},
            {MTLGPUFamilyApple9, "Apple9"},
            {MTLGPUFamilyMetal3, "Metal3"},
        };
        std::printf("  families                   :");
        for (const FamilyProbe& f : kFamilies) {
            if ([dev supportsFamily:f.family]) std::printf(" %s", f.name);
        }
        std::printf("\n");

        // What the candidate attention tiles actually cost, so the answer above
        // is applied rather than merely reported. bf16, head_dim 128, MLX's
        // 16-byte pad. K and V ALIAS one buffer (V is not needed until after S
        // is computed and masked), which is what makes the wide BK reachable.
        std::printf("\nCandidate tiles at D=128, bf16, K/V aliased (pad = 8 halves):\n");
        std::printf("  %-18s %10s %10s %8s\n", "shape", "Q tile", "KV tile", "total");
        const int D = 128, pad = 8;
        const struct { int bq, bk; const char* note; } tiles[] = {
            {32, 16, "pie today (8x8 frag)"},
            {64, 16, "NAX minimum BQ"},
            {64, 32, "MLX bq64_bk32"},
            {64, 64, "MLX bq64_bk64"},
        };
        for (const auto& t : tiles) {
            const int q = t.bq * (D + pad) * 2;
            const int kv = (((t.bk + pad) * D) > (t.bk * (D + pad))
                                ? (t.bk + pad) * D
                                : t.bk * (D + pad)) * 2;
            const int total = q + kv;
            std::printf("  BQ=%-3d BK=%-3d %6.1f KB %9.1f KB %7.1f KB  %s%s\n",
                        t.bq, t.bk, q / 1024.0, kv / 1024.0, total / 1024.0,
                        (NSUInteger)total <= tgm ? "FITS  " : "OVER  ", t.note);
        }
        std::printf("\n  (pie today does NOT alias K and V, and stages Q, K and V\n"
                    "   separately -- 16 KB at BQ=32/BK=16/D=128.)\n");
        return 0;
    }
}
