// device_identity_probe.mm — what the driver thinks this machine is.
//
// T0.2 of the M5 bring-up plan. Nothing in the tree printed
// `pie::metal::query_device_info()`, so "which DeviceTuning block does this
// device select" was an inference from reading `tuning_for()`. This prints the
// driver's own answers — family, core count, and the tuned fields that
// discriminate between the blocks — next to a default-constructed
// `DeviceTuning`, so the selected block is read off rather than deduced.
//
// It also asks the `MTLDevice` directly about families the DRIVER does not
// probe: `query_apple_family()` stops at Apple9, while the macOS 26 SDK
// defines `MTLGPUFamilyApple10`. The gap between the two answers is exactly
// the fact Phase 1 needs — an M5 that answers yes to Apple10 but is reported
// as 9 cannot be given its own `case` until the probe list is extended.
//
// Env overrides (`PIE_METAL_*`) fold into `device_tuning()` before it is
// printed. Run this from a clean environment: an override that is set makes
// the block identification read as "overridden", not as a measurement.

#import <Metal/Metal.h>

#include <cstdio>

#include "device_tuning.hpp"

int main() {
    @autoreleasepool {
        const auto& info = pie::metal::device_info();
        const auto& t = pie::metal::device_tuning();
        const pie::metal::DeviceTuning defaults;  // the M1 Max measurements

        std::printf("driver's device query (device_tuning_apple.mm):\n");
        std::printf("  apple_family   : %d\n", info.apple_family);
        std::printf("  gpu_core_count : %d\n", info.gpu_core_count);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil) {
            std::printf("\nMTLDevice, asked directly (the driver probes 9..5):\n");
            std::printf("  name           : %s\n", dev.name.UTF8String);
            // Raw enum values: MTLGPUFamilyApple1 == 1001. Values past the
            // SDK's last named constant answer NO on a correct stack, so
            // probing a few past Apple9 is informational, not UB.
            for (int n = 9; n <= 12; ++n) {
                const bool yes = [dev supportsFamily:(MTLGPUFamily)(1000 + n)];
                std::printf("  Apple%-2d        : %s\n", n, yes ? "yes" : "no");
            }
        } else {
            std::printf("\nFAIL: no Metal device\n");
            return 1;
        }

        // The fields the blocks in tuning_for() actually move, against the
        // default-constructed struct. Anything else equal-to-default carries
        // no information about which case fired.
        std::printf("\ndevice_tuning() vs default-constructed (M1 Max):\n");
        std::printf("  %-28s %8s %8s\n", "field", "selected", "default");
        std::printf("  %-28s %8d %8d\n", "qmm_bn_crossover_tg",
                    t.qmm_bn_crossover_tg, defaults.qmm_bn_crossover_tg);
        std::printf("  %-28s %8d %8d\n", "qmm_min_batch_moe",
                    t.qmm_min_batch_moe, defaults.qmm_min_batch_moe);
        std::printf("  %-28s %8d %8d\n", "qmm_min_batch",
                    t.qmm_min_batch, defaults.qmm_min_batch);

        const char* block = "DEFAULT (M1 Max constants)";
        if (t.qmm_bn_crossover_tg == 96 &&
            t.qmm_bn_crossover_tg != defaults.qmm_bn_crossover_tg) {
            block = "case 9 (M3/M4: qmm_bn_crossover_tg = 96)";
        } else if (t.qmm_min_batch_moe == 12 &&
                   t.qmm_min_batch_moe != defaults.qmm_min_batch_moe) {
            block = "case 8 (M2: qmm_min_batch_moe = 12)";
        }
        std::printf("\nselected DeviceTuning block: %s\n", block);
        std::printf("(valid only with no PIE_METAL_* override in the environment)\n");
        return 0;
    }
}
