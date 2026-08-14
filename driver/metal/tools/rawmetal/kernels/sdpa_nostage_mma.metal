// The paged MMA kernel with the STAGING removed. Probe only.
//
// The other half of the ablation described in `sdpa_nomath_mma.metal`. The
// barriers stay, the loop structure stays, and the MMAs run on whatever
// happens to be in threadgroup memory -- numerically meaningless, timing-valid,
// and exactly the arithmetic the real kernel issues.
#define PIE_SDPA_NO_STAGE 1
#include "../../../src/kernels/sdpa_paged_mma.metal"
