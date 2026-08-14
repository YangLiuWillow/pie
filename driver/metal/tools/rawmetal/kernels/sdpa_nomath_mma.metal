// The paged MMA kernel with the MULTIPLY removed. Probe only.
//
// Half of a two-sided ablation. This keeps every load, every float->half
// conversion, every threadgroup write and both barriers, and replaces the
// QK^T / softmax / PV block with a single add that consumes the staged tiles.
// `sdpa_nostage_mma.metal` is the other half.
//
// Together they answer the question seven falsified hypotheses were guessing
// at: of the ~6.2 ms/layer this kernel costs, how much is MOVING the keys into
// threadgroup memory and how much is MULTIPLYING them? Every remaining idea
// forks on that number -- untransposed K staging and a wider query tile only
// help the move; MLX's decomposition is the only thing that helps the multiply.
//
// READ THE HALVES AS A DIRECTION, NOT AN ATTRIBUTION. Deleting the MMAs frees
// registers and can raise occupancy, so this half is if anything optimistic
// about how cheap staging is. The check that keeps it honest is whether the two
// halves roughly SUM to the unablated kernel; if they fall well short, the
// interaction is itself the finding.
#define PIE_SDPA_NO_MATH 1
#include "../../../src/kernels/sdpa_paged_mma.metal"
