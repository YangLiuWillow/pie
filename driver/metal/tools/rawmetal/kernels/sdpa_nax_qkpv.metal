// Stage 2: Q.K^T + P.V, no softmax. Isolates the O accumulator.
// 64 floats a lane against the 8x8 kernel's 32 -- a spill here is the thing
// most likely to kill the design, so it is measured before softmax is added.
#define PIE_NAX_STAGE 2
#include "sdpa_nax_qk.metal"
