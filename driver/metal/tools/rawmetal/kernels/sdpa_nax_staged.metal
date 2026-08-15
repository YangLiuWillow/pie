// Stage 2c: the complete kernel WITH real staging from device memory.
// This is the number that compares against the shipped 6.87 ms/layer and
// MLX's 1.31 -- everything before it held staging out to isolate the multiply.
#define PIE_NAX_STAGE 4
#include "sdpa_nax_qk.metal"
