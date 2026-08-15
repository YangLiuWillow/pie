// Correctness mode: real staging, Q.K^T only, S written out raw for kb=0.
// Nothing about the 1.652 ms figure says the kernel computes ATTENTION -- a
// wrong operand orientation does the same FLOPs at the same speed.
#define PIE_NAX_STAGE 6
#include "sdpa_nax_qk.metal"
