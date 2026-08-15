// The move half of the NAX kernel: real staging, no matmul. Pairs with
// sdpa_nax_full (multiply, no real staging) to split the kernel the same way
// sdpa_nomath/sdpa_nostage split the shipped one.
#define PIE_NAX_STAGE 5
#include "sdpa_nax_qk.metal"
