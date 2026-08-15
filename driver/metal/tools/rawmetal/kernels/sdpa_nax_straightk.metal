// Stage 2c with K staged STRAIGHT and transposed by the instruction, instead
// of transposed by the staging write. Isolates one thing: the strided
// threadgroup write. Everything else -- tile, fill, matmul count, V path --
// is identical to sdpa_nax_staged.
#define PIE_NAX_STAGE 4
#define PIE_NAX_KT_STRAIGHT 1
#include "sdpa_nax_qk.metal"
