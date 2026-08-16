// QH=2 query heads per threadgroup, RT=4 row-tiles (BQ=64). Threads = 256.
#define NAXS_QH 2
#define NAXS_RT 4
#include "sdpa_nax_stg.metal"
