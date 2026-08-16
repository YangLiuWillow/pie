// QH=4 query heads per threadgroup, RT=4 row-tiles (BQ=64). Threads = 512.
#define NAXS_QH 4
#define NAXS_RT 4
#include "sdpa_nax_stg.metal"
