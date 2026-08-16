// QH=8 query heads per threadgroup, RT=4 row-tiles (BQ=64). Threads = 1024.
#define NAXS_QH 8
#define NAXS_RT 4
#include "sdpa_nax_stg.metal"
