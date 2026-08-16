// NAXP_NWARPS=4, i.e. a 64-row query tile per threadgroup. Same
// kernel, one constant different. Registers per lane do NOT change with this --
// a simdgroup owns 16 rows either way -- so what it sweeps is threadgroup size
// and grid height, which is an occupancy question and has to be measured.
#define NAXP_NWARPS 4
#include "sdpa_nax_prefill.metal"
