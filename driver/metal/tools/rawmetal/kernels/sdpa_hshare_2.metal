// HEADS=2. Same kernel, one constant different -- so the HEADS-vs-1 ratio is
// the head sharing and nothing else. HEADS=1 IS `sdpa_paged_decode`'s shape.
#define HEADS 2
#include "sdpa_hshare_decode.metal"
