// NAXB_NPG=2 pages per key block = 64 keys. NPG=1 reproduces
// sdpa_nax_prefill exactly and is the control.
#define NAXB_NPG 2
#include "sdpa_nax_bk.metal"
