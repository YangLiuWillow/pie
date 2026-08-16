// NAXB_NPG=4 pages per key block = 128 keys. NPG=1 reproduces
// sdpa_nax_prefill exactly and is the control.
#define NAXB_NPG 4
#include "sdpa_nax_bk.metal"
