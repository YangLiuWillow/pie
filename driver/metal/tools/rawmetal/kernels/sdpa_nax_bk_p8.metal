// NAXB_NPG=8 pages per key block = 256 keys. NPG=1 reproduces
// sdpa_nax_prefill exactly and is the control.
#define NAXB_NPG 8
#include "sdpa_nax_bk.metal"
