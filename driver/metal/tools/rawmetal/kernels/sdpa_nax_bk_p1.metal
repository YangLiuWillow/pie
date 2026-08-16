// NAXB_NPG=1 pages per key block = 32 keys. NPG=1 reproduces
// sdpa_nax_prefill exactly and is the control.
#define NAXB_NPG 1
#include "sdpa_nax_bk.metal"
