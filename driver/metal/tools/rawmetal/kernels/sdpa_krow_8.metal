// KROWS=8. Same kernel, one constant different -- so the k-vs-1 ratio is the
// sharing and nothing else.
#define KROWS 8
#include "sdpa_krow_decode.metal"
