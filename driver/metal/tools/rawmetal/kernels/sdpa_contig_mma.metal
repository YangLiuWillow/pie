// The paged MMA attention kernel with the page walk removed. Probe only.
//
// `sdpa_paged_probe` runs this against `sdpa_paged_mma` to answer one question:
// how much of pie's 7.47 ms/layer is the PAGE TABLE, and how much is the kernel?
//
// The probe binds an IDENTITY page table (`kv_page_indices[p] == p`), so the
// two kernels read exactly the same bytes in exactly the same order -- the KV
// is already physically contiguous under the paged path. Nothing about
// locality differs. What differs is that the paged staging loop computes, for
// every element it stages,
//
//     page = kv_page_indices[page_base + kp / page_size]
//     slot = page * page_size + kp % page_size
//
// -- an integer divide, a modulo and a dependent load -- where this one uses
// `kp` directly. Both produce the SAME `slot`. So the difference in runtime is
// the cost of the addressing and nothing else, which is the quantity a
// "materialize the pages into scratch, then compute" design would actually buy.
//
// One `#define` and one `#include`, deliberately: a copy of the 400-line kernel
// would drift from the original and then the A/B would be measuring the drift.
#define PIE_SDPA_CONTIG_KV 1
#include "../../../src/kernels/sdpa_paged_mma.metal"
