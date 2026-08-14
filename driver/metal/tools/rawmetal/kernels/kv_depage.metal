// Materialize a request's paged KV into one contiguous run. Probe only.
//
// This is the OTHER half of the de-paging question. `sdpa_contig_mma.metal`
// prices what a contiguous kernel saves; this prices what it costs to hand one
// a contiguous buffer in the first place. The two together are the whole
// design: gather once per layer, then attend over the result.
//
// It is written the way a real one would be, not the way a probe would be:
// 16 bytes per thread (eight bfloats, one `uint4`), fully coalesced, one page
// lookup per 16 bytes rather than per element. K and V are copied in the same
// thread because they share the lookup -- the address arithmetic is paid once
// and spent twice, which is the only interesting optimization available here.
//
// The traffic is exact and worth stating, because the result has to be judged
// against it rather than against a guess: at 7424 keys x 4 kv heads x 128 dim
// x 2 bytes, K and V are 7.6 MB each. Read both, write both: 30.4 MB moved.
// A pure-bandwidth floor at ~200 GB/s is therefore ~0.15 ms, and anything much
// above that means this kernel, not the design, is the thing being measured.

#include <metal_stdlib>
using namespace metal;

kernel void kv_depage_16b(
    const device uint4* k_pages         [[buffer(0)]],
    const device uint4* v_pages         [[buffer(1)]],
    device uint4* k_out                 [[buffer(2)]],
    device uint4* v_out                 [[buffer(3)]],
    const device uint* kv_page_indices  [[buffer(4)]],
    const constant int& page_base       [[buffer(5)]],
    const constant int& page_size       [[buffer(6)]],
    // uint4s per key: n_kv_heads * head_dim * sizeof(T) / 16. A key's whole
    // row is a multiple of 16 bytes at every geometry this driver serves, so a
    // vector never straddles two keys and the index split below is exact.
    const constant int& vec_per_key     [[buffer(7)]],
    const constant int& n_keys          [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const int total = n_keys * vec_per_key;
  if (int(gid) >= total) return;

  const int key = int(gid) / vec_per_key;
  const int rem = int(gid) - key * vec_per_key;
  const int page = int(kv_page_indices[page_base + key / page_size]);
  const int slot = page * page_size + (key % page_size);
  const int src = slot * vec_per_key + rem;

  k_out[gid] = k_pages[src];
  v_out[gid] = v_pages[src];
}
