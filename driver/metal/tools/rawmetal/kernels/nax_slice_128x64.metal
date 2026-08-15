// Tile sweep: is 2.30x a floor, or the tile I happened to pick? The slice API
// never touches threadgroup memory -- the library manages its own -- so the
// 32 KB cap that forced BQ=64/BK=32 on the hand-filled kernel does not bind
// here, and larger tiles are simply available.
#define NAX_BQ 128
#define NAX_BK 64
#include "nax_slice_qk.metal"
