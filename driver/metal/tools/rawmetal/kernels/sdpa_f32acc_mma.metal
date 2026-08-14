// The paged MMA kernel with MLX's arithmetic on pie's tile. Probe only.
//
// The move-vs-multiply ablation put 3.38 of the kernel's 6.87 ms/layer in the
// MULTIPLY, and MLX does the move AND the multiply in 1.555 -- so the
// arithmetic is the wall and no staging fix can reach it. Reading MLX 0.31.3's
// own sources settled what is different, and it is NOT the tiling: every bf16
// d=128 kernel it ships is `wm4_wn1` (4 simdgroups, 128 threads) and its
// smallest is `bq32_bk16_bd128` -- pie's exact shape.
//
// What differs is the arithmetic's type and shape:
//   * MLX accumulates S and O in fp32; pie accumulates in half.
//   * Because of that MLX needs no chunking: its QK loop is one unbroken chain
//     of BD/8 = 16 matmads. pie's DCH=4 exists only to bound half-rounding.
//   * MLX holds fragments as `vec<float,2>` thread values, materializing a
//     `simdgroup_matrix` only inside the multiply. pie holds live
//     `simdgroup_matrix` objects across the loop.
//
// THIS EXPERIMENT CONTRADICTS A RECORDED MEASUREMENT and is run for that
// reason. pie measured DCH=16 -- one unbroken chain, MLX's shape -- as 36%
// SLOWER, and blamed register pressure. MLX runs that shape in a WIDER type and
// wins. The third bullet is the candidate reconciliation: the two kernels give
// the register allocator different programs. So all three move together here,
// because testing the accumulator type alone would re-run the experiment that
// already failed.
#define PIE_SDPA_F32ACC 1
#include "../../../src/kernels/sdpa_paged_mma.metal"
