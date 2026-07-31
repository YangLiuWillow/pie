# Patches applied to the vendored FlashInfer tree (CPM cache) at build setup

`moe_gemm_tma_ws_launcher_fp8_alpha_gate.patch` — the vendored 3rdparty/cutlass
predates the `alpha_ptr_array` member on `LinearCombination::Arguments`, so the
.inl's per-expert alpha-scale epilogue constructs (an FP8-dequant feature,
always nullptr for bf16) fail to compile for every dtype. Gate those two
`construct_if_true` conditions on `IsFP8`. Apply with:
  cd /root/.cpm-cache/flashinfer/<rev>  # or wherever CPM unpacked flashinfer
  patch -p1 < patches/moe_gemm_tma_ws_launcher_fp8_alpha_gate.patch
Regenerate the sm90 TU set after bumping FlashInfer:
  run regenerate.sh with arch "90-real" and keep only files invoking
  INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM, filtered to __nv_bfloat16
  instantiations (see gemm_grouped/90/).
