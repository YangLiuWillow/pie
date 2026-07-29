#pragma once

// Env-var-derived runtime knobs shared between Qwen3.5 (non-MoE) and
// Qwen3.5-MoE. Values are cached on first call.

namespace pie_cuda_driver::model {

int  qwen35_small_spec_graph_tokens();
bool qwen35_forward_profile_enabled();
// Route pure-decode batches through the paged-PREFILL kernel (FlashInfer's
// tensor-core path at qo_len=1) instead of BatchDecodeWithPagedKVCache (its
// CUDA-core path). `force_prefill_path` already does exactly this, but it used
// to be reached only as a *fallback* for GQA ratios outside FlashInfer's decode
// dispatch set {1,2,3,4,8}.
//
// ON BY DEFAULT for gqa_group >= 4, which is FlashInfer's own threshold for the
// tensor-core path paying off. Measured on Qwen3-Coder-30B-A3B (gqa 8, H200):
// KV read 1748 -> 4068 GB/s (36% -> 85% of peak, past vLLM's 3708), decode
// +20.8% end-to-end, and aggregate throughput 1.15x-2.34x better at every
// concurrency from 1 to 96 with no regression anywhere.
//
// Ratios below 4 keep the CUDA-core kernel: untested here, and FlashInfer does
// not expect tensor cores to pay at those group sizes.
//
// PIE_QWEN35_TENSOR_CORE_DECODE=0 forces the old path, =1 forces the new one
// regardless of ratio.
bool qwen35_tensor_core_decode_enabled(int gqa_group);
int  qwen35_mtp_draft_position_offset();
bool qwen35_mtp_fused_gemv_enabled();
bool qwen35_mtp_prefix_global_cache();

}  // namespace pie_cuda_driver::model
