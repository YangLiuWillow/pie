#pragma once

// Env-var-derived runtime knobs shared between Qwen3.5 (non-MoE) and
// Qwen3.5-MoE. Values are cached on first call.

namespace pie_cuda_driver::model {

int  qwen35_small_spec_graph_tokens();
bool qwen35_forward_profile_enabled();
// Route pure-decode batches through the paged-PREFILL kernel (FlashInfer's
// tensor-core path at qo_len=1) instead of BatchDecodeWithPagedKVCache (its
// CUDA-core path). `force_prefill_path` already does exactly this, but today it
// is only ever set as a *fallback* for GQA ratios outside FlashInfer's decode
// dispatch set {1,2,3,4,8}. This turns the same route into a deliberate choice
// for ratios that are in the set, where FlashInfer's own guidance is that the
// tensor-core path wins once the GQA group is >= 4.
bool qwen35_tensor_core_decode_enabled();
int  qwen35_mtp_draft_position_offset();
bool qwen35_mtp_fused_gemv_enabled();
bool qwen35_mtp_prefix_global_cache();

}  // namespace pie_cuda_driver::model
