#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

bool flashinfer_cutlass_moe_enabled();

std::size_t flashinfer_cutlass_moe_workspace_bytes(
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank);

bool flashinfer_cutlass_moe_bf16_relu2(
    const std::uint16_t* input,
    const std::int32_t* token_selected_experts,
    const float* token_final_scales,
    const std::uint16_t* fc1_expert_weights,
    const std::uint16_t* fc2_expert_weights,
    std::uint16_t* output,
    std::uint8_t* workspace,
    std::size_t workspace_bytes,
    std::int32_t* unpermuted_row_to_permuted_row,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank,
    cudaStream_t stream);

// Variable-M grouped GEMM over expert-sorted rows (the non-gated CUTLASS
// prefill route). A: [rows, k] bf16, expert-contiguous with padding; B: the
// weight tensor as Pie stores it, [E, n, k] row-major; C: [rows, n];
// expert_offsets: DEVICE int64 cumulative padded rows per expert (length E).
// Throws on setup failure; asynchronous on `stream`.
void cutlass_moe_grouped_gemm_bf16(const void* A, const void* B, void* C,
                                   const long long* expert_offsets,
                                   std::int64_t rows, std::int64_t n,
                                   std::int64_t k, int num_experts,
                                   cudaStream_t stream);

}  // namespace pie_cuda_driver::ops
