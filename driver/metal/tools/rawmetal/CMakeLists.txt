function(add_rawmetal_tool target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE pie_driver_metal_lib)
  target_compile_definitions(${target} PRIVATE
    PIE_METAL_TOOL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels"
    PIE_METAL_TOOL_LOCAL_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
endfunction()

# `decoder_smoke` was removed with `NativeAccess`. It was the standalone gate
# for driving `MetalExecutor` on a real checkpoint, written when it was the only
# way to do so -- but it reached the executor through a private `*_native`
# surface and a `friend` declaration, and its `setup_native` hard-coded
# `arch_name = "qwen3_5"` and passed no config facts, so it could not load any
# checkpoint that shipped after it was written. `tests/llama_bench` does what it
# was for through the PUBLIC path, on four families, gated against mlx-lm.
add_rawmetal_tool(rawmetal_harness harness_main.cpp harness.cpp)
add_rawmetal_tool(rawmetal_microbench bench_kernels.cpp harness.cpp)
add_rawmetal_tool(roofline_probe roofline_probe.cpp harness.cpp)
# Prices pie's PAGED attention on its own, against MLX's contiguous SDPA on
# the same shapes -- the comparison `harness.hpp` says bench_kernel exists for.
add_rawmetal_tool(sdpa_paged_probe sdpa_paged_probe.cpp harness.cpp)
# Prices the M5 neural accelerators against `simdgroup_matrix`, which is the
# last unmeasured explanation for pie's attention gap against MLX.
add_rawmetal_tool(matrix_rate_probe matrix_rate_probe.cpp harness.cpp)
# Prices the ROUTED MoE GEMM, which the dispatch trace makes the largest term
# in a cold prefill once attention moved off the simdgroup matrix unit.
add_rawmetal_tool(qmm_nax_probe qmm_nax_probe.cpp harness.cpp)
# Prices moe_route_sort ALONE. In situ it is 11.6% of a decode step for a body
# of ~150 operations, and only dispatching it by itself separates the kernel's
# own cost from what the dispatch chain does to it.
add_rawmetal_tool(moe_sort_probe moe_sort_probe.cpp harness.cpp)
add_rawmetal_tool(q35contract_probe q35contract_probe.cpp)
if(PIE_LOADER_STATICLIB)
  target_link_libraries(q35contract_probe PRIVATE ${PIE_LOADER_STATICLIB})
endif()
add_rawmetal_tool(g4contract_probe g4contract_probe.cpp)
if(PIE_LOADER_STATICLIB)
  target_link_libraries(g4contract_probe PRIVATE ${PIE_LOADER_STATICLIB})
endif()
add_rawmetal_tool(gptosscontract_probe gptosscontract_probe.cpp)
if(PIE_LOADER_STATICLIB)
  target_link_libraries(gptosscontract_probe PRIVATE ${PIE_LOADER_STATICLIB})
endif()
add_rawmetal_tool(plan_cost_probe plan_cost_probe.cpp)
if(PIE_LOADER_STATICLIB)
  target_link_libraries(plan_cost_probe PRIVATE ${PIE_LOADER_STATICLIB})
endif()
add_rawmetal_tool(sparse_probe sparse_probe.mm)
add_rawmetal_tool(sparse_probe2 sparse_probe2.mm)
add_rawmetal_tool(sparse_probe3 sparse_probe3.mm)

# `st_probe` and `gemma4_decode_run` were removed with §6: both were bring-up
# harnesses built on `SafetensorsView`, the driver-side name-addressed
# checkpoint reader. The plan now carries the byte offsets, so nothing in a
# driver resolves a tensor by name any more and the capability they were written
# against no longer exists.
add_rawmetal_tool(
  gemma4_encode_probe
  gemma4_encode_probe.cpp
  gemma4_encode.cpp)

# The threadgroup-memory cap the attention kernels reason about but never ask
# for. Framework-only and standalone, so it stays answerable without the driver.
add_executable(device_caps device_caps.mm)
set_source_files_properties(device_caps.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(device_caps PRIVATE "-framework Metal" "-framework Foundation")

add_executable(dvfs_probe dvfs_probe.mm)
set_source_files_properties(dvfs_probe.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(dvfs_probe PRIVATE
  "-framework Metal"
  "-framework Foundation")
