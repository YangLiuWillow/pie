# MLX reference implementation and its consumers.

add_subdirectory(mlx)

if(PIE_METAL_BUILD_TESTS)
  add_executable(smoke_load smoke_load.cpp)
  target_link_libraries(smoke_load PRIVATE pie_metal_mlx_test_support)

  add_executable(kv_perlayer_test kv_perlayer_test.cpp)
  target_link_libraries(kv_perlayer_test PRIVATE pie_metal_mlx_test_support)
  add_test(NAME kv_perlayer_test COMMAND kv_perlayer_test)

  # Header-only policy: no Metal device, no mlx support library. Deliberately
  # so -- the program cache's eviction decision is the part that can be
  # silently wrong, and a test that needs a GPU is a test nobody runs while
  # someone else is measuring on it.
  add_executable(program_cache_policy_test program_cache_policy_test.cpp)
  target_include_directories(program_cache_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  add_test(NAME program_cache_policy_test COMMAND program_cache_policy_test)
endif()

if(PIE_METAL_BUILD_PARITY)
  add_subdirectory(parity)
endif()
