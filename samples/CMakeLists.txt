link_libraries(synergy)

option(SYNERGY_SAMPLES_ALL_WARNINGS "Enable all warnings during SYnergy samples compilation" OFF)

if(SYNERGY_SYCL_IMPL STREQUAL "OpenSYCL" AND SYNERGY_LZ_SUPPORT)
  message(STATUS "The supported OpenSYCL target for Level Zero is \"sscp\"")
endif()

add_executable(matrix_mul matrix_mul/matrix_mul.cpp)
add_executable(query_freq query_freq/query_freq.cpp)
add_executable(saxpy saxpy/saxpy.cpp)
add_executable(concurrent_matmul concurrent_matmul/concurrent_matmul.cpp)
add_executable(freq_scale freq_scale/freq_scale.cpp)
add_executable(task_graph task_graph/task_graph.cpp)
add_executable(freq_overhead freq_overhead/freq_overhead.cpp)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run.sh
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/samples/freq_overhead/run.sh ${CMAKE_CURRENT_BINARY_DIR}/run.sh
  DEPENDS ${CMAKE_SOURCE_DIR}/samples/freq_overhead/run.sh
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/parse.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/samples/freq_overhead/parse.py ${CMAKE_CURRENT_BINARY_DIR}/parse.py
  DEPENDS ${CMAKE_SOURCE_DIR}/samples/freq_overhead/parse.py
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/plot.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/samples/freq_overhead/plot.py ${CMAKE_CURRENT_BINARY_DIR}/plot.py
  DEPENDS ${CMAKE_SOURCE_DIR}/samples/freq_overhead/plot.py
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Brommy.bmp
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/samples/freq_overhead/Brommy.bmp ${CMAKE_CURRENT_BINARY_DIR}/Brommy.bmp
  DEPENDS ${CMAKE_SOURCE_DIR}/samples/freq_overhead/Brommy.bmp
)

add_custom_target(copy_scripts ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/run.sh ${CMAKE_CURRENT_BINARY_DIR}/parse.py ${CMAKE_CURRENT_BINARY_DIR}/plot.py ${CMAKE_CURRENT_BINARY_DIR}/Brommy.bmp
)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

foreach(target IN LISTS all_targets)
  if(SYNERGY_SYCL_IMPL STREQUAL "OpenSYCL")
    add_sycl_to_target(TARGET ${target})
  endif()

  if(SYNERGY_SAMPLES_ALL_WARNINGS)
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/D_CRT_SECURE_NO_WARNINGS /MP /W4 /permissive- /bigobj /Zi /utf-8>
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -pedantic -g -march=native>
      $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
    )
  endif()

  if(SYNERGY_SYCL_IMPL STREQUAL "DPC++") # CMAKE_CXX_COMPILER must be set to clang++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    if(SYNERGY_CUDA_SUPPORT)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

      set(SYNERGY_SAMPLES_CUDA_ARCH "" CACHE STRING "CUDA device architecture e.g. sm_70")

      if(NOT SYNERGY_SAMPLES_CUDA_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SYNERGY_SAMPLES_CUDA_ARCH}")
      endif()
    endif()

    if(SYNERGY_ROCM_SUPPORT)
      set(SYNERGY_SAMPLES_ROCM_ARCH "" CACHE STRING "ROCm device architecture e.g. gfx908")

      if(NOT SYNERGY_SAMPLES_ROCM_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amd_gpu_${SYNERGY_SAMPLES_ROCM_ARCH}")
      endif()
    endif()

    if(SYNERGY_LZ_SUPPORT)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

      set(SYNERGY_SAMPLES_LZ_ARCH "" CACHE STRING "Level Zero device architecture e.g. acm-g10")
      if(NOT SYNERGY_SAMPLES_ROCM_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      endif()
    endif()

  endif()
endforeach()