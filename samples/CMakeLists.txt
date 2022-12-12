link_libraries(synergy)

add_executable(matrix_mul matrix_mul/matrix_mul.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

if(SY_SYCL_BACKEND STREQUAL hipSYCL)
  foreach(target IN LISTS all_targets)
    add_sycl_to_target(TARGET ${target})
  endforeach()
endif()