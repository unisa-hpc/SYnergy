add_executable(matrix_mul matrix_mul.cpp)

find_package (Threads)
target_link_libraries(matrix_mul PUBLIC synergy Threads::Threads)

if(SYCL_BACKEND STREQUAL hipSYCL)
	add_sycl_to_target(TARGET matrix_mul)
endif()