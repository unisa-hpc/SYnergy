add_library(synergy INTERFACE)

target_include_directories(synergy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(synergy INTERFACE	Threads::Threads)

if(SY_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)

	if(NOT CUDAToolkit_FOUND)
		message(SEND_ERROR "CUDA Toolkit not found")
	endif()

	target_compile_definitions(synergy INTERFACE SY_CUDA_SUPPORT)
	target_link_libraries(synergy INTERFACE CUDA::nvml)
endif()