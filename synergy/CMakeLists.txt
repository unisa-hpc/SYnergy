add_library(synergy INTERFACE)

target_include_directories(synergy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(SYNERGY_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)

	if(NOT CUDAToolkit_FOUND)
		message(SEND_ERROR "CUDA Toolkit not found")
	endif()

	target_compile_definitions(synergy INTERFACE SYNERGY_CUDA_SUPPORT)
	target_link_libraries(synergy INTERFACE CUDA::nvml)
endif()

if(SYNERGY_ROCM_SUPPORT)
	find_package(rocm_smi)
	if(NOT rocm_smi_FOUND)
		message(SEND_ERROR "ROCm SMI library not found")
	endif()

	target_compile_definitions(synergy INTERFACE SYNERGY_ROCM_SUPPORT)
	target_link_libraries(synergy INTERFACE ${ROCM_SMI_LIBRARIES})
endif()