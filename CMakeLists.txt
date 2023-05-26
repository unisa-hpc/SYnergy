cmake_minimum_required(VERSION 3.13)
project(SYnergy VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYNERGY_CUDA_SUPPORT "Enable CUDA support" OFF)
option(SYNERGY_ROCM_SUPPORT "Enable ROCm support" OFF)
option(SYNERGY_LZ_SUPPORT "Enable Level Zero Support" OFF)

set(SYNERGY_SYCL_BACKEND "" CACHE STRING "Select SYCL backend [hipSYCL | dpcpp]")

if(SYNERGY_SYCL_BACKEND STREQUAL hipSYCL)
	find_package(hipSYCL REQUIRED)
endif()

if(SYNERGY_SYCL_BACKEND STREQUAL "dpcpp") # CMAKE_CXX_COMPILER must be set to clang++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	if(SYNERGY_CUDA_SUPPORT)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

		set(SYNERGY_CUDA_ARCH "" CACHE STRING "CUDA architecture in format \"sm_xy\"")

		if(NOT SYNERGY_CUDA_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SYNERGY_CUDA_ARCH}")
		endif()
	endif()

	if(SYNERGY_ROCM_SUPPORT)
		set(SYNERGY_ROCM_ARCH "" CACHE STRING "ROCm architecture number")

		if(NOT SYNERGY_ROCM_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amd_gpu_gfx${SYNERGY_ROCM_ARCH}")
		else()
			message(SEND_ERROR "Variable SYNERGY_ROCM_ARCH must be set to the AMD GPU architecture")
		endif()
	endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo." FORCE)
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# ###################### SYnergy ######################
find_package(Threads REQUIRED)

add_library(synergy INTERFACE)

target_link_libraries(synergy INTERFACE Threads::Threads)

target_include_directories(synergy INTERFACE include)

option(SYNERGY_ENABLE_PROFILING "Enable energy consumption profiling" OFF)

if(SYNERGY_ENABLE_PROFILING)
	target_compile_definitions(synergy INTERFACE SYNERGY_ENABLE_PROFILING)
endif()

if(SYNERGY_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)

	if(NOT CUDAToolkit_FOUND)
		message(SEND_ERROR "CUDA Toolkit not found")
	endif()

	target_compile_definitions(synergy INTERFACE "SYNERGY_CUDA_SUPPORT")
	target_link_libraries(synergy INTERFACE CUDA::nvml)
	target_sources(synergy INTERFACE include/vendors/nvml_wrapper.hpp)
endif()

if(SYNERGY_ROCM_SUPPORT)
	find_package(rocm_smi)

	if(NOT rocm_smi_FOUND)
		message(SEND_ERROR "ROCm SMI library not found")
	endif()

	target_compile_definitions(synergy INTERFACE SYNERGY_ROCM_SUPPORT)
	target_link_libraries(synergy INTERFACE ${ROCM_SMI_LIBRARIES})
	target_sources(synergy INTERFACE include/vendors/rsmi_wrapper.hpp)
endif()

if(SYNERGY_LZ_SUPPORT)
	target_compile_definitions(synergy INTERFACE SYNERGY_LZ_SUPPORT)
	target_sources(synergy INTERFACE include/vendors/lz_wrapper.hpp)
	if (($ENV{ZES_ENABLE_SYSMAN} STREQUAL "") OR ($ENV{ZES_ENABLE_SYSMAN} STREQUAL ""))
		message(WARNING "Remember to set the env variable ZES_ENABLE_SYSMAN=1 before executing")
	endif()
endif()

# ##################### Samples ######################
option(SYNERGY_BUILD_SAMPLES "Build samples" OFF)

if(SYNERGY_BUILD_SAMPLES)
	add_subdirectory(samples)
endif()
