cmake_minimum_required(VERSION 3.17)
project(SYnergy VERSION 0.1.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYNERGY_CUDA_SUPPORT "Enable CUDA support" OFF)
option(SYNERGY_ROCM_SUPPORT "Enable ROCm support" OFF)
option(SYNERGY_LZ_SUPPORT "Enable Level Zero Support" OFF)

set(SYNERGY_SYCL_IMPL "" CACHE STRING "Select SYCL implementation [OpenSYCL | DPC++]")
set_property(CACHE SYNERGY_SYCL_IMPL PROPERTY STRINGS "OpenSYCL" "DPC++")

if(SYNERGY_SYCL_IMPL STREQUAL "OpenSYCL")
	find_package(OpenSYCL)
	
	if(NOT OpenSYCL_FOUND)
		find_package(hipSYCL)
	endif()

	if(NOT OpenSYCL_FOUND AND NOT hipSYCL_FOUND)
		message(SEND_ERROR "OpenSYCL or hipSYCL package were not found. Specify OpenSYCL_DIR or hipSYCL_DIR.")
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

option(SYNERGY_DEVICE_PROFILING "Enable device energy consumption profiling" OFF)
option(SYNERGY_KERNEL_PROFILING "Enable kernel energy consumption profiling" OFF)

if(SYNERGY_DEVICE_PROFILING)
	target_compile_definitions(synergy INTERFACE SYNERGY_DEVICE_PROFILING)
endif()

if(SYNERGY_KERNEL_PROFILING)
	target_compile_definitions(synergy INTERFACE SYNERGY_KERNEL_PROFILING)
endif()

if(SYNERGY_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)

	target_compile_definitions(synergy INTERFACE "SYNERGY_CUDA_SUPPORT")
	target_link_libraries(synergy INTERFACE CUDA::nvml)
	target_sources(synergy INTERFACE include/vendors/nvml_wrapper.hpp)
endif()

if(SYNERGY_ROCM_SUPPORT)
	find_package(rocm_smi REQUIRED)

	target_compile_definitions(synergy INTERFACE SYNERGY_ROCM_SUPPORT)
	target_link_libraries(synergy INTERFACE ${ROCM_SMI_LIBRARIES})
	target_sources(synergy INTERFACE include/vendors/rsmi_wrapper.hpp)
endif()

if(SYNERGY_LZ_SUPPORT)
	find_package(LevelZero REQUIRED)

	target_compile_definitions(synergy INTERFACE SYNERGY_LZ_SUPPORT)
	target_link_libraries(synergy INTERFACE LevelZero::LevelZero)
	target_sources(synergy INTERFACE include/vendors/lz_wrapper.hpp)
endif()

# ##################### Samples ######################
option(SYNERGY_BUILD_SAMPLES "Build samples" OFF)

if(SYNERGY_BUILD_SAMPLES)
	add_subdirectory(samples)
endif()
