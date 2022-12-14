cmake_minimum_required(VERSION 3.13)
project(SYnergy VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYNERGY_CUDA_SUPPORT "Enable CUDA support" OFF)

set(SYNERGY_SYCL_BACKEND "" CACHE STRING "Select SYCL backend [hipSYCL | dpc++]")

if(SYNERGY_SYCL_BACKEND STREQUAL hipSYCL)
	find_package(hipSYCL REQUIRED)
endif()

if(SYNERGY_SYCL_BACKEND STREQUAL "dpc++") # CMAKE_CXX_COMPILER must be set to clang++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	if(SYNERGY_CUDA_SUPPORT)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
		set(SYNERGY_CUDA_ARCH "" CACHE STRING "CUDA architecture in format \"sm_xy\"")

		if(NOT SYNERGY_CUDA_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SYNERGY_CUDA_ARCH}")
		endif()

	endif()
endif()


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo."	FORCE)	
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

option(SYNERGY_BUILD_SAMPLES "Build samples" OFF)
if(SYNERGY_BUILD_SAMPLES)
	add_subdirectory(samples)
endif()

add_subdirectory(synergy)
