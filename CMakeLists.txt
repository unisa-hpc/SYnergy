cmake_minimum_required(VERSION 3.13)
project(SYnergy VERSION 0.1.1 LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SY_SYCL_BACKEND "" CACHE STRING "Select SYCL backend [hipSYCL | dpcpp]")
option(SY_CUDA_SUPPORT "Enable CUDA support" OFF)

if(SY_SYCL_BACKEND STREQUAL hipSYCL)
	find_package(hipSYCL REQUIRED)
endif()

if(SY_SYCL_BACKEND STREQUAL dpcpp) # CMAKE_CXX_COMPILER must be set to clang++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	if(SY_CUDA_SUPPORT)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
		set(SY_CUDA_ARCH "" CACHE STRING "CUDA architecture in format \"sm_xy\"")

		if(NOT SY_CUDA_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SY_CUDA_ARCH}")
		endif()

	endif()
endif()


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo."	FORCE)	
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

option(SY_BUILD_SAMPLES "Build samples" OFF)
if(SY_BUILD_SAMPLES)
	add_subdirectory(samples)
endif()

add_subdirectory(synergy)
