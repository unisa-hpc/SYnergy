cmake_minimum_required(VERSION 3.13)
project(synergy_project VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYCL_BACKEND "Select SYCL backend [hipSYCL | dpcpp]" "")

find_package(Threads REQUIRED)
if(SYCL_BACKEND STREQUAL hipSYCL)
	find_package(hipSYCL REQUIRED)
endif()


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build" FORCE)
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

add_subdirectory(synergy)
add_subdirectory(samples EXCLUDE_FROM_ALL)
