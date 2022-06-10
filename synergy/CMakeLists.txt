add_library(synergy INTERFACE)
find_package(CUDAToolkit REQUIRED)


if(CUDA_SUPPORT)
	target_compile_definitions(synergy INTERFACE CUDA_SUPPORT)
endif()

set(header_files 
	"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/synergy.hpp",
	"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/queue.hpp",
	"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/energy_interface.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/energy.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/energy_implementations.hpp"
)

if(CUDA_SUPPORT)
	set(cuda_header_files 
		"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/cuda/utils.hpp",
		"${CMAKE_CURRENT_SOURCE_DIR}/include/synergy/cuda/energy_nvidia.hpp"
	)
endif()

set_target_properties(synergy
	PROPERTIES
	PUBLIC_HEADER ${header_files}
)

if(CUDA_SUPPORT)
	set_target_properties(synergy
		PROPERTIES
		PUBLIC_HEADER ${cuda_header_files}
	)
endif()

target_include_directories(synergy
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(synergy 
	INTERFACE 
	CUDA::nvml
	Threads::Threads
)
