# SYnergy
Header-Only SYCL queue wrapper for energy measurement and frequency scaling.  
SYnergy allows to get standard power-related feature such as per-application and per-kernel energy measurements as well as frequency scaling capabilities, all with minimal configuration. 
Currently supported target architecture: 
- NVIDIA GPUs supported through the [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml).

## Build
### Dependencies
- A supported SYCL implementation:
	- hipSYCL
	- DPC++
- Cmake (3.13 or newer)  
- NVML (Nvidia support)
- A C++17 or newer compiler

To build SYnergy, type:
```bash
	cd SYnergy
	mkdir build && cd build/
	cmake .. -DSY_SYCL_BACKEND=[hipSYCL | dpcpp] -DSY_CUDA_SUPPORT=[ON | OFF]
```

## Usage
To use SYnergy, just swap your current `sycl::queue` with `synergy::queue`. Under the `samples/` folder you can find an example of SYnergy usage.
