# SYnergy
Header-Only energy wrapper for SYCL.  
SYnergy allows to get kernel-level energy measuremen with minimal configuration needed.  
Actually, only Nvidia hardware are supported through the [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml).

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
	cmake .. -DSYCL_BACKEND=[hipSYCL | dpcpp] -DCUDA_SUPPORT=[ON | OFF]
```

## Usage
To use SYnergy, just swap your current `sycl::queue` with `synergy::queue`. Under the `samples/` folder you can find an example of SYnergy usage.
