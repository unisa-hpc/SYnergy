# SYnergy
SYCL library for energy measurement and frequency scaling.
SYnergy allows to get standard power-related feature such as per-application and per-kernel energy measurements as well as frequency scaling capabilities, all with minimal configuration. 
Currently supported target architectures: 
- NVIDIA GPUs supported through the [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml).
- AMD GPUs supported through the [ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib)

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
	cmake .. -DSYNERGY_SYCL_BACKEND=[hipSYCL | dpcpp] -DSYNERGY_CUDA_SUPPORT=[ON | OFF]
```

## Usage
To use SYnergy, just swap your current `sycl::queue` with `synergy::queue`. Under the `samples/` folder you can find an example of SYnergy usage.
