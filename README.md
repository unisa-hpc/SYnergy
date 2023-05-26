# SYnergy
SYCL library for energy measurement and frequency scaling.
SYnergy allows to get standard power-related feature such as per-application and per-kernel energy measurements as well as frequency scaling capabilities, all with minimal configuration. 
Currently supported target architectures: 
- NVIDIA GPUs supported through the [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml).
- AMD GPUs supported through the [ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib)
- Intel GPUs through the [Sysman API](https://spec.oneapi.io/level-zero/latest/sysman/PROG.html)

## Build
### Dependencies
- cmake (3.17 or newer)  
- C++17 or newer compiler
- A supported SYCL implementation:
	- DPC++
	- OpenSYCL
- A supported target architecture
	- CUDA with NVML
	- ROCm with ROCm SMI
	- Level Zero with Sysman

To build SYnergy samples, type:
```bash
cd SYnergy
mkdir build && cd build/
cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_BACKEND=[OpenSYCL | DPC++] -DSYNERGY_CUDA_SUPPORT=[ON | OFF] -DSYNERGY_ROCM_SUPPORT=[ON | OFF]
```

## Usage
To use SYnergy, just swap your current `sycl::queue` with `synergy::queue`. Under the `samples/` folder you can find an example of SYnergy usage.
