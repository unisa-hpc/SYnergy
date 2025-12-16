# SYnergy
SYCL library for energy measurement and frequency scaling.
SYnergy allows to get standard power-related features such as per-application and per-kernel energy measurements as well as frequency scaling capabilities, all with minimal configuration. 
Currently supported target architectures: 
- NVIDIA GPUs supported through the [NVML library](https://developer.nvidia.com/nvidia-management-library-nvml).
- AMD GPUs supported through the [ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib)
- Intel GPUs through the [Sysman API](https://spec.oneapi.io/level-zero/latest/sysman/PROG.html)
SYnergy integrates also GEOPM library to enable energy profiling and frequency scaling.
## Build
### Dependencies
- cmake (3.17 or newer)  
- C++17 or newer compiler
- A supported SYCL implementation:
	- DPC++
	- AdaptiveCPP
- A supported target architecture
	- CUDA with NVML
	- ROCm with ROCm SMI
	- Level Zero with Sysman
	- GEOPM for frequency scaling and energy measurements on clusters. 

To build SYnergy samples, type:
```bash
cd SYnergy
mkdir build && cd build/

# Build SYnergy with CUDA backend
cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_CUDA_SUPPORT=ON
# Build SYnergy with ROCm backend
cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_ROCM_SUPPORT=ON
# Build SYnergy with Level Zero backend
cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_LZ_SUPPORT=ON
# Build SYnergy with GEOPM backend
cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=DPC++ -DSYNERGY_GEOPM_SUPPORT=ON -DSYNERGY_DEVICE_PROFILING=ON -DSYNERGY_HOST_PROFILING=ON -DSYNERGY_KERNEL_PROFILING=ON

make -j 
```


## Usage
To use SYnergy, just swap your current `sycl::queue` with `synergy::queue`. Under the `samples/` folder you can find an example of SYnergy usage.

## SYnergy Interface 
### SYnergy Queue
The `synergy::queue` class extends the `sycl::queue` class with energy measurement and frequency scaling capabilities. 
To create a SYnergy queue, just instantiate a `synergy::queue` object instead of a `sycl::queue` object. 
```cpp
#include <synergy/synergy.hpp>
synergy::queue q(sycl::gpu_selector_v);
```
### Device Energy Measurement
To measure the energy consumption of a device, use the `get_device_energy()` method of the `synergy::queue` class. This method returns the energy consumption in Joules starting from the `synergy::queue` construction.
```cpp
synergy::queue q(sycl::gpu_selector_v);
q.submit([&](sycl::handler &h) {
	// Kernel code here
}).wait();
double energy = q.get_device_energy(); // Energy in Joules consumed by the device since the queue creation
```

### Kernel Energy Measurement
To measure the energy consumption of a kernel, use the `kernel_energy_consumption(sycl::event&)` method of the `synergy::queue` class. This method takes a `sycl::event` object as input and returns the energy consumption in Joules for the kernel associated with the event.
```cpp
synergy::queue q(sycl::gpu_selector_v);
sycl::event e = q.submit([&](sycl::handler &h) {
	// Kernel code here
});
e.wait();
double kernel_energy = q.kernel_energy_consumption(e); // Energy in Joules consumed by the kernel
```
### Coarse-grained frequency scaling
To set the frequency of the device, use the `q.get_synergy_device().set_core_frequency(unsigned core_freq);` method of the `synergy::queue` class. This method takes the desired frequency in MHz as input. Or you can construct the SYnergy queue with a specific frequency using the constructor `synergy::queue(unsigned mem_freq, unsigned core_freq, sycl::device_selector selector)`.
```cpp	
synergy::queue q(sycl::gpu_selector_v);
q.get_synergy_device().set_core_frequency(1200); // Set device frequency to 1200 MHz
```
or
```cpp
 // Create a SYnergy queue with device core frequency set to 1200 MHz and memory frequency default (0)
synergy::queue q(0, 1200, sycl::gpu_selector_v);
```

### Fine-grained frequency scaling
To change the frequency of the device before the kernel execution, SYnergy allows to specify the memory and core frequency as additional parameters of the `submit` function. 
```cpp
synergy::queue q(sycl::gpu_selector_v);
q.submit(0, 800, [&](sycl::handler &h) { // Set core frequency to 800 MHz and memory frequency to default
	// Kernel code here
}).wait();
```	
