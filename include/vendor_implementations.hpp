#ifdef SYNERGY_CUDA_SUPPORT
#include "nvidia/profiling_nvidia.hpp"
#include "nvidia/scaling_nvidia.hpp"
#endif

#ifdef SYNERGY_ROCM_SUPPORT
#include "amd/profiling_amd.hpp"
#include "amd/scaling_amd.hpp"
#endif