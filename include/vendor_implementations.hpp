#pragma once

#ifdef SYNERGY_CUDA_SUPPORT
#include "vendors/nvml_wrapper.hpp"
#endif

#ifdef SYNERGY_ROCM_SUPPORT
#include "vendors/rsmi_wrapper.hpp"
#endif

#ifdef SYNERGY_LZ_SUPPORT
#include "vendors/lz_wrapper.hpp"
#endif