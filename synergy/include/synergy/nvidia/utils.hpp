#ifndef _SYNERGY_NVIDIA_UTILS_H_
#define _SYNERGY_NVIDIA_UTILS_H_

#include <nvml.h>
#include <stdexcept>

namespace synergy
{
	namespace details
	{
		// Check NVML error
		inline void check_nvml_error(nvmlReturn_t err)
		{
			if (err != NVML_SUCCESS)
			{
				throw std::runtime_error("NVML error: " + std::to_string(err));
			}
		}

	}
}

#endif