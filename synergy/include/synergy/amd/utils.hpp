#ifndef _SYNERGY_AMD_UTILS_H_
#define _SYNERGY_AMD_UTILS_H_

#include <rocm_smi/rocm_smi.h>
#include <stdexcept>

namespace synergy
{
	namespace details
	{
		// Check NVML error
		inline void check_rsmi_error(rsmi_status_t err)
		{
			if (err != RSMI_STATUS_SUCCESS)
			{
				throw std::runtime_error("ROCm SMI error: " + std::to_string(err));
			}
		}

	}
}

#endif