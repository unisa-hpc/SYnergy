#ifndef _SYNERGY_NVIDIA_UTILS_H_
#define _SYNERGY_NVIDIA_UTILS_H_

#include <nvml.h>
#include <stdexcept>

#define synergyCheckNvmlError(f) synergy::details::_checkNvmlError(f, #f)

namespace synergy
{
	namespace details
	{
		// Should be used through macro function synergyCheckNvmlError
		inline void _checkNvmlError(nvmlReturn_t returnValue, const std::string& functionCall)
		{
			if (returnValue != NVML_SUCCESS)
			{
				throw std::runtime_error("NVML call \"" + functionCall + "\"\n\tfailed with return value: " + std::to_string(returnValue));
			}
		}

	}
	
}

#endif