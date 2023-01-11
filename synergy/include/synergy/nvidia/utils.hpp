#ifndef _SYNERGY_NVIDIA_UTILS_H_
#define _SYNERGY_NVIDIA_UTILS_H_

#include <nvml.h>
#include <stdexcept>

#define synergy_check_nvml(f) synergy::details::_check_nvml(f, #f)

namespace synergy {
namespace details {

// Should be used through macro function synergy_check_nvml
inline void _check_nvml(nvmlReturn_t returnValue, const std::string &functionCall)
{
  if (returnValue != NVML_SUCCESS) {
    throw std::runtime_error("NVML call \"" + functionCall + "\"\n\tfailed with return value: " + std::to_string(returnValue));
  }
}

} // namespace details

} // namespace synergy

#endif