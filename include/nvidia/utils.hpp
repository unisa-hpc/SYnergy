#ifndef _SYNERGY_NVIDIA_UTILS_H_
#define _SYNERGY_NVIDIA_UTILS_H_

#include <nvml.h>

#include <stdexcept>

#define synergy_check_nvml(f) synergy::details::_check_nvml(f, #f)

namespace synergy {
namespace details {

// Should be used through macro function synergy_check_nvml
inline void _check_nvml(nvmlReturn_t return_value, const std::string &function_call)
{
  if (return_value != NVML_SUCCESS) {
    throw std::runtime_error("NVML call \"" + function_call + "\"\n\tfailed with error: " + nvmlErrorString(return_value));
  }
}

} // namespace details

} // namespace synergy

#endif