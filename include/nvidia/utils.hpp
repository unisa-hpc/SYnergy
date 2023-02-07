#ifndef _SYNERGY_NVIDIA_UTILS_H_
#define _SYNERGY_NVIDIA_UTILS_H_

#include <iostream>
#include <stdexcept>

#include <nvml.h>

#define synergy_only_success(f) synergy::details::_check_nvml(f, #f)
#define synergy_notify_noroot(f) synergy::details::_check_nvml_except_root(f, #f)

namespace synergy {
namespace details {

// Should be used through macro function synergy_only_success
inline nvmlReturn_t _check_nvml(nvmlReturn_t return_value, const std::string &function_call)
{
  if (return_value != NVML_SUCCESS) {
    throw std::runtime_error("NVML call \"" + function_call + "\"\n\tfailed with error: " + nvmlErrorString(return_value));
  }
  return return_value;
}

inline nvmlReturn_t _check_nvml_except_root(nvmlReturn_t return_value, const std::string &function_call)
{
  if (return_value != NVML_SUCCESS && return_value != NVML_ERROR_NO_PERMISSION) {
    throw std::runtime_error("NVML call \"" + function_call + "\"\n\tfailed with error: " + nvmlErrorString(return_value));
  }
  if (return_value == NVML_ERROR_NO_PERMISSION) {
    std::cerr << "Application does not have root permission. NVML call \"" << function_call << "\" failed. Some features may not be available.\n";
  }

  return return_value;
}

} // namespace details

} // namespace synergy

#endif