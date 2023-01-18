#ifndef SYNERGY_AMD_UTILS_H
#define SYNERGY_AMD_UTILS_H

#include <rocm_smi/rocm_smi.h>
#include <stdexcept>

#define synergy_check_rsmi(f) synergy::details::_check_rsmi(f, #f)

namespace synergy {
namespace details {

// Should be used through macro function synergy_check_rsmi
inline void _check_rsmi(rsmi_status_t return_value, const std::string &function_call)
{
  if (return_value != RSMI_STATUS_SUCCESS) {
    throw std::runtime_error("ROCm SMI call \"" + function_call + "\"\n\tfailed with return value: " + std::to_string(return_value));
  }
}

} // namespace details
} // namespace synergy

#endif