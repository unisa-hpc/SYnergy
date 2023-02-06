#ifndef SYNERGY_AMD_UTILS_H
#define SYNERGY_AMD_UTILS_H

#include <stdexcept>

#include <rocm_smi/rocm_smi.h>

#define synergy_check_rsmi(f) synergy::details::_check_rsmi(f, #f)

namespace synergy {
namespace details {

// Should be used through macro function synergy_check_rsmi
inline void _check_rsmi(rsmi_status_t return_value, const std::string &function_call)
{
  if (return_value != RSMI_STATUS_SUCCESS) {
    char *error_string;
    rsmi_status_string(return_value, &error_string);

    throw std::runtime_error("ROCm SMI call \"" + function_call + "\"\n\tfailed with error: " + error_string);
  }
}

} // namespace details
} // namespace synergy

#endif