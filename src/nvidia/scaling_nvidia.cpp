#include <array>
#include <nvml.h>

#include "nvidia/scaling_nvidia.hpp"
#include "nvidia/utils.hpp"

namespace synergy {

scaling_nvidia::scaling_nvidia()
{
  synergy_check_nvml(nvmlInit());

  synergy_check_nvml(nvmlDeviceGetHandleByIndex(0, &device_handle));
  synergy_check_nvml(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_MEM, &current_memory_clock));
  synergy_check_nvml(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_GRAPHICS, &current_core_clock));
}

std::vector<frequency> scaling_nvidia::get_supported_memory_frequencies()
{
  std::array<uint32_t, max_clocks> memory_clocks;
  uint32_t count_memory_clocks;
  synergy_check_nvml(nvmlDeviceGetSupportedMemoryClocks(device_handle, &count_memory_clocks, memory_clocks.data()));

  std::vector<frequency> freq(count_memory_clocks);
  for (int i = count_memory_clocks - 1, j = 0; i >= 0; i--, j++)
    freq[j] = static_cast<frequency>(memory_clocks[i]);

  return freq;
}

std::vector<frequency> scaling_nvidia::get_supported_core_frequencies()
{
  uint32_t mem_freq = static_cast<uint32_t>(current_memory_clock);
  std::array<uint32_t, max_clocks> core_clocks;
  uint32_t count_core_clocks;

  synergy_check_nvml(nvmlDeviceGetSupportedGraphicsClocks(device_handle, mem_freq, &count_core_clocks, core_clocks.data()));

  std::vector<frequency> freq(count_core_clocks);
  for (int i = count_core_clocks - 1, j = 0; i >= 0; i--, j++)
    freq[j] = static_cast<frequency>(core_clocks[i]);

  return freq;
}

void scaling_nvidia::set_memory_frequency(frequency freq)
{
  synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, static_cast<uint32_t>(freq), current_core_clock));
  frequency_has_changed = true;
}

void scaling_nvidia::set_core_frequency(frequency freq)
{
  synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, freq));
  frequency_has_changed = true;
}

void scaling_nvidia::set_device_frequency(frequency memory_frequency, frequency core_frequency)
{
  synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, static_cast<uint32_t>(memory_frequency), static_cast<uint32_t>(core_frequency)));
  frequency_has_changed = true;
}

scaling_nvidia::~scaling_nvidia()
{
  if (frequency_has_changed)
    synergy_check_nvml(nvmlDeviceResetApplicationsClocks(device_handle));
}

void scaling_nvidia::prepare_scaling()
{
  nvmlEnableState_t isRestricted;
  nvmlEnableState_t currentAutoboostState;
  nvmlEnableState_t defaultAutoboostState;

  // Enable Persistence Mode (required to disable Auto Boost)
  // synergy_check_nvml(nvmlDeviceSetPersistenceMode(device_handle, NVML_FEATURE_ENABLED));

  // Disable Autoboost
  // synergy_check_nvml(nvmlDeviceGetAPIRestriction(device_handle, NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS, &isRestricted));   // default: ENABLE
  // if(isRestricted == NVML_FEATURE_ENABLED) { // Disable Restricted Mode to disable Auto Boost for non-root users
  //     synergy_check_nvml(nvmlDeviceSetAPIRestriction(device_handle, NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS, NVML_FEATURE_DISABLED));
  // }

  // only on CC < Pascal
  // synergy_check_nvml(nvmlDeviceSetAutoBoostedClocksEnabled(device_handle, NVML_FEATURE_DISABLED)); //for non-root users
  // synergy_check_nvml(nvmlDeviceGetAutoBoostedClocksEnabled(device_handle, &currentAutoboostState, &defaultAutoboostState));

  // Get permission of setting application clocks
  // synergy_check_nvml(nvmlDeviceGetAPIRestriction(device_handle, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, &isRestricted));
  // if (isRestricted == NVML_FEATURE_ENABLED) {
  //     synergy_check_nvml(nvmlDeviceSetAPIRestriction(device_handle, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, NVML_FEATURE_DISABLED));
  // }
}

} // namespace synergy