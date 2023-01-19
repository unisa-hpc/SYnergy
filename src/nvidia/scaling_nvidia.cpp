#include "../../include/nvidia/scaling_nvidia.hpp"
#include "../../include/nvidia/utils.hpp"
#include <array>
#include <nvml.h>

namespace synergy {

scaling_nvidia::scaling_nvidia()
{
  synergy_check_nvml(nvmlInit());

  synergy_check_nvml(nvmlDeviceGetHandleByIndex(0, &device_handle));
  synergy_check_nvml(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_MEM, &default_memory_clock));
  synergy_check_nvml(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_GRAPHICS, &default_core_clock));

  current_memory_clock = default_memory_clock;
  current_core_clock = default_core_clock;

  std::array<uint32_t, max_clocks> memory_clocks;
  uint32_t count_memory_clocks;
  synergy_check_nvml(nvmlDeviceGetSupportedMemoryClocks(device_handle, &count_memory_clocks, memory_clocks.data()));

  max_memory_clock = memory_clocks[0];
  min_memory_clock = memory_clocks[count_memory_clocks - 1];
}

std::vector<frequency> scaling_nvidia::memory_frequencies()
{
  std::array<uint32_t, max_clocks> memory_clocks;
  uint32_t count_memory_clocks;
  synergy_check_nvml(nvmlDeviceGetSupportedMemoryClocks(device_handle, &count_memory_clocks, memory_clocks.data()));

  std::vector<frequency> freq(count_memory_clocks);
  for (int i = 0; i < count_memory_clocks; i++)
    freq[i] = static_cast<frequency>(memory_clocks[i]);

  return freq;
}

std::vector<frequency> scaling_nvidia::core_frequencies(frequency memory_frequency)
{
  uint32_t mem_freq = static_cast<uint32_t>(memory_frequency);
  std::array<uint32_t, max_clocks> core_clocks;
  uint32_t count_core_clocks;

  synergy_check_nvml(nvmlDeviceGetSupportedGraphicsClocks(device_handle, mem_freq, &count_core_clocks, core_clocks.data()));

  std::vector<frequency> freq(count_core_clocks);
  for (int i = 0; i < count_core_clocks; i++)
    freq[i] = static_cast<frequency>(core_clocks[i]);

  return freq;
}

void scaling_nvidia::change_frequency(frequency_preset memory_frequency, frequency_preset core_frequency)
{
  if (memory_frequency == frequency_preset::default_frequency && core_frequency == frequency_preset::default_frequency) {
    current_memory_clock = default_memory_clock;
    current_core_clock = default_core_clock;

    synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, current_core_clock));

    return;
  }

  switch (memory_frequency) {
  case frequency_preset::min_frequency:
    current_memory_clock = min_memory_clock;
    break;
  case frequency_preset::max_frequency:
    current_memory_clock = max_memory_clock;
    break;
  default:
    current_memory_clock = default_memory_clock;
  }

  if (core_frequency != frequency_preset::default_frequency) {
    std::array<uint32_t, max_clocks> core_clocks;
    uint32_t count_core_clocks;

    synergy_check_nvml(nvmlDeviceGetSupportedGraphicsClocks(device_handle, current_memory_clock, &count_core_clocks, core_clocks.data()));

    current_core_clock = core_frequency == frequency_preset::min_frequency
                          ? core_clocks[count_core_clocks - 1]
                          : core_clocks[0];
  } else
    current_core_clock = default_core_clock;

  synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, current_core_clock));

  std::printf("default clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n\n", default_memory_clock, default_core_clock, current_memory_clock, current_core_clock);
}

void scaling_nvidia::change_frequency(frequency memory_frequency, frequency core_frequency)
{
  synergy_check_nvml(nvmlDeviceSetApplicationsClocks(device_handle, static_cast<uint32_t>(memory_frequency), static_cast<uint32_t>(core_frequency)));

  std::printf("default clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n\n", default_memory_clock, default_core_clock, current_memory_clock, current_core_clock);
}

scaling_nvidia::~scaling_nvidia()
{
  uint32_t curr_mem;
  uint32_t curr_core;

  synergy_check_nvml(nvmlDeviceGetClockInfo(device_handle, NVML_CLOCK_MEM, &curr_mem));
  synergy_check_nvml(nvmlDeviceGetClockInfo(device_handle, NVML_CLOCK_GRAPHICS, &curr_core));

  std::printf("\ndefault clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n", default_memory_clock, default_core_clock, curr_mem, curr_core);
  synergy_check_nvml(nvmlDeviceResetApplicationsClocks(device_handle));
}

void prepare_scaling()
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