#include <array>
#include <iostream>

#include <nvml.h>

#include "nvidia/scaling_nvidia.hpp"
#include "nvidia/utils.hpp"

using namespace std;

namespace synergy {

scaling_nvidia::scaling_nvidia()
{
  synergy_only_success(nvmlInit());
  synergy_only_success(nvmlDeviceGetHandleByIndex(0, &device_handle));

  prepare_scaling();
  synergy_only_success(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_MEM, &current_memory_clock));
  synergy_only_success(nvmlDeviceGetDefaultApplicationsClock(device_handle, NVML_CLOCK_GRAPHICS, &current_core_clock));
}

vector<frequency> scaling_nvidia::get_supported_memory_frequencies()
{
  array<uint32_t, max_clocks> memory_clocks;
  uint32_t count_memory_clocks;
  synergy_only_success(nvmlDeviceGetSupportedMemoryClocks(device_handle, &count_memory_clocks, memory_clocks.data()));

  vector<frequency> freq(count_memory_clocks);
  for (int i = count_memory_clocks - 1, j = 0; i >= 0; i--, j++)
    freq[j] = static_cast<frequency>(memory_clocks[i]);

  return freq;
}

vector<frequency> scaling_nvidia::get_supported_core_frequencies()
{
  uint32_t mem_freq = static_cast<uint32_t>(current_memory_clock);
  array<uint32_t, max_clocks> core_clocks;
  uint32_t count_core_clocks;

  synergy_only_success(nvmlDeviceGetSupportedGraphicsClocks(device_handle, mem_freq, &count_core_clocks, core_clocks.data()));

  vector<frequency> freq(count_core_clocks);
  for (int i = count_core_clocks - 1, j = 0; i >= 0; i--, j++)
    freq[j] = static_cast<frequency>(core_clocks[i]);

  return freq;
}

void scaling_nvidia::set_memory_frequency(frequency freq)
{
  if (synergy_notify_noroot(nvmlDeviceSetApplicationsClocks(device_handle, static_cast<uint32_t>(freq), current_core_clock)) == NVML_SUCCESS)
    frequency_has_changed = true;
}

void scaling_nvidia::set_core_frequency(frequency freq)
{
  if (synergy_notify_noroot(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, freq)))
    frequency_has_changed = true;
}

void scaling_nvidia::set_device_frequency(frequency memory_frequency, frequency core_frequency)
{
  if (synergy_notify_noroot(nvmlDeviceSetApplicationsClocks(device_handle, static_cast<uint32_t>(memory_frequency), static_cast<uint32_t>(core_frequency))))
    frequency_has_changed = true;
}

scaling_nvidia::~scaling_nvidia()
{
  if (frequency_has_changed)
    synergy_notify_noroot(nvmlDeviceResetApplicationsClocks(device_handle));
}

void scaling_nvidia::prepare_scaling()
{
  nvmlEnableState_t persistence_mode_state;
  synergy_only_success(nvmlDeviceGetPersistenceMode(device_handle, &persistence_mode_state));

  if (persistence_mode_state == NVML_FEATURE_DISABLED) // enable persistence mode (required to disable auto boost)
    synergy_notify_noroot(nvmlDeviceSetPersistenceMode(device_handle, NVML_FEATURE_ENABLED));

  // here we want to disable auto boost
  // persistence mode must be enabled, otherwise the call will fail
  // we check again because if we don't have root access, then persistence mode may be disabled
  synergy_only_success(nvmlDeviceGetPersistenceMode(device_handle, &persistence_mode_state));
  if (persistence_mode_state == NVML_FEATURE_DISABLED) // in this case we just don't set auto boost
    return;

  // here persistence mode is enabled
  nvmlDeviceArchitecture_t device_arch;
  synergy_only_success(nvmlDeviceGetArchitecture(device_handle, &device_arch));

  if (device_arch < NVML_DEVICE_ARCH_PASCAL)
    synergy_only_success(nvmlDeviceSetAutoBoostedClocksEnabled(device_handle, NVML_FEATURE_DISABLED));
}

} // namespace synergy