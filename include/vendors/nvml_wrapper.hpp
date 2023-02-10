#pragma once

#include <array>

#include <nvml.h>

#include "../management_wrapper.hpp"

namespace synergy {

namespace management {
struct nvml {
  static constexpr unsigned int max_frequencies = 128;
  static constexpr unsigned int sampling_rate = 15; // ms
  using device_identifier = unsigned int;
  using device_handle = nvmlDevice_t;
};

} // namespace management

template <>
class management_wrapper<management::nvml> {

public:
  inline unsigned int get_devices_count()
  {
    unsigned int count = 0;
    nvmlDeviceGetCount(&count);

    return count;
  }

  inline void initialize() { nvmlInit(); }

  inline void shutdown() { nvmlShutdown(); }

  using nvml = management::nvml;

  inline nvml::device_handle get_device_handle(nvml::device_identifier id)
  {
    nvml::device_handle handle;
    nvmlDeviceGetHandleByIndex(id, &handle);
    return handle;
  }

  inline power get_power_usage(nvml::device_handle handle)
  {
    unsigned int power;
    nvmlDeviceGetPowerUsage(handle, &power); // milliwatts
    return power * 1000;                     // return microwatts
  }

  inline std::vector<frequency> get_supported_core_frequencies(nvml::device_handle handle)
  {
    using namespace std;

    unsigned int current_uncore_frequency = get_uncore_frequency(handle);
    array<unsigned int, nvml::max_frequencies> core_frequencies;
    unsigned int count_core_frequencies;

    nvmlDeviceGetSupportedGraphicsClocks(handle, current_uncore_frequency, &count_core_frequencies, core_frequencies.data());

    vector<frequency> frequencies(count_core_frequencies);
    for (int i = count_core_frequencies - 1, j = 0; i >= 0; i--, j++) // enforce non-decrescent order
      frequencies[j] = core_frequencies[i];

    return frequencies;
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(nvml::device_handle handle)
  {
    using namespace std;

    array<unsigned int, nvml::max_frequencies> memory_frequencies;
    unsigned int count_uncore_frequencies;

    nvmlDeviceGetSupportedMemoryClocks(handle, &count_uncore_frequencies, memory_frequencies.data());

    vector<frequency> frequencies(count_uncore_frequencies);
    for (int i = count_uncore_frequencies - 1, j = 0; i >= 0; i--, j++) // enforce non-decrescent order
      frequencies[j] = memory_frequencies[i];

    return frequencies;
  }

  inline frequency get_core_frequency(nvml::device_handle handle)
  {
    unsigned int frequency;
    nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_GRAPHICS, &frequency);
    return frequency;
  }

  inline frequency get_uncore_frequency(nvml::device_handle handle)
  {
    unsigned int frequency;
    nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_MEM, &frequency);
    return frequency;
  }

  inline void set_core_frequency(nvml::device_handle handle, frequency target)
  {
    unsigned int uncore_frequency = get_uncore_frequency(handle);
    nvmlDeviceSetApplicationsClocks(handle, uncore_frequency, target);
  }

  inline void set_uncore_frequency(nvml::device_handle handle, frequency target)
  {
    unsigned int core_frequency = get_core_frequency(handle);
    nvmlDeviceSetApplicationsClocks(handle, target, core_frequency);
  }

  inline void set_all_frequencies(nvml::device_handle handle, frequency core, frequency uncore)
  {
    nvmlDeviceSetApplicationsClocks(handle, uncore, core);
  }

  inline void setup_profiling(nvml::device_handle) {}

  inline void setup_scaling(nvml::device_handle handle)
  {
    nvmlDeviceArchitecture_t device_arch;
    nvmlDeviceGetArchitecture(handle, &device_arch);

    if (device_arch < NVML_DEVICE_ARCH_PASCAL) { // we need to disable Auto Boost
      nvmlEnableState_t persistence_mode_state;

      nvmlDeviceGetPersistenceMode(handle, &persistence_mode_state);
      if (persistence_mode_state == NVML_FEATURE_DISABLED)
        nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_ENABLED); // requires root access

      nvmlDeviceSetAutoBoostedClocksEnabled(handle, NVML_FEATURE_DISABLED);
    }
  }
};

} // namespace synergy
