#pragma once

#include <array>
#include <string_view>

#include <nvml.h>

#include "../management_wrapper.hpp"

namespace synergy {

namespace detail {
namespace management {
struct nvml {
  static constexpr std::string_view name = "NVML";
  static constexpr unsigned int max_frequencies = 256;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = nvmlDevice_t;
  using return_type = nvmlReturn_t;
  static constexpr nvmlReturn_t return_success = NVML_SUCCESS;
};

} // namespace management

template <>
class management_wrapper<management::nvml> {

public:
  inline unsigned int get_devices_count() const {
    unsigned int count = 0;
    check(nvmlDeviceGetCount(&count));

    return count;
  }

  inline void initialize() const { check(nvmlInit()); }

  inline void shutdown() const { check(nvmlShutdown()); }

  using nvml = management::nvml;

  inline nvml::device_handle get_device_handle(nvml::device_identifier id) const {
    nvml::device_handle handle;
    check(nvmlDeviceGetHandleByIndex(id, &handle));
    return handle;
  }

  inline power get_power_usage(nvml::device_handle handle) const {
    unsigned int power;
    check(nvmlDeviceGetPowerUsage(handle, &power)); // milliwatts
    return power * 1000;                            // return microwatts
  }

  inline std::vector<frequency> get_supported_core_frequencies(nvml::device_handle handle) const {
    using namespace std;

    unsigned int current_uncore_frequency = get_uncore_frequency(handle);
    array<unsigned int, nvml::max_frequencies> core_frequencies;
    unsigned int count_core_frequencies;

    check(nvmlDeviceGetSupportedGraphicsClocks(handle, current_uncore_frequency, &count_core_frequencies, core_frequencies.data()));

    vector<frequency> frequencies(count_core_frequencies);
    for (int i = count_core_frequencies - 1, j = 0; i >= 0; i--, j++) // enforce non-decrescent order
      frequencies[j] = core_frequencies[i];

    return frequencies;
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(nvml::device_handle handle) const {
    using namespace std;

    array<unsigned int, nvml::max_frequencies> memory_frequencies;
    unsigned int count_uncore_frequencies;

    check(nvmlDeviceGetSupportedMemoryClocks(handle, &count_uncore_frequencies, memory_frequencies.data()));

    vector<frequency> frequencies(count_uncore_frequencies);
    for (int i = count_uncore_frequencies - 1, j = 0; i >= 0; i--, j++) // enforce non-decrescent order
      frequencies[j] = memory_frequencies[i];

    return frequencies;
  }

  inline frequency get_core_frequency(nvml::device_handle handle) const {
    unsigned int frequency;
    check(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_GRAPHICS, &frequency));
    return frequency;
  }

  inline frequency get_uncore_frequency(nvml::device_handle handle) const {
    unsigned int frequency;
    check(nvmlDeviceGetApplicationsClock(handle, NVML_CLOCK_MEM, &frequency));
    return frequency;
  }

  inline void set_core_frequency(nvml::device_handle handle, frequency target) const {
    unsigned int uncore_frequency = get_uncore_frequency(handle);
    check(nvmlDeviceSetApplicationsClocks(handle, uncore_frequency, target));
  }

  inline void set_uncore_frequency(nvml::device_handle handle, frequency target) const {
    std::array<unsigned int, nvml::max_frequencies> core_frequencies;
    unsigned int count_core_frequencies;
    check(nvmlDeviceGetSupportedGraphicsClocks(handle, target, &count_core_frequencies, core_frequencies.data()));

    check(nvmlDeviceSetApplicationsClocks(handle, target, core_frequencies[0])); // put highest core frequency
  }

  inline void set_all_frequencies(nvml::device_handle handle, frequency core, frequency uncore) const {
    check(nvmlDeviceSetApplicationsClocks(handle, uncore, core));
  }

  inline void setup_profiling(nvml::device_handle) const {}

  inline void setup_scaling(nvml::device_handle handle) const {
    nvmlDeviceArchitecture_t device_arch;
    check(nvmlDeviceGetArchitecture(handle, &device_arch));

    if (device_arch < NVML_DEVICE_ARCH_PASCAL) { // we need to disable Auto Boost
      nvmlEnableState_t persistence_mode_state;

      check(nvmlDeviceGetPersistenceMode(handle, &persistence_mode_state));
      if (persistence_mode_state == NVML_FEATURE_DISABLED)
        check(nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_ENABLED)); // requires root access

      nvmlEnableState_t auto_boost_enabled, auto_boost_enabled_default;
      check(nvmlDeviceGetAutoBoostedClocksEnabled(handle, &auto_boost_enabled, &auto_boost_enabled_default));

      if (auto_boost_enabled == NVML_FEATURE_ENABLED)
        check(nvmlDeviceSetAutoBoostedClocksEnabled(handle, NVML_FEATURE_DISABLED));
    }
  }

  inline std::string error_string(nvml::return_type return_value) const {
    return std::string{nvmlErrorString(return_value)};
  }

private:
  error_checker<management::nvml> check{*this};
};

} // namespace detail

} // namespace synergy
