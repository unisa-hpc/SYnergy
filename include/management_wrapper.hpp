#pragma once

#include <vector>

#include "types.h"

namespace synergy {

template <typename vendor>
class management_wrapper {

public:
  void initialize();
  void shutdown();

  unsigned int get_devices_count();
  vendor::device_handle get_device_handle(vendor::device_identifier);

  power get_power_usage(vendor::device_handle);

  // sorted in ascending order
  std::vector<frequency> get_supported_core_frequencies(vendor::device_handle);
  std::vector<frequency> get_supported_uncore_frequencies(vendor::device_handle);

  // TODO: use templates to define only one get_frequency, and discriminate using template parameters
  frequency get_core_frequency(vendor::device_handle);
  frequency get_uncore_frequency(vendor::device_handle);

  void set_core_frequency(vendor::device_handle, frequency);
  void set_uncore_frequency(vendor::device_handle, frequency);
  void set_all_frequencies(vendor::device_handle, frequency core, frequency uncore);

  void setup_profiling(vendor::device_handle);
  void setup_scaling(vendor::device_handle);
};

} // namespace synergy