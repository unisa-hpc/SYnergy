#pragma once

#include <vector>

#include "types.hpp"

namespace synergy {

template <typename vendor>
class management_wrapper {

  typedef typename vendor::device_handle device_handle;
  typedef typename vendor::device_identifier device_indentifier;

public:
  void initialize();
  void shutdown();

  unsigned int get_devices_count();
  device_handle get_device_handle(device_indentifier);

  power get_power_usage(device_handle);

  // sorted in ascending order
  std::vector<frequency> get_supported_core_frequencies(device_handle);
  std::vector<frequency> get_supported_uncore_frequencies(device_handle);

  // TODO: use templates to define only one get_frequency, and discriminate using template parameters
  frequency get_core_frequency(device_handle);
  frequency get_uncore_frequency(device_handle);

  void set_core_frequency(device_handle, frequency);
  void set_uncore_frequency(device_handle, frequency);
  void set_all_frequencies(device_handle, frequency core, frequency uncore);

  void setup_profiling(device_handle);
  void setup_scaling(device_handle);
};

} // namespace synergy