#pragma once

#include <vector>

#include "types.h"

namespace synergy {

template <typename library>
class management_wrapper {

public:
  void initialize();
  void shutdown();

  unsigned int get_devices_count();
  library::device_handle get_device_handle(library::device_identifier);

  power get_power_usage(library::device_handle);

  // sorted in ascending order
  std::vector<frequency> get_supported_core_frequencies(library::device_handle);
  std::vector<frequency> get_supported_uncore_frequencies(library::device_handle);

  frequency get_core_frequency(library::device_handle);
  frequency get_uncore_frequency(library::device_handle);

  void set_core_frequency(library::device_handle, frequency);
  void set_uncore_frequency(library::device_handle, frequency);
  void set_all_frequencies(library::device_handle, frequency core, frequency uncore);

  void setup_profiling(library::device_handle);
  void setup_scaling(library::device_handle);
};

} // namespace synergy