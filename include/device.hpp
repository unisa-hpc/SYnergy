#pragma once

#include "management_wrapper.hpp"
#include "types.h"

namespace synergy {

// TODO: synergy::device need some kind of 1:n relation to sycl::device
template <typename vendor>
class device {

public:
  inline device(vendor::device_identifier id)
  {
    library.initialize();
    handle = library.get_device_handle(id);

    current_core_frequency = library.get_core_frequency();
    current_uncore_frequency = library.get_uncore_frequency();
  }

  inline ~device() { library.shutdown(); }

  inline std::vector<frequency> supported_core_frequencies() { return library.get_supported_core_frequencies(handle); }

  inline std::vector<frequency> supported_uncore_frequencies() { return library.get_supported_uncore_frequencies(handle); }

  inline frequency get_core_frequency() { return current_core_frequency; }

  inline frequency get_uncore_frequency() { return current_uncore_frequency; }

  inline void set_core_frequency(frequency target)
  {
    library.set_core_frequency(handle, target);
    current_core_frequency = target;
  }

  inline void set_uncore_frequency(frequency target)
  {
    library.set_uncore_frequency(handle, target);
    current_uncore_frequency = target;
  }

  inline void set_all_frequencies(frequency core, frequency uncore)
  {
    library.set_all_frequencies(handle, core, uncore);
    current_core_frequency = core;
    current_uncore_frequency = uncore;
  }

  inline power get_power_usage()
  {
    return library.get_power_usage(handle);
  }

private:
  management_wrapper<vendor> library;
  vendor::device_handle handle;
  frequency current_core_frequency;
  frequency current_uncore_frequency;
};

} // namespace synergy
