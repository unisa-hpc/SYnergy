#pragma once

#include "management_wrapper.hpp"
#include "types.hpp"
#include "vendor_device.hpp"

namespace synergy {

template <typename vendor>
class vendor_device : public device {

public:
  inline vendor_device(typename vendor::device_identifier id)
  {
    library.initialize();
    handle = library.get_device_handle(id);

    current_core_frequency = library.get_core_frequency(handle);
    current_uncore_frequency = library.get_uncore_frequency(handle);
  }

  inline ~vendor_device() { library.shutdown(); }

  inline virtual std::vector<frequency> supported_core_frequencies() { return library.get_supported_core_frequencies(handle); }

  inline virtual std::vector<frequency> supported_uncore_frequencies() { return library.get_supported_uncore_frequencies(handle); }

  inline virtual frequency get_core_frequency() { return current_core_frequency; }

  inline virtual frequency get_uncore_frequency() { return current_uncore_frequency; }

  inline virtual void set_core_frequency(frequency target)
  {
    library.set_core_frequency(handle, target);
    current_core_frequency = target;
  }

  inline virtual void set_uncore_frequency(frequency target)
  {
    library.set_uncore_frequency(handle, target);
    current_uncore_frequency = target;
  }

  inline virtual void set_all_frequencies(frequency core, frequency uncore)
  {
    library.set_all_frequencies(handle, core, uncore);
    current_core_frequency = core;
    current_uncore_frequency = uncore;
  }

  inline virtual power get_power_usage()
  {
    return library.get_power_usage(handle);
  }

  inline virtual unsigned get_power_sampling_rate()
  {
    return vendor::sampling_rate;
  }

private:
  management_wrapper<vendor> library;
  typename vendor::device_handle handle;
  frequency current_core_frequency;
  frequency current_uncore_frequency;
};

} // namespace synergy
