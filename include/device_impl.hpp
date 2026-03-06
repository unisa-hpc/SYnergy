#pragma once

#include "management_wrapper.hpp"
#include "types.hpp"
#include "log_msg.hpp"

#include <optional>
namespace synergy {

namespace detail {

class device_impl {

public:
  virtual ~device_impl() = default;

  virtual std::vector<frequency> supported_core_frequencies() = 0;

  virtual std::vector<frequency> supported_uncore_frequencies() = 0;

  virtual frequency get_core_frequency(bool cached = true) = 0;

  virtual frequency get_uncore_frequency(bool cached = true) = 0;

  virtual void set_core_frequency(frequency target) = 0;

  virtual void set_uncore_frequency(frequency target) = 0;

  virtual void set_all_frequencies(frequency core, frequency uncore) = 0;
  virtual temperature_t get_temperature() = 0;
  virtual power get_power_usage() = 0;

  virtual energy get_energy_usage() = 0;

  virtual unsigned get_power_sampling_rate() = 0;
};

template <typename vendor>
class vendor_device : public device_impl {

public:
    
  
  inline vendor_device(typename vendor::device_identifier id, std::optional<synergy::gpu_domain> domain) {
    // Check the GPU domain. Required only for GEOPM backend
    library.initialize();
    if constexpr (vendor_traits<vendor>::needs_gpu_domain) {

      if (!domain) {
        throw std::logic_error("error synergy: gpu_domain required for this vendor");
      }
      handle = library.get_device_handle(id, domain.value());
      synergy::log::synergy_log(synergy::log::LogLevel::Debug, "Vendor id: " + std::to_string(id) + " domain: " + std::to_string(static_cast<int>(domain.value())));
    }
    else{
      handle = library.get_device_handle(id); // when domain is not defined, i.e. NVML, Level Zero and ROCm-smi, domain.value() throw exeption.
    }
    /* 
      TODO: This initialzie is called by all the device that are root and sub device for both composite and flat mode.
      In geopm we still not handle composite mode where a device is composed by two or more devices. 
      The problem is that in composite mode there is not a single core_freq but we have to handle a single core freq for each tile. 
      Work in progresso to find a clean solution.
      For now we do not save current_core and uncore frequency in the initialize function.
    */
    // current_core_frequency = library.get_core_frequency(handle);
    // current_uncore_frequency = library.get_uncore_frequency(handle);
  }

  inline ~vendor_device() { library.shutdown(); }

  inline std::vector<frequency> supported_core_frequencies() { return library.get_supported_core_frequencies(handle); }

  inline std::vector<frequency> supported_uncore_frequencies() { return library.get_supported_uncore_frequencies(handle); }

  inline frequency get_core_frequency(bool cached = false) { return cached ? current_core_frequency : library.get_core_frequency(handle); }

  inline frequency get_uncore_frequency(bool cached = false) { return cached ? current_uncore_frequency : library.get_uncore_frequency(handle); }

  inline void set_core_frequency(frequency target) {
    library.set_core_frequency(handle, target);
    current_core_frequency = target;
    current_uncore_frequency = library.get_uncore_frequency(handle);
  }

  inline void set_uncore_frequency(frequency target) {
    library.set_uncore_frequency(handle, target);
    current_uncore_frequency = target;
    current_core_frequency = library.get_core_frequency(handle);
  }

  inline void set_all_frequencies(frequency core, frequency uncore) {
    library.set_all_frequencies(handle, core, uncore);
    current_core_frequency = core;
    current_uncore_frequency = uncore;
  }

  inline power get_power_usage() {
    return library.get_power_usage(handle);
  }

  inline energy get_energy_usage() {
    return library.get_energy_usage(handle);
  }

  inline unsigned get_power_sampling_rate() {
    return vendor::sampling_rate;
  }
  inline temperature_t get_temperature() {
    return library.get_temperature(handle);
  }

 
private:
  management_wrapper<vendor> library;
  typename vendor::device_handle handle;
  frequency current_core_frequency;
  frequency current_uncore_frequency;

 
  
};

} // namespace detail

} // namespace synergy
