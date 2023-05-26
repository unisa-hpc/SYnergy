#pragma once

#include <ze_api.h>
#include <zes_api.h>

#include <array>

#include "../management_wrapper.hpp"
#include <string>
#include <ctime>

#include <iostream>


namespace synergy {
namespace detail {
namespace management {
  struct lz {
    static constexpr std::string_view name = "LZ";
    static constexpr unsigned int max_frequencies = 256;
    static constexpr unsigned int sampling_rate = 5; // ms
    using device_identifier = unsigned int;
    using device_handle = zes_device_handle_t;
    using return_type = ze_result_t;
    static constexpr return_type return_success = ZE_RESULT_SUCCESS;
  };
}; // namespace management

template<>
class management_wrapper<management::lz> {

public:
  inline unsigned int get_devices_count() const {
    return devices.size();
  }

  inline void initialize() { //ho dovuto togliere const
    check(zeInit(0)); 
    init_drivers();
    init_devices();
  }

  inline void shutdown() {} //ho dovuto togliere const

  using lz = management::lz;

  inline lz::device_handle get_device_handle(lz::device_identifier id) const {
    auto ret = (lz::device_handle) devices[id];
    return ret;
  }

  inline power get_power_usage(lz::device_handle handle) const {
    const unsigned SAMPLING_RATEO = 100000; // to increase precision increase this value

    zes_pwr_handle_t hPwr;
    check(zesDeviceGetCardPowerDomain(handle, &hPwr));

    zes_power_energy_counter_t counter1;
    zes_power_energy_counter_t counter2;
    
    zesPowerGetEnergyCounter(hPwr, &counter1);

    do {
      zesPowerGetEnergyCounter(hPwr, &counter2);
    } while (counter2.timestamp - counter1.timestamp < SAMPLING_RATEO);

    float energy = counter2.energy - counter1.energy;
    float timestamp = counter2.timestamp - counter1.timestamp;
    return (energy / timestamp) * 1000000; // watt to microwatt
  }

  inline std::vector<frequency> get_supported_core_frequencies(const lz::device_handle handle) const {
    return get_supported_frequency<ZES_FREQ_DOMAIN_GPU>(handle);
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(const lz::device_handle handle) const {
    return get_supported_frequency<ZES_FREQ_DOMAIN_MEMORY>(handle);
  }

  inline frequency get_core_frequency(const lz::device_handle handle) const  {
    return get_frequency<ZES_FREQ_DOMAIN_GPU>(handle);
  }

  inline frequency get_uncore_frequency(const lz::device_handle handle) const  {
    return get_frequency<ZES_FREQ_DOMAIN_MEMORY>(handle);
  }

  inline void set_core_frequency(const lz::device_handle handle, frequency target) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_GPU>(handle);
    check(zesFrequencyOcSetFrequencyTarget(h_freq, target));
  }
  
  inline void set_uncore_frequency(const lz::device_handle handle, frequency target) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_MEMORY>(handle);
    check(zesFrequencyOcSetFrequencyTarget(h_freq, target));
  }

  inline void set_all_frequencies(lz::device_handle handle, frequency core, frequency uncore) const {
    set_core_frequency(handle, core);
    set_core_frequency(handle, uncore);
  }

  inline void setup_profiling(lz::device_handle) const {}

  inline void setup_scaling(lz::device_handle) const {}

  inline std::string error_string(lz::return_type return_value) const {
    switch (return_value) {
      case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
      case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        return "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
      case ZE_RESULT_ERROR_DEVICE_LOST:
        return "ZE_RESULT_ERROR_DEVICE_LOST";
      case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        return "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
      case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
      case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
      case ZE_RESULT_ERROR_NOT_AVAILABLE:
        return "ZE_RESULT_ERROR_NOT_AVAILABLE";
      case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
      case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
      case ZE_RESULT_ERROR_UNKNOWN:
        return "ZE_RESULT_ERROR_UNKNOWN";
      case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
      case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
      case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
      case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
      case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
      case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
      default:
        return "[Code error] " + std::to_string(return_value);
    }
  }

private:
  error_checker<management::lz> check{*this};
  std::vector<ze_driver_handle_t> drivers;
  std::vector<ze_device_handle_t> devices;

  inline void init_drivers() {
    unsigned int drivers_count = 0;
    if (zeDriverGet(&drivers_count, nullptr) == management::lz::return_success && drivers_count) {
      drivers.resize(drivers_count);
      zeDriverGet(&drivers_count, drivers.data());
    }
  }

  inline void init_devices() {
    unsigned int devices_count = 0;
    unsigned int tmp = 0;
    for (unsigned i = 0; i < drivers.size(); i++) {
      zeDeviceGet(drivers[i], &tmp, nullptr);
      devices_count += tmp;
    }
  
    devices.resize(devices_count);
    for (unsigned i = 0, offset = 0; i < devices_count; i++) {
      zeDeviceGet(drivers[i], &tmp, &devices.data()[offset]);
      offset += tmp;
    }
  }

  template<zes_freq_domain_t domain>
  zes_freq_handle_t get_frequency_handle(const management::lz::device_handle handle) const {
    zes_freq_handle_t ret = nullptr;
    unsigned handles_count = 0;
    check(zesDeviceEnumFrequencyDomains(handle, &handles_count, nullptr));
    if (handles_count) {
      std::vector<zes_freq_handle_t> handles (handles_count);
      zesDeviceEnumFrequencyDomains(handle, &handles_count, handles.data());
      for (int i = 0; i < handles_count; i++) {
        zes_freq_properties_t props {};
        props.stype = ZES_STRUCTURE_TYPE_FREQ_PROPERTIES;
        if (zesFrequencyGetProperties(handles[i], &props) == lz::return_success) {
          if (props.type == domain) {
            ret = handles[i];
            break;
          }
        }
      }
    }
    return ret;
  }

  template<zes_freq_domain_t domain>
  std::vector<frequency> get_supported_frequency(const lz::device_handle handle) const {
    std::vector<frequency> freqs;
    zes_freq_handle_t h_freq = get_frequency_handle<domain>(handle);
    if (h_freq != nullptr) {
      unsigned count = 0;
      check(zesFrequencyGetAvailableClocks(h_freq, &count, nullptr));
      if (count) {
        std::vector<double> freq_arr (count);
        check(zesFrequencyGetAvailableClocks(h_freq, &count, freq_arr.data()));
        for (int i = 0; i < count; i++) {
          freqs.push_back(static_cast<frequency>(freq_arr[i]));
        }
      }
    }
    return freqs;
  }

  template<zes_freq_domain_t domain>
  inline frequency get_frequency(const lz::device_handle handle) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_GPU>(handle);
    zes_freq_state_t state {};
    state.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;

    check(zesFrequencyGetState(h_freq, &state));
    return state.actual;
  }
};

}; // namespace detail
}; // namespace synergy