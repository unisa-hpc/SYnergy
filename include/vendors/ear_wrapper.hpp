#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <iostream>
#include <cmath>

#include <ear/PlatformIO.hpp>
#include <ear/PlatformTopo.hpp>

#include "../management_wrapper.hpp"

namespace synergy {

namespace detail {

namespace management {
struct ear {
  static constexpr std::string_view name = "EAR";
  static constexpr unsigned int max_frequencies = 256;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = unsigned int;
  using return_type = unsigned int;
  static constexpr return_type return_success = 0;
};

} // namespace management

template <>
class management_wrapper<management::ear> {

public:
  inline unsigned int get_devices_count() const {
    // TODO
  }

  inline void initialize() const { 
    // TODO
  }

  inline void shutdown() const {
    // TODO
  }

  using ear = management::ear;

  inline ear::device_handle get_device_handle(ear::device_identifier id) const {
    return id;
  }

  inline power get_power_usage(ear::device_handle handle) const {
    // TODO
  }

  inline energy get_energy_usage(ear::device_handle handle) const {
    // TODO
  }

  inline std::vector<frequency> get_supported_core_frequencies(ear::device_handle handle) const {
    // TODO
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(ear::device_handle handle) const {
    // TODO
  }

  inline frequency get_core_frequency(ear::device_handle handle) const {
    // TODO
  }

  inline frequency get_uncore_frequency(ear::device_handle handle) const {
    // TODO
  }

  inline void set_core_frequency(ear::device_handle handle, frequency target) const {
    // TODO
  }

  inline void set_uncore_frequency(ear::device_handle handle, frequency target) const {
    // TODO
  }

  inline void set_all_frequencies(ear::device_handle handle, frequency core, frequency uncore) const {
    // TODO
  }

  inline void setup_profiling(ear::device_handle) const {
    // TODO
  }

  inline void setup_scaling(ear::device_handle handle) const {
    // TODO
  }

  inline std::string error_string(ear::return_type return_value) const {
    // TODO
  }

private:
  error_checker<management::ear> check{*this};
};

} // namespace detail

} // namespace synergy
