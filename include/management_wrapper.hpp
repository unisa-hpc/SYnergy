#pragma once

#include <vector>

#include "types.hpp"

namespace synergy {

namespace detail {

template <typename vendor>
class management_wrapper {

  typedef typename vendor::device_handle device_handle;
  typedef typename vendor::device_identifier device_indentifier;
  typedef typename vendor::return_type return_type;

public:
  void initialize() const;
  void shutdown() const;

  unsigned int get_devices_count() const;
  device_handle get_device_handle(device_indentifier) const;

  power get_power_usage(device_handle) const;

  // sorted in ascending order
  std::vector<frequency> get_supported_core_frequencies(device_handle);
  std::vector<frequency> get_supported_uncore_frequencies(device_handle);

  // TODO: use templates to define only one get_frequency, and discriminate using template parameters
  frequency get_core_frequency(device_handle) const;
  frequency get_uncore_frequency(device_handle) const;

  void set_core_frequency(device_handle, frequency) const;
  void set_uncore_frequency(device_handle, frequency) const;
  void set_all_frequencies(device_handle, frequency core, frequency uncore) const;

  void setup_profiling(device_handle) const;
  void setup_scaling(device_handle) const;

  std::string error_string(return_type) const;
};

template <typename vendor>
class error_checker {
  typedef typename vendor::return_type return_type;

public:
  error_checker(const management_wrapper<vendor>& w) : lib{w} {}

  void operator()(return_type return_value) const
  {
    if (return_value != vendor::return_success)
      throw std::runtime_error{"synergy " + std::string(vendor::name) + " wrapper error: " + lib.error_string(return_value)};
  }

private:
  const management_wrapper<vendor>& lib;
};

} // namespace detail

} // namespace synergy