#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <iostream>
#include <cmath>

#include <geopm/PlatformIO.hpp>
#include <geopm/PlatformTopo.hpp>

#include "../management_wrapper.hpp"

namespace g = geopm;

namespace synergy {

namespace detail {

namespace management {
struct geopm {
  static constexpr std::string_view name = "GEOPM";
  static constexpr unsigned int max_frequencies = 256;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = unsigned int;
  using return_type = unsigned int;
  static constexpr return_type return_success = 0;
};

} // namespace management

template <>
class management_wrapper<management::geopm> {

public:
  inline unsigned int get_devices_count() const {
    return g::platform_topo().num_domain(GEOPM_DOMAIN_GPU);
  }

  inline void initialize() const { }

  inline void shutdown() const { }

  using geopm = management::geopm;

  inline geopm::device_handle get_device_handle(geopm::device_identifier id) const {
    return id;
  }

  inline power get_power_usage(geopm::device_handle handle) const {
    unsigned int power = g::platform_io().read_signal("GPU_POWER", GEOPM_DOMAIN_GPU, handle);
    return power * 1e6; // from W to uW
  }

  inline energy get_energy_usage(geopm::device_handle handle) const {
    unsigned int energy = g::platform_io().read_signal("GPU_ENERGY", GEOPM_DOMAIN_GPU, handle);
    return energy * 1e6; // from J to uJ
  }

  inline std::vector<frequency> get_supported_core_frequencies(geopm::device_handle handle) const {
    auto min = g::platform_io().read_signal("GPU_CORE_FREQUENCY_MIN_AVAIL", GEOPM_DOMAIN_GPU, handle);
    auto max = g::platform_io().read_signal("GPU_CORE_FREQUENCY_MAX_AVAIL", GEOPM_DOMAIN_GPU, handle);
    auto step = g::platform_io().read_signal("GPU_CORE_FREQUENCY_STEP", GEOPM_DOMAIN_GPU, handle);

    std::vector<frequency> frequencies;
    for (auto i = min; i <= max; i += step) {
      frequency freq = std::round(i * 1e-6);
      frequencies.push_back(freq);
    }
    return frequencies;
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(geopm::device_handle handle) const {
    return {}; // TODO: we need this, but it is not supported by GEOPM
  }

  inline frequency get_core_frequency(geopm::device_handle handle) const {
    return g::platform_io().read_signal("GPU_CORE_FREQUENCY_STATUS", GEOPM_DOMAIN_GPU, handle) * 1e-6;
  }

  inline frequency get_uncore_frequency(geopm::device_handle handle) const {
    return 0; // TODO: we need this, but it is not supported by GEOPM
  }

  inline void set_core_frequency(geopm::device_handle handle, frequency target) const {
    g::platform_io().write_control("GPU_CORE_FREQUENCY_MIN_CONTROL", GEOPM_DOMAIN_GPU, handle, target * 1e6);
    g::platform_io().write_control("GPU_CORE_FREQUENCY_MAX_CONTROL", GEOPM_DOMAIN_GPU, handle, target * 1e6);
  }

  inline void set_uncore_frequency(geopm::device_handle handle, frequency target) const {
    throw std::runtime_error{"synergy " + std::string(geopm::name) + " wrapper error: set_uncore_frequency is not supported"};
  }

  inline void set_all_frequencies(geopm::device_handle handle, frequency core, frequency uncore) const {
    set_core_frequency(handle, core);
    std::cerr << "synergy " << geopm::name << " wrapper warning: set_all_frequencies does not support uncore frequency" << std::endl;
  }

  inline void setup_profiling(geopm::device_handle) const {}

  inline void setup_scaling(geopm::device_handle handle) const {}

  inline std::string error_string(geopm::return_type return_value) const {
    return std::string{""};
  }

private:
  error_checker<management::geopm> check{*this};
};

} // namespace detail

} // namespace synergy
