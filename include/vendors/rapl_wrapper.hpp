#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <istream>
#include <unistd.h>

#include "../management_wrapper.hpp"

namespace synergy {
namespace detail {
namespace management {
struct rapl {
  static constexpr std::string_view name = "Intel-RAPL";
  static constexpr unsigned int max_frequencies = 256;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = unsigned int;
  using return_type = unsigned int;
  static constexpr return_type return_success = 0;
};
}; // namespace management

template <>
class management_wrapper<management::rapl> {

public:
  using rapl = management::rapl;

  inline unsigned int get_devices_count() const {
    return 1;
  }

  inline void initialize() {
    do_root();
  }

  inline void shutdown() {
    undo_root();
  }

  inline rapl::device_handle get_device_handle(rapl::device_identifier id) const {
    return id;
  }

  inline power get_power_usage(rapl::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: get_power_usage is not supported"};
  }

  inline energy get_energy_usage(rapl::device_handle handle) const {
    unsigned long long int energy;
    std::string path {"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:" + std::to_string(handle) + "/energy_uj"};
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("synergy " + std::string{rapl::name} + "wrapper error: could not open MSR file. Are you root?");
    }
    file >> energy;
    if (file.fail()) {
      throw std::runtime_error("synergy " + std::string{rapl::name} + "wrapper error: could not read MSR file");
    }
    
    return static_cast<synergy::energy>(energy); // microjouls
  }

  inline std::vector<frequency> get_supported_core_frequencies(const rapl::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: get_supported_core_frequencies is not supported"};
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(const rapl::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: get_supported_uncore_frequencies is not supported"};
  }

  inline frequency get_core_frequency(const rapl::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: get_core_frequency is not supported"};
  }

  inline frequency get_uncore_frequency(const rapl::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: get_uncore_frequency is not supported"};
  }

  inline void set_core_frequency(const rapl::device_handle handle, frequency target) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: set_core_frequency is not supported"};
  }

  inline void set_uncore_frequency(const rapl::device_handle handle, frequency target) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: set_uncore_frequency is not supported"};
  }

  inline void set_all_frequencies(rapl::device_handle handle, frequency core, frequency uncore) const {
    set_core_frequency(handle, core);
    set_uncore_frequency(handle, uncore);
  }

  inline void setup_profiling(rapl::device_handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: setup_profiling is not supported"};
  }

  inline void setup_scaling(rapl::device_handle) const {
    throw std::runtime_error{"synergy " + std::string(rapl::name) + " wrapper error: setup_scaling is not supported"};
  }

  inline std::string error_string(rapl::return_type return_value) const {
    
  }

private:
  error_checker<management::rapl> check{*this};

  void do_root() {
    setreuid(0, 0);
  }

  void undo_root() {
    setreuid(geteuid(), getuid());
  }
};

}; // namespace detail
}; // namespace synergy
