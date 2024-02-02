#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <rocm_smi/rocm_smi.h>

#include "../management_wrapper.hpp"

namespace synergy {

namespace detail {
namespace management {

struct rsmi {
  static constexpr std::string_view name = "RSMI";
  static constexpr unsigned int max_frequencies = RSMI_MAX_NUM_FREQUENCIES;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = unsigned int;
  using return_type = rsmi_status_t;
  static constexpr rsmi_status_t return_success = RSMI_STATUS_SUCCESS;
};

} // namespace management

template <>
class management_wrapper<management::rsmi> {

public:
  inline unsigned int get_devices_count() const {
    unsigned int count = 0;
    check(rsmi_num_monitor_devices(&count));
    return count;
  }

  inline void initialize() const { check(rsmi_init(0)); }

  inline void shutdown() const {
    /*check(rsmi_shut_down());*/
  }

  using rsmi = management::rsmi;

  inline rsmi::device_handle get_device_handle(rsmi::device_identifier id) const { return id; }

  inline power get_power_usage(rsmi::device_handle handle) const {
    uint64_t power;
    check(rsmi_dev_power_ave_get(handle, 0, &power)); // microwatts
    return power;
  }

  inline energy get_energy_usage(rsmi::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(rsmi::name) + " wrapper error: get_energy_usage is not supported"};
  }

  inline std::vector<frequency> get_supported_core_frequencies(rsmi::device_handle handle) const {
    rsmi_frequencies_t core;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core));
    std::vector<frequency> frequencies(core.num_supported);
    for (size_t i = 0; i < core.num_supported; i++)
      frequencies[i] = core.frequency[i] / 1e6;

    return frequencies;
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(rsmi::device_handle handle) const {
    rsmi_frequencies_t uncore;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore));

    std::vector<frequency> frequencies(uncore.num_supported);
    for (size_t i = 0; i < uncore.num_supported; i++)
      frequencies[i] = uncore.frequency[i] / 1e6;

    return frequencies;
  }

  inline frequency get_core_frequency(rsmi::device_handle handle) const {
    rsmi_frequencies_t core;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core));
    return core.frequency[core.current] / 1e6;
  }

  inline frequency get_uncore_frequency(rsmi::device_handle handle) const {
    rsmi_frequencies_t uncore;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore));
    return uncore.frequency[uncore.current] / 1e6;
  }

  inline void set_core_frequency(rsmi::device_handle handle, frequency target) const {
    rsmi_frequencies_t core;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core));
    // the AMD freqs are in Hz while target are in MHz
    target *= 1e6;

    int target_index = -1;
    for (size_t i = 0; i < core.num_supported && target_index == -1; i++) {
      if (core.frequency[i] == target)
        target_index = i;
    }

    if (target_index == -1)
      return; // silent fail, must handle better

    check(rsmi_dev_gpu_clk_freq_set(handle, RSMI_CLK_TYPE_SYS, make_bitmask(target_index)));
  }

  inline void set_uncore_frequency(rsmi::device_handle handle, frequency target) const {
    rsmi_frequencies_t uncore;
    check(rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore));
    target *= 1e6;
    int target_index = -1;
    for (size_t i = 0; i < uncore.num_supported && target_index == -1; i++)
      if (uncore.frequency[i] == target)
        target_index = i;

    if (target_index == -1)
      return; // silent fail, must handle better check(RSMI_STATUS_NOT_SUPPORTED)

    if (uncore.num_supported == 1) // with only one freqeuncy available there is no need to change the frequency
      return;

    check(rsmi_dev_gpu_clk_freq_set(handle, RSMI_CLK_TYPE_MEM, make_bitmask(target_index)));
  }

  inline void set_all_frequencies(rsmi::device_handle handle, frequency core, frequency uncore) const {
    set_uncore_frequency(handle, uncore);
    set_core_frequency(handle, core);
  }

  inline void setup_profiling(rsmi::device_handle) const {}

  inline void setup_scaling(rsmi::device_handle) const {}

  inline std::string error_string(rsmi::return_type return_value) const {
    const char* error_string;
    rsmi_status_string(return_value, &error_string);
    return std::string{error_string};
  }

private:
  unsigned long make_bitmask(uint32_t desired_frequency_index) const {
    return 1UL << desired_frequency_index;
  }

private:
  error_checker<management::rsmi> check{*this};
};

} // namespace detail

} // namespace synergy
