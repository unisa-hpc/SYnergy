#pragma once

#include <rocm_smi/rocm_smi.h>

#include "../management_wrapper.hpp"

namespace synergy {

namespace management {

struct rsmi {
  static constexpr unsigned int max_frequencies = RSMI_MAX_NUM_FREQUENCIES;
  static constexpr unsigned int sampling_rate = 15; // ms
  using device_identifier = unsigned int;
  using device_handle = unsigned int;
};

} // namespace management

template <>
class management_wrapper<management::rsmi> {

public:
  inline unsigned int get_devices_count()
  {
    unsigned int count = 0;
    rsmi_num_monitor_devices(&count);
    return count;
  }

  inline void initialize() { rsmi_init(0); }

  inline void shutdown()
  {
    /*rsmi_shut_down();*/
  }

  using rsmi = management::rsmi;

  inline rsmi::device_handle get_device_handle(rsmi::device_identifier id) { return id; }

  inline power get_power_usage(rsmi::device_handle handle)
  {
    uint64_t power;
    rsmi_dev_power_ave_get(handle, 0, &power); // microwatts
    return power;
  }

  inline std::vector<frequency> get_supported_core_frequencies(rsmi::device_handle handle)
  {
    rsmi_frequencies_t core;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core);

    std::vector<frequency> frequencies(core.num_supported);
    for (int i = 0; i < core.num_supported; i++)
      frequencies[i] = core.frequency[i];

    return frequencies;
  }
  inline std::vector<frequency> get_supported_uncore_frequencies(rsmi::device_handle handle)
  {
    rsmi_frequencies_t uncore;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore);

    std::vector<frequency> frequencies(uncore.num_supported);
    for (int i = 0; i < uncore.num_supported; i++)
      frequencies[i] = uncore.frequency[i];

    return frequencies;
  }

  inline frequency get_core_frequency(rsmi::device_handle handle)
  {
    rsmi_frequencies_t core;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core);
    return core.frequency[core.current];
  }

  inline frequency get_uncore_frequency(rsmi::device_handle handle)
  {
    rsmi_frequencies_t uncore;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore);
    return uncore.frequency[uncore.current];
  }

  inline void set_core_frequency(rsmi::device_handle handle, frequency target)
  {
    rsmi_frequencies_t core;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_SYS, &core);

    size_t target_index = -1;
    for (size_t i = 0; i < core.num_supported && target_index == -1; i++)
      if (core.frequency[i] == target)
        target_index = i;

    if (target_index == -1)
      return; // silent fail, must handle better

    rsmi_dev_gpu_clk_freq_set(handle, RSMI_CLK_TYPE_SYS, make_bitmask(core.num_supported, target_index));
  }

  inline void set_uncore_frequency(rsmi::device_handle handle, frequency target)
  {
    rsmi_frequencies_t uncore;
    rsmi_dev_gpu_clk_freq_get(handle, RSMI_CLK_TYPE_MEM, &uncore);

    size_t target_index = -1;
    for (size_t i = 0; i < uncore.num_supported && target_index == -1; i++)
      if (uncore.frequency[i] == target)
        target_index = i;

    if (target_index == -1)
      return; // silent fail, must handle better

    rsmi_dev_gpu_clk_freq_set(handle, RSMI_CLK_TYPE_MEM, make_bitmask(uncore.num_supported, target_index));
  }

  inline void set_all_frequencies(rsmi::device_handle handle, frequency core, frequency uncore)
  {
    set_uncore_frequency(handle, uncore);
    set_core_frequency(handle, core);
  }

  inline void setup_profiling(rsmi::device_handle) {}

  inline void setup_scaling(rsmi::device_handle) {}

private:
  unsigned long make_bitmask(uint32_t num_supported_clocks, uint32_t desired_frequency_index)
  {
    uint64_t freq_bitmask = 1UL;
    uint32_t shift_amount = num_supported_clocks +
                            (num_supported_clocks - 1) - desired_frequency_index;
    freq_bitmask <<= (64 - shift_amount);
    return freq_bitmask;
  }
};

} // namespace synergy
