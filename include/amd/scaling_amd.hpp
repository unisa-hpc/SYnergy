#ifndef _SYNERGY_SCALING_AMD_H
#define _SYNERGY_SCALING_AMD_H

#include "../scaling_interface.hpp"
#include "utils.hpp"
#include <rocm_smi/rocm_smi.h>

namespace synergy {

class scaling_amd : public scaling_interface {
public:
  scaling_amd()
  {
    initialize_device_info();
  }

  void change_frequency(frequency_preset memory_frequency, frequency_preset core_frequency)
  {
    // if (memory_frequency == frequency_preset::default_frequency && core_frequency == frequency_preset::default_frequency) {
    //   current_memory_clock = default_memory_clock;
    //   current_core_clock = default_core_clock;

    //   synergy_check_rsmi(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, current_core_clock));

    //   return;
    // }

    // switch (memory_frequency) {
    // case frequency_preset::min_frequency:
    //   current_memory_clock = min_memory_clock;
    //   break;
    // case frequency_preset::max_frequency:
    //   current_memory_clock = max_memory_clock;
    //   break;
    // default:
    //   current_memory_clock = default_memory_clock;
    // }

    // if (core_frequency != frequency_preset::default_frequency) {
    //   std::array<uint32_t, 128> core_clocks;
    //   synergy_check_rsmi(nvmlDeviceGetSupportedGraphicsClocks(device_handle, current_memory_clock, &count_core_clocks, core_clocks.data()));

    //   current_core_clock = core_frequency == frequency_preset::min_frequency
    //                         ? core_clocks[count_core_clocks - 1]
    //                         : core_clocks[0];
    // } else
    //   current_core_clock = default_core_clock;

    // synergy_check_rsmi(nvmlDeviceSetApplicationsClocks(device_handle, current_memory_clock, current_core_clock));

    // std::printf("default clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n\n", default_memory_clock, default_core_clock, current_memory_clock, current_core_clock);
  }

  void change_frequency(uint32_t memory_frequency, uint32_t core_frequency)
  {
    // synergy_check_rsmi(nvmlDeviceSetApplicationsClocks(device_handle, memory_frequency, core_frequency));

    // std::printf("default clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n\n", default_memory_clock, default_core_clock, current_memory_clock, current_core_clock);
  }

  ~scaling_amd()
  {
    uint32_t curr_mem;
    uint32_t curr_core;

    synergy_check_rsmi(nvmlDeviceGetClockInfo(device_handle, NVML_CLOCK_MEM, &curr_mem));
    synergy_check_rsmi(nvmlDeviceGetClockInfo(device_handle, NVML_CLOCK_GRAPHICS, &curr_core));

    std::printf("\ndefault clocks:mem-%dMHz\t core-%dMHz,\tCurrent clocks:mem-%dMHz\t core-%dMHz\n", default_memory_clock, default_core_clock, curr_mem, curr_core);
    synergy_check_rsmi(nvmlDeviceResetApplicationsClocks(device_handle));
  }

private:
  uint32_t device_handle;

  uint32_t count_memory_clocks;
  uint32_t count_core_clocks;

  uint32_t default_memory_clock;
  uint32_t default_core_clock;

  uint32_t min_memory_clock;
  uint32_t max_memory_clock;

  uint32_t current_memory_clock;
  uint32_t current_core_clock;

  void initialize_device_info()
  {
    synergy_check_rsmi(rsmi_init(0));

    synergy_check_rsmi(rsmi_dev_gpu_clk_freq_get(device_handle, RSMI_CLK_TYPE_MEM, )
  }

  void prepare_scaling()
  {
  }
};

} // namespace synergy

#endif