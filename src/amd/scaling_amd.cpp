#include "../../include/amd/scaling_amd.hpp"
#include "../../include/amd/utils.hpp"
#include <rocm_smi/rocm_smi.h>

namespace synergy {

scaling_amd::scaling_amd()
{
  synergy_check_rsmi(rsmi_init(0));

  rsmi_frequencies_t memory, core;
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_get(device_handle, RSMI_CLK_TYPE_MEM, &memory));
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_get(device_handle, RSMI_CLK_TYPE_SYS, &core));

  default_memory_clock = current_memory_clock = memory.frequency[memory.current];
  default_core_clock = current_core_clock = core.frequency[core.current];

  min_memory_clock = memory.frequency[0];
  min_core_clock = core.frequency[0];

  max_memory_clock = memory.frequency[memory.num_supported - 1];
  max_core_clock = core.frequency[core.num_supported - 1];

  for (int i = 0; i < memory.num_supported; i++) {
    memory_clocks.insert(std::pair(memory.frequency[i], i));
  }
  for (int i = 0; i < core.num_supported; i++) {
    core_clocks.insert(std::pair(core.frequency[i], i));
  }
}

std::vector<frequency> scaling_amd::get_supported_memory_frequencies()
{
  std::vector<frequency> mem_freq;

  for (const std::pair<frequency, uint32_t> &pair : memory_clocks) {
    mem_freq.push_back(pair.first);
  }

  mem_freq.shrink_to_fit();
  return mem_freq;
}

std::vector<frequency> scaling_amd::get_supported_core_frequencies()
{
  std::vector<frequency> core_freq;

  for (const std::pair<frequency, uint32_t> &pair : core_clocks) {
    core_freq.push_back(pair.first);
  }

  core_freq.shrink_to_fit();
  return core_freq;
}

void scaling_amd::set_memory_frequency(frequency freq)
{
  uint32_t mem_clock_index = memory_clocks.at(freq);

  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_MEM, make_bitmask(memory_clocks.size(), mem_clock_index)));

  current_memory_clock = freq;
  frequency_has_changed = true;
}

void scaling_amd::set_core_frequency(frequency freq)
{
  uint32_t core_clock_index = core_clocks.at(freq);

  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_SYS, make_bitmask(core_clocks.size(), core_clock_index)));

  current_core_clock = freq;
  frequency_has_changed = true;
}

void scaling_amd::set_device_frequency(frequency memory_frequency, frequency core_frequency)
{
  uint32_t mem_clock_index = memory_clocks.at(memory_frequency);
  uint32_t core_clock_index = core_clocks.at(core_frequency);

  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_MEM, make_bitmask(memory_clocks.size(), mem_clock_index)));
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_SYS, make_bitmask(core_clocks.size(), core_clock_index)));

  current_memory_clock = memory_frequency;
  current_core_clock = core_frequency;
  frequency_has_changed = true;
}

scaling_amd::~scaling_amd()
{
  if (frequency_has_changed)
    synergy_check_rsmi(rsmi_dev_gpu_reset(device_handle));
}

uint64_t scaling_amd::make_bitmask(uint32_t supported_clocks, uint32_t clock_index)
{
  uint64_t freq_bitmask = 1UL;
  uint32_t shift_amount = supported_clocks + (supported_clocks - 1) - clock_index;
  freq_bitmask <<= (64 - shift_amount);
  return freq_bitmask;
}

} // namespace synergy