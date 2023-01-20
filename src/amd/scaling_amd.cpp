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

  for (int i = 0; i < memory.num_supported; i++) {
    memory_clocks.insert(std::pair(memory.frequency[i], i));
  }
  for (int i = 0; i < core.num_supported; i++) {
    core_clocks.insert(std::pair(core.frequency[i], i));
  }
}

std::vector<frequency> scaling_amd::query_supported_frequencies()
{
  std::vector<frequency> mem_freq;

  for (const std::pair<frequency, uint32_t> &pair : memory_clocks) {
    mem_freq.push_back(pair.first);
  }

  mem_freq.shrink_to_fit();
  return mem_freq;
}

std::vector<frequency> scaling_amd::query_supported_core_frequencies(frequency memory_frequency)
{
  std::vector<frequency> core_freq;

  for (const std::pair<frequency, uint32_t> &pair : core_clocks) {
    core_freq.push_back(pair.first);
  }

  core_freq.shrink_to_fit();
  return core_freq;
}

void scaling_amd::set_device_frequency(frequency_preset memory_frequency, frequency_preset core_frequency)
{
  rsmi_frequencies_t memory, core;
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_get(device_handle, RSMI_CLK_TYPE_MEM, &memory));
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_get(device_handle, RSMI_CLK_TYPE_SYS, &core));
}

void scaling_amd::set_device_frequency(frequency memory_frequency, frequency core_frequency)
{
  uint32_t mem_index = memory_clocks.at(memory_frequency);
  uint32_t core_index = core_clocks.at(core_frequency);

  current_memory_clock = memory_frequency;
  current_core_clock = core_frequency;

  uint64_t mem_freq_bitmask = 1UL;
  uint32_t shift_amt = memory_clocks.size() + (memory_clocks.size() - 1) - mem_index;
  mem_freq_bitmask <<= (64 - shift_amt);
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_MEM, mem_freq_bitmask));

  uint64_t core_freq_bitmask = 1UL;
  shift_amt = core_clocks.size() + (core_clocks.size() - 1) - core_index;
  core_freq_bitmask <<= (64 - shift_amt);
  synergy_check_rsmi(rsmi_dev_gpu_clk_freq_set(device_handle, RSMI_CLK_TYPE_SYS, core_freq_bitmask));
}

scaling_amd::~scaling_amd()
{
  synergy_check_rsmi(rsmi_dev_gpu_reset(device_handle));
}

} // namespace synergy