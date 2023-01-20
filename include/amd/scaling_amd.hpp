#ifndef SYNERGY_SCALING_AMD_H
#define SYNERGY_SCALING_AMD_H

#include "../scaling_interface.hpp"
#include "utils.hpp"
#include <map>
#include <rocm_smi/rocm_smi.h>

namespace synergy {

class scaling_amd : public scaling_interface {
public:
  scaling_amd();

  std::vector<frequency> query_supported_frequencies();
  std::vector<frequency> query_supported_core_frequencies(frequency memory_frequency);

  void set_device_frequency(frequency_preset memory_frequency, frequency_preset core_frequency);
  void set_device_frequency(frequency memory_frequency, frequency core_frequency);

  ~scaling_amd();

private:
  uint32_t device_handle;

  std::map<frequency, uint32_t> memory_clocks;
  std::map<frequency, uint32_t> core_clocks;

  frequency current_memory_clock;
  frequency current_core_clock;

  frequency default_memory_clock;
  frequency default_core_clock;

  frequency min_memory_clock;
  frequency max_memory_clock;

  frequency min_memory_clock;
  frequency max_memory_clock;
};

} // namespace synergy

#endif