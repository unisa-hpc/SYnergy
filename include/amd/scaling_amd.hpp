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

  std::vector<frequency> memory_frequencies();
  std::vector<frequency> core_frequencies(frequency memory_frequency);

  void change_frequency(frequency_preset memory_frequency, frequency_preset core_frequency);
  void change_frequency(frequency memory_frequency, frequency core_frequency);

  ~scaling_amd();

private:
  uint32_t device_handle;

  uint32_t default_memory_clock;
  uint32_t default_core_clock;

  std::map<frequency, uint32_t> memory_clocks;
  std::map<frequency, uint32_t> core_clocks;

  uint32_t current_memory_clock;
  uint32_t current_core_clock;
};

} // namespace synergy

#endif