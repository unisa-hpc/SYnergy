#ifndef SYNERGY_SCALING_AMD_H
#define SYNERGY_SCALING_AMD_H

#include <map>

#include <rocm_smi/rocm_smi.h>

#include "../scaling_interface.hpp"
#include "utils.hpp"

namespace synergy {

class scaling_amd : public scaling_interface {
public:
  scaling_amd();

  std::vector<frequency> get_supported_memory_frequencies();
  std::vector<frequency> get_supported_core_frequencies();

  void set_memory_frequency(frequency);
  void set_core_frequency(frequency);
  void set_device_frequency(frequency memory_frequency, frequency core_frequency);

  ~scaling_amd();

private:
  uint32_t device_handle;
  bool frequency_has_changed = false;

  std::map<uint64_t, uint32_t> memory_clocks;
  std::map<uint64_t, uint32_t> core_clocks;

  uint64_t current_memory_clock;
  uint64_t current_core_clock;

  uint64_t make_bitmask(uint32_t supported_clocks, uint32_t clock_index);
};

} // namespace synergy

#endif