#ifndef _SYNERGY_SCALING_NVIDIA_H
#define _SYNERGY_SCALING_NVIDIA_H

#include "../scaling_interface.hpp"
#include "utils.hpp"
#include <nvml.h>

namespace synergy {

class scaling_nvidia : public scaling_interface {
public:
  scaling_nvidia();

  std::vector<frequency> memory_frequencies();
  std::vector<frequency> core_frequencies(frequency memory_frequency);

  void change_frequency(frequency_preset memory_frequency, frequency_preset core_frequency);
  void change_frequency(frequency memory_frequency, frequency core_frequency);

  ~scaling_nvidia();

private:
  nvmlDevice_t device_handle;

  uint32_t default_memory_clock;
  uint32_t default_core_clock;

  uint32_t min_memory_clock;
  uint32_t max_memory_clock;

  uint32_t current_memory_clock;
  uint32_t current_core_clock;
};

} // namespace synergy

#endif