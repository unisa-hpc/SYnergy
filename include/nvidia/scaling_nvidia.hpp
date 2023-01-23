#ifndef _SYNERGY_SCALING_NVIDIA_H
#define _SYNERGY_SCALING_NVIDIA_H

#include "../scaling_interface.hpp"
#include "utils.hpp"
#include <nvml.h>

namespace synergy {

class scaling_nvidia : public scaling_interface {
public:
  scaling_nvidia();

  std::vector<frequency> get_supported_memory_frequencies();
  std::vector<frequency> get_supported_core_frequencies();

  void set_memory_frequency(frequency);
  void set_core_frequency(frequency);
  void set_device_frequency(frequency memory_frequency, frequency core_frequency);

  ~scaling_nvidia();

private:
  nvmlDevice_t device_handle;
  bool frequency_has_changed = false;

  uint32_t current_memory_clock;
  uint32_t current_core_clock;

  void prepare_scaling();
};

} // namespace synergy

#endif