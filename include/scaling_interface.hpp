#ifndef SYNERGY_SCALING_INTERFACE_H
#define SYNERGY_SCALING_INTERFACE_H

#include "types.h"
#include <vector>

namespace synergy {

class scaling_interface {
public:
  virtual std::vector<frequency> query_supported_frequencies() = 0;
  virtual std::vector<frequency> query_supported_core_frequencies(frequency memory_frequency) = 0;

  virtual void set_device_frequency(frequency_preset memory_frequency, frequency_preset core_frequency) = 0;
  virtual void set_device_frequency(frequency memory_frequency, frequency core_frequency) = 0;

  virtual ~scaling_interface() = default;

protected:
  static constexpr int max_clocks = 512;
};

} // namespace synergy

#endif