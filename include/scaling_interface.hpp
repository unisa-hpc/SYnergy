#ifndef SYNERGY_SCALING_INTERFACE_H
#define SYNERGY_SCALING_INTERFACE_H

#include "types.h"
#include <vector>

namespace synergy {

class scaling_interface {
public:
  virtual std::vector<frequency> get_supported_memory_frequencies() = 0;

  // Supported core frequencies may or may not depend on current memory frequency
  virtual std::vector<frequency> get_supported_core_frequencies() = 0;

  virtual void set_memory_frequency(frequency) = 0;
  virtual void set_core_frequency(frequency) = 0;
  virtual void set_device_frequency(frequency memory_frequency, frequency core_frequency) = 0;

  virtual ~scaling_interface() = default;

protected:
  static constexpr int max_clocks = 512;
};

} // namespace synergy

#endif