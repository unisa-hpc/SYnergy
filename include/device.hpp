#pragma once

#include "management_wrapper.hpp"
#include "types.hpp"

namespace synergy {

class device {
public:
  virtual std::vector<frequency> supported_core_frequencies() = 0;

  virtual std::vector<frequency> supported_uncore_frequencies() = 0;

  virtual frequency get_core_frequency() = 0;

  virtual frequency get_uncore_frequency() = 0;

  virtual void set_core_frequency(frequency target) = 0;

  virtual void set_uncore_frequency(frequency target) = 0;

  virtual void set_all_frequencies(frequency core, frequency uncore) = 0;

  virtual power get_power_usage() = 0;

  virtual unsigned get_power_sampling_rate() = 0;

  inline double get_energy_consumption() { return energy; }
  inline void increase_energy_consumption(double energy_increase) { energy += energy_increase; }

private:
  double energy = 0.0;
};

} // namespace synergy
