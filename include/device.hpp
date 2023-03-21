#pragma once

#include <memory>

#include "device_impl.hpp"
#include "types.hpp"

namespace synergy {

class device {
public:
  device() = default;
  device(std::shared_ptr<detail::device_impl> impl) : impl{impl} {}

  inline std::vector<frequency> supported_core_frequencies() { return impl->supported_core_frequencies(); }

  inline std::vector<frequency> supported_uncore_frequencies() { return impl->supported_uncore_frequencies(); }

  inline frequency get_core_frequency(bool cached = true) { return impl->get_core_frequency(cached); }

  inline frequency get_uncore_frequency(bool cached = true) { return impl->get_uncore_frequency(cached); }

  inline void set_core_frequency(frequency target) { impl->set_core_frequency(target); }

  inline void set_uncore_frequency(frequency target) { impl->set_uncore_frequency(target); }

  inline void set_all_frequencies(frequency core, frequency uncore) { impl->set_all_frequencies(core, uncore); }

  inline power get_power_usage() { return impl->get_power_usage(); }

  inline unsigned get_power_sampling_rate() { return impl->get_power_sampling_rate(); }

private:
  std::shared_ptr<detail::device_impl> impl;
};

} // namespace synergy
