#pragma once

#include <iostream>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "types.hpp"

namespace synergy {
namespace detail {

class device_scaling {
public:
  device_scaling(device& device, const frequency core_frequency, const frequency uncore_frequency)
      : device{device}, core_target_frequency{core_frequency}, uncore_target_frequency{uncore_frequency} {}

  void operator()(sycl::handler& h) {
    try {
      device.set_all_frequencies(core_target_frequency, uncore_target_frequency);
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }
  }

private:
  device& device;
  const frequency core_target_frequency;
  const frequency uncore_target_frequency;
};

} // namespace detail

} // namespace synergy
