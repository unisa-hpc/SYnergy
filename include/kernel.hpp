#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"

namespace synergy {

namespace detail {
struct kernel {

  kernel(sycl::event event, frequency core_target_frequency, frequency uncore_target_frequency)
      : event{event}, core_target_frequency{core_target_frequency}, uncore_target_frequency{uncore_target_frequency} {
    if (core_target_frequency != 0 && uncore_target_frequency != 0)
      has_target = true;
  }

  sycl::event event;
  double energy = 0.0;
  frequency core_target_frequency;
  frequency uncore_target_frequency;
  bool has_target = false;
};

} // namespace detail

} // namespace synergy
