#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"

namespace synergy {
struct kernel {
  kernel(sycl::event event) : event{event} {}

  sycl::event event;
  double energy = 0.0;
  frequency core_target_frequency;
  frequency memory_target_frequency;
};
} // namespace synergy
