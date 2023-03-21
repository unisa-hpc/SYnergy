#pragma once

#include <sycl/sycl.hpp>

namespace synergy {

namespace detail {
struct kernel {
  kernel(sycl::event event) : event{event} {}

  sycl::event event;
  double energy = 0.0;
};

inline bool operator==(const kernel& lhs, const kernel& rhs) { return lhs.event == rhs.event; }

} // namespace detail

} // namespace synergy
