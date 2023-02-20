#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "vendor_implementations.hpp"

namespace synergy {
class runtime {
public:
  static runtime& get_instance();

  std::shared_ptr<device> assign_device(const sycl::device&);

  runtime(runtime const&) = delete;
  runtime(runtime&&) = delete;
  runtime& operator=(runtime const&) = delete;
  runtime& operator=(runtime&&) = delete;

private:
  runtime();
  std::unordered_map<sycl::device, std::shared_ptr<device>> devices;
};

} // namespace synergy
