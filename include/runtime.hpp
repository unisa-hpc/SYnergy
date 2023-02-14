#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "vendor_implementations.hpp"

namespace synergy {
class runtime {
public:
  static void initialize();
  static runtime& get_instance();
  inline static bool is_initialized() { return instance != nullptr; }

  std::shared_ptr<device> assign_device(const sycl::device&);

private:
  runtime();
  runtime(const runtime&) = delete;
  runtime(runtime&&) = delete;

  static std::unique_ptr<runtime> instance;
  std::unordered_map<sycl::device, std::shared_ptr<device>> devices;
};

} // namespace synergy
