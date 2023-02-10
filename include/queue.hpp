#pragma once

#include <atomic>
#include <future>
#include <thread>
#include <unordered_set>

#include <sycl/sycl.hpp>

#include "profiling.hpp"
#include "runtime.hpp"
#include "utils.hpp"

namespace synergy {

class queue : public sycl::queue {
public:
  using base = sycl::queue;

  friend class profiler;

  template <typename... Args, std::enable_if_t<!::details::is_present_v<sycl::property_list, Args...> && !::details::has_property_v<Args...>, bool> = true>
  queue(Args&&... args)
      : base(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{})
  {
    if (!synergy::runtime::is_initialized)
      synergy::runtime::initialize();

    runtime& syn = runtime::get_instance();
  }

  template <typename... Args, std::enable_if_t<::details::is_present_v<sycl::property_list, Args...> || ::details::has_property_v<Args...>, bool> = true>
  queue(Args&&... args)
      : base(std::forward<Args>(args)...) {}

  template <typename... Args>
  inline sycl::event submit(Args&&... args)
  {
    auto&& event = sycl::queue::submit(std::forward<Args>(args)...);

    kernels_energy.insert({event, 0.0});
    std::async(std::launch::async, profiler{*this}, event);

    return event;
  }

  device<std::any> get_synergy_device() const;

  inline double energy_consumption() const { return energy; }
  double kernel_energy_consumption(sycl::event& event);

private:
  device<std::any> device;
  double energy = 0.0;
  std::unordered_map<sycl::event, double> kernels_energy;
};

} // namespace synergy
