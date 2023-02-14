#pragma once

#include <future>
#include <thread>
#include <unordered_map>

#include <sycl/sycl.hpp>

#include "kernel.hpp"
#include "profiling.hpp"
#include "runtime.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace synergy {

class queue : public sycl::queue {
public:
  using base = sycl::queue;

  template <typename... Args, std::enable_if_t<!::details::is_present_v<sycl::property_list, Args...> && !::details::has_property_v<Args...>, bool> = true>
  queue(Args&&... args)
      : base(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{})
  {
    if (!synergy::runtime::is_initialized())
      synergy::runtime::initialize();

    runtime& syn = runtime::get_instance();
    device = syn.assign_device(get_device());
  }

  template <typename... Args, std::enable_if_t<::details::is_present_v<sycl::property_list, Args...> || ::details::has_property_v<Args...>, bool> = true>
  queue(Args&&... args)
      : base(std::forward<Args>(args)...)
  {
    if (!synergy::runtime::is_initialized())
      synergy::runtime::initialize();

    runtime& syn = runtime::get_instance();
    device = syn.assign_device(get_device());
  }

  template <typename... Args>
  inline sycl::event submit(Args&&... args)
  {
    sycl::event event = sycl::queue::submit(std::forward<Args>(args)...);
    kernel k{event};

    auto async = std::async(std::launch::async, profiler(kernels.insert({event, k}).first->second, device));

    return event;
  }

  inline std::shared_ptr<device> get_synergy_device() const { return device; }

  inline double kernel_energy_consumption(sycl::event& event) const
  {
    auto search = kernels.find(event);
    if (search == kernels.end())
      throw std::runtime_error("synergy::queue error: kernel was not submitted to the queue");

    return search->second.energy;
  }

private:
  std::shared_ptr<device> device;
  std::unordered_map<sycl::event, kernel> kernels;
};

} // namespace synergy
