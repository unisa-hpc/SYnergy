#pragma once

#include <future>
#include <thread>
#include <unordered_map>

#include <sycl/sycl.hpp>

#include "kernel.hpp"
#include "profiling.hpp"
#include "runtime.hpp"
#include "types.hpp"

namespace synergy {

class queue : public sycl::queue {
public:
  template <typename... Args>
  queue(Args&&... args)
  {
    if constexpr ((std::is_same_v<sycl::property::queue::enable_profiling, Args> || ...)) {
      sycl::queue(std::forward<Args>(args)...);
    } else {
      sycl::queue(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{});
    }

    if (!synergy::runtime::is_initialized())
      synergy::runtime::initialize();

    runtime& syn = runtime::get_instance();
    device = syn.assign_device(get_device());
  }

  template <typename... Args>
  queue(Args&&... args, frequency uncore_frequency, frequency core_frequency)
  {
    queue(std::forward<Args>(args)...);

    core_target_frequency = core_frequency;
    uncore_target_frequency = uncore_frequency;
  }

  // TODO: when should the frequency be changed?
  template <typename... Args>
  inline sycl::event submit(Args&&... args)
  {
    sycl::event event = sycl::queue::submit(std::forward<Args>(args)...);
    kernel k{event, core_target_frequency, uncore_target_frequency};

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

  frequency core_target_frequency = 0;
  frequency uncore_target_frequency = 0;
};

} // namespace synergy
