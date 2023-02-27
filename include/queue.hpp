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
      : sycl::queue(synergy::queue::check_args(std::forward<Args>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())} {}

  template <typename... Args>
  queue(frequency uncore_frequency, frequency core_frequency, Args&&... args)
      : sycl::queue(synergy::queue::check_args(std::forward<Args>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())} {
    core_target_frequency = core_frequency;
    uncore_target_frequency = uncore_frequency;
    has_target = core_target_frequency != 0 && uncore_target_frequency != 0;
  }

  // esplicitly declared to avoid clashes with the variadic constructor
  queue(queue&) = default;
  queue(const queue&) = default;
  queue(queue&&) = default;
  queue& operator=(const queue&) = default;
  queue& operator=(queue&) = default;
  queue& operator=(queue&&) = default;

  template <typename T, typename... Args>
  inline sycl::event submit(T cfg, Args&&... args) {
    sycl::event event;

    if (has_target) {
      sycl::event scaling_event = sycl::queue::submit([&](sycl::handler& h) {
        try {
          device.set_all_frequencies(core_target_frequency, uncore_target_frequency);
        } catch (const std::exception& e) {
          std::cerr << e.what() << '\n';
        }
      });

      event = sycl::queue::submit(
          [&](sycl::handler& h) {
            h.depends_on(scaling_event);
            cfg(h);
          },
          std::forward<Args>(args)...
      );
    } else {
      event = sycl::queue::submit(cfg, std::forward<Args>(args)...);
    }

#ifdef SYNERGY_ENABLE_PROFILING
    detail::kernel k{event, core_target_frequency, uncore_target_frequency};
    auto async = std::async(
        std::launch::async,
        detail::profiler(kernels.insert({event, k}).first->second, device)
    );
#endif

    return event;
  }

  inline device get_synergy_device() const { return device; }

  inline void set_target_frequencies(frequency uncore_frequency, frequency core_frequency) {
    core_target_frequency = core_frequency;
    uncore_target_frequency = uncore_frequency;
  }

#ifdef SYNERGY_ENABLE_PROFILING
  inline double kernel_energy_consumption(sycl::event& event) const {
    auto search = kernels.find(event);
    if (search == kernels.end())
      throw std::runtime_error(
          "synergy::queue error: kernel was not submitted to the queue"
      );

    return search->second.energy;
  }
#endif

private:
  device device;
#ifdef SYNERGY_ENABLE_PROFILING
  std::unordered_map<sycl::event, detail::kernel> kernels;
#endif
  frequency core_target_frequency = 0;
  frequency uncore_target_frequency = 0;
  bool has_target = false;

  template <typename... Args>
  static sycl::queue check_args(Args&&... args) {
    if constexpr ((std::is_same_v<sycl::property::queue::enable_profiling, Args> ||
                   ...))
      return sycl::queue(std::forward<Args>(args)...);
    else
      return sycl::queue(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{});
  }
};

} // namespace synergy
