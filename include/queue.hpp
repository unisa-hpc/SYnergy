#pragma once

#include <sycl/sycl.hpp>

#include "kernel.hpp"
#include "profiling_manager.hpp"
#include "runtime.hpp"
#include "scaling.hpp"
#include "types.hpp"

namespace synergy {

class queue : public sycl::queue {
public:
#ifdef SYNERGY_ENABLE_PROFILING
  template <typename... Rest>
  queue(Rest&&... args)
      : sycl::queue(synergy::queue::check_args(std::forward<Rest>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())},
        profiling{std::make_shared<detail::profiling_manager>(device)} {
    assert_profiling_properties();
  }

  template <typename... Rest>
  queue(frequency uncore_frequency, frequency core_frequency, Rest&&... args)
      : sycl::queue(synergy::queue::check_args(std::forward<Rest>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())},
        profiling{std::make_shared<detail::profiling_manager>(device)},
        core_target_frequency{core_frequency},
        uncore_target_frequency{uncore_frequency} {
    assert_profiling_properties();
  }
#else
  template <typename... Rest>
  queue(Rest&&... args)
      : sycl::queue(synergy::queue::check_args(std::forward<Rest>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())} {}

  template <typename... Rest>
  queue(frequency uncore_frequency, frequency core_frequency, Rest&&... args)
      : sycl::queue(synergy::queue::check_args(std::forward<Rest>(args)...)),
        device{synergy::detail::runtime::synergy_device_from(get_device())},
        core_target_frequency{core_frequency},
        uncore_target_frequency{uncore_frequency} {}
#endif

#ifdef SYNERGY_ENABLE_PROFILING
  ~queue() {
    sycl::queue::wait();
  }
#endif

  // esplicitly declared to avoid clashes with the variadic constructor
  queue(queue&) = default;
  queue(const queue&) = default;
  queue(queue&&) = default;
  queue& operator=(const queue&) = default;
  queue& operator=(queue&) = default;
  queue& operator=(queue&&) = default;

  template <typename T, typename... Rest>
  sycl::event submit(T cfg, Rest&&... rest) {
    sycl::event event;

    if (has_target()) {
      sycl::event scaling_event = sycl::queue::submit(detail::device_scaling{device, core_target_frequency, uncore_target_frequency});

      event = sycl::queue::submit(
          [&](sycl::handler& h) {
            h.depends_on(scaling_event);
            cfg(h);
          },
          std::forward<Rest>(rest)...
      );
    } else {
      event = sycl::queue::submit(cfg, std::forward<Rest>(rest)...);
    }

#ifdef SYNERGY_ENABLE_PROFILING
    profiling->profile_kernel(event);
#endif

    return event;
  }

  template <typename T, typename... Rest>
  sycl::event submit(frequency uncore_frequency, frequency core_frequency, T cfg, Rest&&... rest) {
    sycl::event event;

    if (uncore_frequency != 0 && core_frequency != 0) {
      sycl::event scaling_event = sycl::queue::submit(detail::device_scaling{device, core_target_frequency, uncore_target_frequency});

      event = sycl::queue::submit(
          [&](sycl::handler& h) {
            h.depends_on(scaling_event);
            cfg(h);
          },
          std::forward<Rest>(rest)...
      );

    } else if (has_target()) {
      return submit(cfg, std::forward<Rest>(rest)...);
    } else {
      event = sycl::queue::submit(cfg, std::forward<Rest>(rest)...);
    }

#ifdef SYNERGY_ENABLE_PROFILING
    profiling->profile_kernel(event);
#endif

    return event;
  }

  inline device get_synergy_device() const {
    return device;
  }

  inline void set_target_frequencies(frequency uncore_frequency, frequency core_frequency) {
    core_target_frequency = core_frequency;
    uncore_target_frequency = uncore_frequency;
  }

#ifdef SYNERGY_ENABLE_PROFILING
  inline double kernel_energy_consumption(const sycl::event& event) const {
    return profiling->kernel_energy(event);
  }

  inline double device_energy_consumption() const {
    return profiling->device_energy();
  }
#endif

private:
  device device;
  frequency core_target_frequency = 0;
  frequency uncore_target_frequency = 0;

#ifdef SYNERGY_ENABLE_PROFILING
  std::shared_ptr<detail::profiling_manager> profiling;
#endif

  inline bool has_target() { return core_target_frequency != 0 && uncore_target_frequency != 0; }

  template <typename... Args>
  static sycl::queue check_args(Args&&... args) {
#ifdef SYNERGY_ENABLE_PROFILING
    // check if it has some standard property
    if constexpr (
        (std::is_same_v<sycl::property::queue::enable_profiling&, Args> || ...) ||
        (std::is_same_v<sycl::property::queue::enable_profiling, Args> || ...) ||
        (std::is_same_v<sycl::property::queue::in_order&, Args> || ...) ||
        (std::is_same_v<sycl::property::queue::in_order, Args> || ...) ||
        (std::is_same_v<sycl::property_list&, Args> || ...) ||
        (std::is_same_v<sycl::property_list, Args> || ...)
    )
      return sycl::queue(std::forward<Args>(args)...);
    else {
      return sycl::queue(std::forward<Args>(args)..., sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order {}});
    }
#else
    return sycl::queue(std::forward<Args>(args)...);
#endif
  }

#ifdef SYNERGY_ENABLE_PROFILING
  inline void assert_profiling_properties() {
    if (!has_property<sycl::property::queue::enable_profiling>())
      throw std::runtime_error("synergy::queue error: queue must be constructed with the enable_profiling property");
    if (!has_property<sycl::property::queue::in_order>())
      throw std::runtime_error("synergy::queue error: queue must be constructed with the in_order property");
  }
#endif
};

} // namespace synergy
