#pragma once

#include "profiling_manager.hpp"

namespace synergy {

namespace detail {

template <typename Manager>
class fine_grained_profiler {
public:
  fine_grained_profiler(Manager& manager, kernel& kernel)
      : manager{manager}, kernel{kernel} {}

  void operator()() {
    synergy::device& device = manager.device;
    auto sampling_rate = device.get_power_sampling_rate();

    double energy_sample = 0.0;
// Wait until start
#ifdef __HIPSYCL__
    kernel.event.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++ and on HIP with hipSYCL
#else
    while (kernel.event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted) // not working hipSYCL CUDA and HIP (infinite loop)
      ;
#endif

    while (kernel.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {

      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      // std::cout << "power: " << device.get_power_usage() << ", energy: " << energy_sample << "\n";
      kernel.energy += energy_sample;

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
  }

private:
  Manager& manager;
  kernel& kernel;
};

template <typename Manager>
class coarse_grained_profiler {
public:
  coarse_grained_profiler(Manager& manager)
      : manager{manager} {}

  void operator()() {
    synergy::device& device = manager.device;
    auto sampling_rate = device.get_power_sampling_rate();

    double energy_sample = 0.0;
    while (!manager.finished) {
      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      manager.device_energy_consumption += energy_sample;

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
  }

private:
  Manager& manager;
};

} // namespace detail

} // namespace synergy
