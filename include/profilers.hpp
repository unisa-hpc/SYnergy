#pragma once

#include "profiling_manager.hpp"

namespace synergy {

namespace detail {

template <typename Manager>
class fine_grained_profiler {
public:
  fine_grained_profiler(Manager& manager)
      : manager{manager} {}

  void operator()() {
    synergy::device& device = manager.device;
    auto sampling_rate = device.get_power_sampling_rate();

    size_t current_kernel = 0;
    while (!manager.finished) {

      while (current_kernel < manager.kernels.size()) {
        double energy_sample = 0.0;
        kernel& kernel = manager.kernels[current_kernel];

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
          manager.device_energy_consumption += energy_sample;

          std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
        }

        current_kernel++;
      }
    }
  }

private:
  Manager& manager;
};

} // namespace detail

} // namespace synergy
