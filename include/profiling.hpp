#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "kernel.hpp"

namespace synergy {
namespace detail {
class profiler {
public:
  profiler(kernel& kernel, synergy::device device)
      : kernel{kernel}, device{device} {}

  void operator()() {
    auto sampling_rate = device.get_power_sampling_rate();
    double energy_sample = 0.0;

// Wait until start
#ifdef __HIPSYCL__
    kernel.event.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++ and on HIP with hipSYCL
#else
    while (kernel.event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted) // not working hipSYCL CUDA and HIP (infinite loop)
      ;
#endif

    // TODO: manage multiple kernel execution on the same queue
    while (kernel.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {

      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      // std::cout << "power: " << device.get_power_usage() << ", energy: " << energy_sample << "\n";

      kernel.energy += energy_sample;
      device.increase_energy_consumption(energy_sample);

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
  }

private:
  synergy::device device;
  kernel& kernel; // reference to the main thread kernel (beware of race conditions)
};

} // namespace detail

} // namespace synergy
