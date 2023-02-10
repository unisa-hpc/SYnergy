#pragma once

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "queue.hpp"

namespace synergy {

class profiler {
public:
  profiler(queue& queue)
      : queue{queue} {}

  void operator()(sycl::event& event)
  {
    auto device = queue.get_synergy_device();

    // Wait until start
#ifdef __HIPSYCL__
    event.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else
    while (event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
      ;
#endif

    // TODO: manage multiple kernel execution on the same queue

    auto sampling_rate = device.get_power_sampling_rate();
    double energy_sample = 0.0;
    double& kernel_energy = queue.kernels_energy.find(event)->second;

    while (event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
      energy_sample = device.get_power_usage() * sampling_rate / 1000000.0; // Get the integral of the power usage over the interval

      kernel_energy += energy_sample;
      queue.energy += energy_sample;

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
  }

private:
  queue& queue; // reference to the main thread queue (beware of race conditions)
};

} // namespace synergy
