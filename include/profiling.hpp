#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "kernel.hpp"

namespace synergy {

class profiler {
public:
  profiler(synergy::kernel& kernel, std::shared_ptr<synergy::device> device)
      : kernel{kernel}, device{device} {}

  void operator()()
  {
    sycl::event event = kernel.event;
    // Wait until start
#ifdef __HIPSYCL__
    event.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else
    while (event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
      ;
#endif

    // TODO: manage multiple kernel execution on the same queue

    auto sampling_rate = device->get_power_sampling_rate();
    double energy_sample = 0.0;

    while (event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
      energy_sample = device->get_power_usage() * sampling_rate / 1000000.0; // Get the integral of the power usage over the interval

      kernel.energy += energy_sample;
      device->increase_energy_consumption(energy_sample);

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
  }

private:
  std::shared_ptr<synergy::device> device;
  synergy::kernel& kernel; // reference to the main thread kernel (beware of race conditions)
};

} // namespace synergy
