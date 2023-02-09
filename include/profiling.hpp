#pragma once

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "queue.hpp"

namespace synergy {

template <typename vendor>
class profiler {
public:
  profiler(const queue& queue)
  {
    m_device = device;
  }

  void operator()(sycl::event event)
  {
    auto device = queue.get_synergy_device();

    double kernel_energy = 0.0;

    // Wait until start
#ifdef __HIPSYCL__
    e.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else
    while (e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
      ;
#endif

    while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {

      nvmlDeviceGetPowerUsage(device_handle, &power);

      kernel_energy += device.get_power_usage() * sampling_rate / 1000.0; // Get the integral of the power usage over the interval

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }

    // increase queue energy consumption
    // increase kernel energy consumption
  }

private:
  queue queue;
  static constexpr int sampling_rate = 15; // ms
};

} // namespace synergy
