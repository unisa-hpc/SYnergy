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
class sequential_kernel_profiler {
public:
  sequential_kernel_profiler(Manager& manager, kernel& kernel)
      : manager{manager}, kernel{kernel} {}

  void operator()() {
    synergy::device& device = manager.device;
    double energy_sample = 0.0;
  
#ifdef SYNERGY_LZ_SUPPORT

    auto start = device.get_energy_usage();
    while (kernel.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
      ;
    auto end = device.get_energy_usage();
    auto sampling_time = kernel.event.get_profiling_info<sycl::info::event_profiling::command_end>() 
                      - kernel.event.get_profiling_info<sycl::info::event_profiling::command_start>(); //nanoseconds
    energy_sample = (end - start) / 1000000.0; // microjouls to jouls
    kernel.energy = energy_sample;
    
#else
    auto sampling_rate = device.get_power_sampling_rate();

    while (kernel.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {

      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      kernel.energy += energy_sample;

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
#endif
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

#ifdef SYNERGY_LZ_SUPPORT

    auto e_start = device.get_energy_usage();
    auto t_start = std::chrono::steady_clock::now();
    while (!manager.finished.load(std::memory_order_acquire));
    auto e_end = device.get_energy_usage();
    auto t_end = std::chrono::steady_clock::now();
    auto delta_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

    manager.device_energy_consumption = (e_end - e_start) / delta_time / 1000000.0; 
#else
    while (!manager.finished.load(std::memory_order_acquire)) {
      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      manager.device_energy_consumption += energy_sample;

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
#endif
  }

private:
  Manager& manager;
};

} // namespace detail

} // namespace synergy
