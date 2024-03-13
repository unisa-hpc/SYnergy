#pragma once

#include "profiling_manager.hpp"
#ifdef SYNERGY_HOST_PROFILING
#include "host_profiler.hpp"
#endif
namespace synergy {

namespace detail {

template <typename Manager>
class concurrent_kernel_profiler {
public:
  concurrent_kernel_profiler(Manager& manager, kernel& kernel)
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
  sequential_kernel_profiler(Manager& manager, kernel& kernel, energy start_energy = 0)
      : manager{manager}, kernel{kernel}, start_energy{start_energy} {}

  void operator()() {
    synergy::device& device = manager.device;
    double energy_sample = start_energy;

#if defined(SYNERGY_USE_PROFILING_ENERGY) && (defined(SYNERGY_LZ_SUPPORT) || defined(SYNERGY_CUDA_SUPPORT))
    while (kernel.event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
      ;

    energy end = 0; 
    while( (end = device.get_energy_usage()) == start_energy)
      ;
    energy_sample = (end - energy_sample) / 1000000.0; // microjoules to joules
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
  energy start_energy;
};

template <typename Manager>
class device_profiler {
public:
  device_profiler(Manager& manager)
      : manager{manager} {}

  void operator()() {
    synergy::device& device = manager.device;

#if defined(SYNERGY_USE_PROFILING_ENERGY) && (defined(SYNERGY_LZ_SUPPORT) || defined(SYNERGY_CUDA_SUPPORT))
    auto e_start = device.get_energy_usage();

    while (!manager.finished.load(std::memory_order_acquire)) {
      auto e_end = device.get_energy_usage();
      manager.device_energy_consumption = (e_end - e_start) / 1000000.0; // microjoules to joules
    }
#else
    auto sampling_rate = device.get_power_sampling_rate();
    double energy_sample = 0.0;

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

#ifdef SYNERGY_HOST_PROFILING
template <typename Manager>
class host_device_profiler {
public:
  host_device_profiler(Manager& manager)
      : manager{manager} {}

  void operator()() {
    synergy::device& device = manager.device;
    auto eh_start = host_profiler::get_host_energy();

#if defined(SYNERGY_USE_PROFILING_ENERGY) && (defined(SYNERGY_LZ_SUPPORT) || defined(SYNERGY_CUDA_SUPPORT))
    auto ed_start = device.get_energy_usage();

    while (!manager.finished.load(std::memory_order_acquire)) {
      auto ed_end = device.get_energy_usage();
      manager.device_energy_consumption = (ed_end - ed_start) / 1000000.0; // microjoules to joules
      auto eh_end = host_profiler::get_host_energy();
      manager.host_energy_consumption = (eh_end - eh_start) / 1000000.0; // microjoules to joules
    }
#else
    auto sampling_rate = device.get_power_sampling_rate();
    double energy_sample = 0.0;

    while (!manager.finished.load(std::memory_order_acquire)) {
      energy_sample = device.get_power_usage() / 1000000.0 * sampling_rate / 1000; // Get the integral of the power usage over the interval
      manager.device_energy_consumption += energy_sample;
      auto eh_end = host_profiler::get_host_energy();

      manager.host_energy_consumption = (eh_end - eh_start) / 1000000.0; // microjoules to joules

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }
#endif
  }

private:
  /**
   * @brief Get the host energy value in microjoules
   */

  Manager& manager;
};
#endif

} // namespace detail

} // namespace synergy
