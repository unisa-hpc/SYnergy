#ifndef _SYNERGY_ENERGY_NVIDIA_H_
#define _SYNERGY_ENERGY_NVIDIA_H_

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <thread>
#include <utility>

#include "../energy_interface.hpp"
#include "utils.hpp"
#include <nvml.h>

namespace synergy {

class energy_nvidia : public energy_interface {
public:
  energy_nvidia()
  {
    synergy_check_nvml(nvmlInit());
    synergy_check_nvml(nvmlDeviceGetHandleByIndex(0, &device_handle));

    energy_function = [this](sycl::event e) {
      uint32_t power;
      double kernel_energy = 0.0;

      // Wait until start
#ifdef __HIPSYCL__
      e.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else
      while (e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
        ;
#endif

      while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
        synergy_check_nvml(nvmlDeviceGetPowerUsage(device_handle, &power));

        kernel_energy += power * intervals_length / 1000.0; // Get the integral of the power usage over the interval

        std::this_thread::sleep_for(std::chrono::milliseconds(intervals_length));
      }

      energy_consumption += kernel_energy;
    };
  }

  ~energy_nvidia()
  {
    synergy_check_nvml(nvmlShutdown());
  }

  void process(sycl::event &e)
  {
    auto &&res = std::async(std::launch::async, energy_function, e);
  }

  double consumption()
  {
    return energy_consumption;
  }

private:
  nvmlDevice_t device_handle;
  std::function<void(sycl::event)> energy_function;
  static constexpr int intervals = 100000;
  static constexpr int intervals_length = 15; // ms

  double energy_consumption = 0.0;
};

} // namespace synergy

#endif