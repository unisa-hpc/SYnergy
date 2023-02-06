#include <future>
#include <thread>

#include <nvml.h>
#include <sycl/sycl.hpp>

#include "nvidia/profiling_nvidia.hpp"
#include "nvidia/utils.hpp"

namespace synergy {

profiling_nvidia::profiling_nvidia()
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

      kernel_energy += power * sampling_rate / 1000.0; // Get the integral of the power usage over the interval

      std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
    }

    energy_consumption += kernel_energy;
  };
}

profiling_nvidia::~profiling_nvidia()
{
  synergy_check_nvml(nvmlShutdown());
}

void profiling_nvidia::profile(sycl::event &e)
{
  auto &&res = std::async(std::launch::async, energy_function, e);
}

double profiling_nvidia::consumption()
{
  return energy_consumption;
}

} // namespace synergy