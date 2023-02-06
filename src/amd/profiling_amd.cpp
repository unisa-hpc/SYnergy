#include <future>
#include <thread>

#include <rocm_smi/rocm_smi.h>
#include <sycl/sycl.hpp>

#include "amd/profiling_amd.hpp"
#include "amd/utils.hpp"

namespace synergy {

profiling_amd::profiling_amd()
{
  synergy_check_rsmi(rsmi_init(0));
  energy_function = [this](sycl::event e) {
    rsmi_status_t rsmi_result;
    uint64_t power;
    double kernel_energy = 0.0;

    // Wait until start
#ifdef __HIPSYCL__
    e.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else
    while (e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
      ;
#endif

    while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
      synergy_check_rsmi(rsmi_dev_power_ave_get(device_handle, 0, &power));

      kernel_energy += power * intervals_length / 1000000.0; // Get the integral of the power usage over the interval

      std::this_thread::sleep_for(std::chrono::milliseconds(intervals_length));
    }

    energy_consumption += kernel_energy;
  };
}

profiling_amd::~profiling_amd()
{
  synergy_check_rsmi(rsmi_shut_down());
}

void profiling_amd::profile(sycl::event &e)
{
  auto &&res = std::async(std::launch::async, energy_function, e);
}

double profiling_amd::consumption()
{
  return energy_consumption;
}

} // namespace synergy
